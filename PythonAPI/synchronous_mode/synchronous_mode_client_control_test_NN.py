#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import copy
import csv


try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass

import carla

import random
from collections import deque

try:
	import pygame
	from pygame.locals import * # for manual input
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
	import numpy as np
except ImportError:
	raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
	import queue
except ImportError:
	import Queue as queue
sys.path.insert(0,'/home/ruihan/UnrealEngine_4.22/carla/Dist/CARLA_0.9.5-428-g0ce908db/LinuxNoEditor/PythonAPI/carla')
# print(sys.path)

# for mpc_verify
from scipy import optimize
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev
import sympy as sym
from sympy.tensor.array import derive_by_array
sym.init_printing()
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.set_default_dtype(torch.float32)

from model_predicative_control_new import MPCController
from agents.navigation.my_basic_agent import BasicAgent
from agents.navigation.my_local_planner import _retrieve_options, RoadOption
from agents.navigation.my_local_planner import *
from agents.tools.misc import distance_vehicle, draw_waypoints
from NN_controller import mlp
from NN_controller_img import MyCoil
from ilqr.ilqr import ILQRController

from model_predicative_control_new import _EqualityConstraints


# set MACROS for data normalization
max_pos_val = 500
max_yaw_val = 180
max_speed_val = 40

class CarlaSyncMode(object):
	"""
	Context manager to synchronize output from different sensors. Synchronous
	mode is enabled as long as we are inside this context

		with CarlaSyncMode(world, sensors) as sync_mode:
			while True:
				data = sync_mode.tick(timeout=1.0)

	"""

	def __init__(self, world, *sensors, **kwargs):
		self.world = world
		self.sensors = sensors
		self.frame = None
		self.delta_seconds = 1.0 / kwargs.get('fps', 20)
		self._queues = []
		self._settings = None

	def __enter__(self):
		self._settings = self.world.get_settings()
		self.frame = self.world.apply_settings(carla.WorldSettings(
			no_rendering_mode=False,
			synchronous_mode=True,
			fixed_delta_seconds=self.delta_seconds))

		def make_queue(register_event):
			q = queue.Queue()
			register_event(q.put)
			self._queues.append(q)

		make_queue(self.world.on_tick)
		for sensor in self.sensors:
			make_queue(sensor.listen)
		return self

	def tick(self, timeout):
		self.frame = self.world.tick()
		data = [self._retrieve_data(q, timeout) for q in self._queues]
		assert all(x.frame == self.frame for x in data)
		return data

	def __exit__(self, *args, **kwargs):
		self.world.apply_settings(self._settings)

	def _retrieve_data(self, sensor_queue, timeout):
		while True:
			data = sensor_queue.get(timeout=timeout)
			if data.frame == self.frame:
				return data


def draw_image(surface, image, blend=False):
	array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
	array = np.reshape(array, (image.height, image.width, 4))
	array = array[:, :, :3]
	array = array[:, :, ::-1]
	image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
	if blend:
		image_surface.set_alpha(100)
	surface.blit(image_surface, (0, 0))


def get_font():
	fonts = [x for x in pygame.font.get_fonts()]
	default_font = 'ubuntumono'
	font = default_font if default_font in fonts else fonts[0]
	font = pygame.font.match_font(font)
	return pygame.font.Font(font, 14)


def should_quit():
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			return True
		elif event.type == pygame.KEYUP:
			if event.key == pygame.K_ESCAPE:
				return True
	return False

# =================================
# RH: customized methods
# ===============================
class AttrDict(dict):
	def __init__(self, *args, **kwargs):
		super(AttrDict, self).__init__(*args, **kwargs)
		self.__dict__ = self

def get_mpc_control(world, vehicle, m):
	# TODO: add more params in the dict, 
	# which can be automatically converted (to replace args)
	params_dict = {'target_speed': 30}
	params = AttrDict(params_dict)
	controller = MPCController(params.target_speed)

	# get current "measurements"
	t = vehicle.get_transform()
	v = vehicle.get_velocity()
	c = vehicle.get_control()
	measurements_dict = {"v": v,
						 "t": t} # TODO: create a dict that return the world&vehicle data similar as 0.8 API
	measurements = AttrDict(measurements_dict)
	
	cur_wp = m.get_waypoint(t.location)

	local_interval = 0.5
	horizon = 5
	# initiate a series of waypoints
	future_wps = []
	future_wps.append(cur_wp)

	for i in range(horizon):
		# TODO: check whether "next" works here
		future_wps.append(random.choice(future_wps[-1].next(local_interval)))

	one_log_dict = controller.control(future_wps, measurements)
	# print("one_log_dict in run_carla_client")
	# print(one_log_dict)
	control = carla.VehicleControl()
	control.throttle, control.steer = one_log_dict['throttle'], one_log_dict['steer']

	return control
	

def reach_destiny(destiny_loc, vehicle):
	veh_loc = vehicle.get_transform().location
	dist_vec = np.array([destiny_loc.x-veh_loc.x, destiny_loc.y-veh_loc.y])
	# print("dist", dist_vec, np.linalg.norm(dist_vec))
	return np.linalg.norm(dist_vec)


def transform_to_arr(tf):
	return np.array([tf.location.x, tf.location.y, tf.location.z, tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll])


def compute_target_waypoint(last_waypoint, target_speed):

	local_sampling_radius = 0.5
	sampling_radius = target_speed/3.6 # km/h -> m/s
	sampling_radius = local_sampling_radius
	next_waypoints = list(last_waypoint.next(sampling_radius))

	if len(next_waypoints) == 1:
		# only one option available ==> lanefollowing
		target_waypoint = next_waypoints[0]
		road_option = RoadOption.LANEFOLLOW
	else:
		# random choice between the possible options
		road_options_list = _retrieve_options(
			next_waypoints, last_waypoint)
		if RoadOption.LANEFOLLOW in road_options_list:
			road_option = RoadOption.LANEFOLLOW 
		else: 
			road_option = random.choice(road_options_list)
		target_waypoint = next_waypoints[road_options_list.index(
			road_option)]
	# print("road_option", road_option)
	return target_waypoint, road_option


def get_nn_controller(state, model, device):
	model = model.to(device)
	state = torch.from_numpy(state).float().to(device)
	state = state.view(1, -1)

	# print("state", state.size(), state)
	output = model(state) #RuntimeError: size mismatch, m1: [10 x 1], m2: [10 x 100] at /pytorch/aten/src/TH/generic/THTensorMath.cpp:961
	output = output.data.cpu().numpy()[0]
	print("output from nn_controller", output)
	control = carla.VehicleControl()
	control.throttle, control.steer = output[0].item(), output[1].item()
	# control = carla.VehicleControl(throttle=output[0], steer=output[1])
	return control


# TODO: write state parsing to be compatible with NN_controller_img, checking needed
def get_nn_img_controller(array, state, model, device, img_height, img_width):
	model = model.to(device)
	img = array.astype(np.float)
	img = torch.from_numpy(img).type(torch.FloatTensor)
	img = img / 255.
	img = img.view(img.shape[0], -1, img_height, img_width)
	state = torch.from_numpy(state).float().to(device)
	state = state.view(1, -1)

	output = model(img, m)
	output = output.data.cpu().numpy()[0]
	print("output from nn_controller", output)
	control = carla.VehicleControl()
	control.throttle, control.steer = output[0].item(), output[1].item()
	# control = carla.VehicleControl(throttle=output[0], steer=output[1])
	return control


def wait():
	while True:
		for event in pygame.event.get():
			if event.type == QUIT:
				pygame.quit()
				sys.exit()
			if event.type == KEYDOWN:
				return


def spawn_trolley(world, blueprint_library, x=0, y=0, z=8):
	 # spawn a trolley to visualilze target_waypoint
	trolley_bp = random.choice(blueprint_library.filter('static.prop.shoppingtrolley')) # vehicle.toyota.prius static.prop.shoppingtrolley
	trolley_tf = carla.Transform(location=carla.Location(x=x, y=y, z=z))
	# print("trolley_bp", trolley_bp, trolley_tf.location)
	trolley = world.spawn_actor(trolley_bp, trolley_tf)
	trolley.set_simulate_physics(False)
	return trolley


def mpc_verify(world, blueprint_library, model, device, map, cur_state, control, pred_traj=[], num_wps=10, verify_horizon=20, target_speed=30):
	# return two values (if_safe, unsafe timestep)
	print("mpc_verify")
	# use bicycle model to predict next step
	# state vector (x, y, v, yaw)
	# initialilze arrays
	x = np.zeros(verify_horizon)
	y = np.zeros(verify_horizon)
	v = np.zeros(verify_horizon)
	yaw = np.zeros(verify_horizon)
	# print("cur_state")
	# print(cur_state)
	x[0], y[0], v[0], yaw[0] = cur_state[1], cur_state[2],cur_state[0], cur_state[3]
	# copy one row multiple times to keep z value
	# state_arr = np.tile(cur_state, (verify_horizon+1,1))
	z = cur_state[3]
	dt = 0.05
	Lf = 1
	LANE_WIDTH = 3.5 # TODO: check again later
	SAFE_DIST = LANE_WIDTH/2.
	# use numpy for now, switch to sympy later if necessary
	for t in range(1, verify_horizon):
		# compute next state based on bicycle model
		x[t] = x[t-1] + v[t-1]*np.cos(yaw[t-1])*dt
		y[t] = y[t-1] + v[t-1]*np.cos(yaw[t-1])*dt
		yaw[t] = yaw[t-1] - v[t-1] * control.steer/Lf*dt
		v[t] = v[t-1] + control.throttle*dt
		# calculate the dist to the closest waypoint projected on the road center (assume)
		# TODO
		# print("predicted next state")
		next_loc = carla.Location(x=x[t], y=y[t], z=z)
		# print(next_loc)
		# find perpendicular waypoint
		# print("next waypoint")
		pp_wp = map.get_waypoint(next_loc)
		# print(pp_wp)
		# Method 1: calculate pp_wp to wp at the center
		dist = np.sqrt((pp_wp.transform.location.x - next_loc.x)**2 + (pp_wp.transform.location.y - next_loc.y)**2)
		
		print("dist", dist)
		if dist > SAFE_DIST:
			return False, t, pred_traj
		# if safe, call NN controller to get next action
		last_waypoint = map.get_waypoint(carla.Location(x=x[t], y=y[t]))
		target_waypoints = []
		for k in range(num_wps):
			target_waypoint, road_option = compute_target_waypoint(last_waypoint, target_speed)
			target_waypoints.append([target_waypoint, road_option])
			last_waypoint = target_waypoint

		target_waypoints_np = []
		for wp_ro in target_waypoints:
			target_waypoints_np.append(transform_to_arr(wp_ro[0].transform))
		target_waypoints_np = np.array([target_waypoints_np]).flatten()

		# unnormalized = np.array([v[t], x[t], y[t], yaw[t]])
		state = np.array([v[t]/max_speed_val, x[t]/max_pos_val, y[t]/max_pos_val, yaw[t]/max_yaw_val, target_speed/max_speed_val])

		for j in range(num_wps): # concatenate x, y, yaw of future_wps
			state = np.hstack((state, target_waypoints_np[j*6]/max_pos_val, target_waypoints_np[1+j*6]/max_pos_val, target_waypoints_np[4+j*6]/max_yaw_val))
		# print("state in mpc_verify", state.shape)
		control = get_nn_controller(state, model, device)
		pred_traj.append(spawn_trolley(world, blueprint_library, x=x[t], y=y[t]))

	return True, 0, pred_traj



class verification():

	def __init__(self, init_state, control, horizon, target_waypoints_np):
		# init_state format [x_init, y_init. v_init, yaw_init]
		self.num_state = len(init_state)
		self.num_action = 2
		# concatenate state and action
		self.init_state = np.hstack((np.array(init_state), np.array([control.throttle, control.steer])))
		self.dt = 0.05
		self.Lf = 1
		self.horizon = horizon
		self.target_waypoints_np = target_waypoints_np	
		self.bounds = (
			(self.horizon+1) * (4 * [(None, None)]
			+ [(0,1)]
			+ [(-1, 1)]) + [(0, None)]
			)
		
		print("bounds", len(self.bounds))


	def func(self, sign=-1.0, lane_width=3.5):
		""" Objective function """
		# return lambda x: sign*(lane_width/2 - x[-1])
		return lambda x:-x[-1]**2


	def func_deriv(self, x, sign=-1.0):
		""" derivative of objective func """
		return np.array([1])


	# set dynamics model eq constrains
	def dyn_eq_cons_arr(self, x, t):
		# state_arr = x[: -1].reshape(((self.num_state+self.num_action), -1))
		# (self.num_state+self.num_action)*t+0
		return np.array([
		x[(self.num_state+self.num_action)*(t+1)+0] - (x[(self.num_state+self.num_action)*t+0] + x[(self.num_state+self.num_action)*t+2]*np.cos(x[(self.num_state+self.num_action)*t+3])*self.dt),
		x[(self.num_state+self.num_action)*(t+1)+1] - (x[(self.num_state+self.num_action)*t+1] + x[(self.num_state+self.num_action)*t+2]*np.cos(x[(self.num_state+self.num_action)*t+3])*self.dt),
		x[(self.num_state+self.num_action)*(t+1)+2] - (x[(self.num_state+self.num_action)*t+2] - x[(self.num_state+self.num_action)*t+2] * x[(self.num_state+self.num_action)*t+5] / self.Lf*self.dt),
		x[(self.num_state+self.num_action)*(t+1)+3] - (x[(self.num_state+self.num_action)*t+3] + x[(self.num_state+self.num_action)*t+4]*self.dt)
		])


	# set safety ineq constrains
	def safety_ineq_cons_arr(self, x, t):
		# state_arr = x[: -1].reshape(((self.num_state+self.num_action), -1))
		num_wps = len(self.target_waypoints_np)/6
		# print("look {} steps ahead for safety ineq cons".format(num_wps))
		inequ_cons_arr = []
		for i in range(int(num_wps)):
			wp_loc = np.array([self.target_waypoints_np[i*6], self.target_waypoints_np[i*6+1]])
			dist = np.array([x[(self.num_state+self.num_action)*t+0], x[(self.num_state+self.num_action)*t+1]]) - wp_loc
			dist_sq = dist[0]**2 + dist[1]**2
			# print("wp_loc", wp_loc)
			# print("state", np.array([x[(self.num_state+self.num_action)*t+0], x[(self.num_state+self.num_action)*t+1]]))
			# print("dist_sq", dist_sq)
			# print("s_sq", x[-1]**2)
			inequ_cons_arr.append( dist_sq- x[-1]**2)
		return np.array(inequ_cons_arr)

	def x_next(self, x, t):
		return x[(self.num_state+self.num_action)*(t+1)+0] - (x[(self.num_state+self.num_action)*t+0] + x[(self.num_state+self.num_action)*t+2]*np.cos(x[(self.num_state+self.num_action)*t+3])*self.dt)
	
	def y_next(self, x, t):
		return x[(self.num_state+self.num_action)*(t+1)+1] - (x[(self.num_state+self.num_action)*t+1] + x[(self.num_state+self.num_action)*t+2]*np.cos(x[(self.num_state+self.num_action)*t+3])*self.dt)

	def v_next(self, x, t):
		return x[(self.num_state+self.num_action)*(t+1)+2] - (x[(self.num_state+self.num_action)*t+2] - x[(self.num_state+self.num_action)*t+2] * x[(self.num_state+self.num_action)*t+5] / self.Lf*self.dt)

	def yaw_next(self, x, t):
		return x[(self.num_state+self.num_action)*(t+1)+3] - (x[(self.num_state+self.num_action)*t+3] + x[(self.num_state+self.num_action)*t+4]*self.dt)

	def dist(self, x, t, wp_loc):
		dist = np.array([x[(self.num_state+self.num_action)*t+0], x[(self.num_state+self.num_action)*t+1]]) - wp_loc
		dist_sq = dist[0]**2 + dist[1]**2
		return dist_sq- x[-1]**2


	def x_jac(self, x, t):
		eps = np.sqrt(np.finfo(float).eps)
		x_jac = optimize.approx_fprime(x, self.x_next, [eps], t)
		print("x_jac", x_jac)
		return x_jac


	def y_jac(self, x, t):
		eps = np.sqrt(np.finfo(float).eps)
		y_jac = optimize.approx_fprime(x, self.y_next, [eps], t)
		print("y_jac", y_jac)
		return y_jac


	def v_jac(self, x, t):
		eps = np.sqrt(np.finfo(float).eps)
		v_jac = optimize.approx_fprime(x, self.v_next, [eps], t)
		print("v_jac", v_jac)
		return v_jac

	def yaw_jac(self, x, t):
		eps = np.sqrt(np.finfo(float).eps)
		yaw_jac = optimize.approx_fprime(x, self.yaw_next, [eps], t)
		print("yaw_jac", yaw_jac)
		return yaw_jac

	def get_all_constraints(self):
		print("get_all_constraints")
		eps = np.sqrt(np.finfo(float).eps)

		cons = []
		# state_arr = lambda x: np.fromfunction(lambda i, j: x[i*(self.horizon+1)+j], (self.num_state+self.num_action))

		# set init state constrains
		print("num_state", self.num_state, self.init_state)
		for i in range(self.num_state+self.num_action):
			print("i", i)
			cons.append({'type': 'eq',
						 'fun': lambda x: np.array([x[i] - self.init_state[i]])})

		print("add other states constraints")
		# set constrains for each following state
		for t in range(self.horizon):

			# dissect the array
			# for each element inside, calculate its jacobian and append it to cons



			# print("append equlity constriant")

			cons.append({'type': 'eq',
						 'fun': lambda x: self.x_next(x, t),
						 'jac': lambda x: self.x_jac(x, t)})

			cons.append({'type': 'eq',
						 'fun': lambda x: self.y_next(x, t),
						 'jac': lambda x: self.y_jac(x, t)})
			
			cons.append({'type': 'eq',
						 'fun': lambda x: self.v_next(x, t),
						 'jac': lambda x: self.v_jac(x, t)})
			
			cons.append({'type': 'eq',
						 'fun': lambda x: self.yaw_next(x, t),
						 'jac': lambda x: self.yaw_jac(x, t)})


			# # print("append inequality constraints")
			# num_wps = len(self.target_waypoints_np)/6
			# for j in range(int(num_wps)):
			# 	wp_loc = np.array([self.target_waypoints_np[j*6], self.target_waypoints_np[j*6+1]])
			# 	cons.append({'type': 'ineq',
			# 				 'fun': lambda x: self.dist(x, t, wp_loc),
			# 				 'jac': lambda x: optimize.approx_fprime(x, self.dist, [eps], t, wp_loc)})

		print("return cons")



			# for i in range(len(lambda x: self.dyn_eq_cons_arr(x, t))):
			# 	cons.append({'type': 'eq',
			# 				 'fun': lambda x: self.dyn_eq_cons_arr(x, t)[i],
			# 				 'jac': lambda x: optimize.approx_fprime(x, dyn_eq_cons, eps)})

			# for i in range(len(lambda x: self.safety_ineq_cons_arr(x, t))):
			# 	cons.append({'type': 'ineq',
			# 				 'fun': lambda x: self.safety_ineq_cons_arr(x, t)[i],
			# 				 'jac': lambda x: optimize.approx_fprime(x, safety_ineq_cons, eps)})				

			# cons.append({'type': 'eq',
			# 			 'fun': lambda x: self.dyn_eq_cons_arr(x, t)})

			# cons.append({'type': 'ineq',
			# 			 'fun': lambda x: self.safety_ineq_cons_arr(x, t)})

		# # set constrain for distance >= 0
		# print("append dist > 0 const")
		# cons.append({'type': 'ineq',
		# 			 'fun': lambda x: np.array([x[-1]-0])})

		return cons


	def cost(self, x):
		cost = 0
		coeff = 0.01
		for t in range(self.horizon):
			cost += coeff* (x[(self.num_state+self.num_action)*(t+1)+4] - x[(self.num_state+self.num_action)*(t)+4])**2
		print("calculating cost", cost)
		return cost


	def optimization(self):
		print("in optimization")
		print("init_state", self.init_state)
		x = np.zeros((self.num_state + self.num_action)* (self.horizon+1)+1) # shape(127,)
		init_guess = np.concatenate((self.init_state[:2], np.array([0, 0.5, 1, 0]))) # concatenate zero control so that it satisfies dyn eq constraints
		x_guess = np.tile(init_guess, self.horizon+1)
		x_guess = np.hstack((x_guess, np.array([0])))
		print("x_guess", x_guess.shape)
		print(x_guess[:12])

		# x_guess = np.hstack((np.zeros((self.num_state + self.num_action)* (self.horizon+1)), np.array([0])))
		# pass the init-state and target_waypoints_np to the constraint func by "self"
		sign=-1.0
		lane_width=3.5
		""" Objective function """
		print("obj func")
		# result = minimize(self.func(), x_guess, constraints=self.get_all_constraints(), method='SLSQP', bounds= self.bounds, options={'disp':True})
		result = minimize(lambda x: self.cost(x), x_guess, constraints=self.get_all_constraints(), method='SLSQP', bounds= self.bounds, options={'disp':True})

		return result


def mpc_verify_2(world, blueprint_library, model, device, map, cur_state, control, pred_traj=[], num_wps=10, horizon=20, target_speed=30):
	# Method 2: return OptimizeResult
	# x[0], y[0], v[0], yaw[0] = cur_state[1], cur_state[2],cur_state[0], cur_state[3]
	# init_state format [x_init, y_init. v_init, yaw_init]
	print("enter mpc_verify_2")
	init_state = np.array([cur_state[1], cur_state[2],cur_state[0], cur_state[3]])
	last_waypoint = map.get_waypoint(carla.Location(x=init_state[0], y=init_state[1]))
	target_waypoints = []
	for k in range(num_wps):
		target_waypoint, road_option = compute_target_waypoint(last_waypoint, target_speed)
		target_waypoints.append([target_waypoint, road_option])
		last_waypoint = target_waypoint

	target_waypoints_np = []
	for wp_ro in target_waypoints:
		target_waypoints_np.append(transform_to_arr(wp_ro[0].transform))
	target_waypoints_np = np.array([target_waypoints_np]).flatten()
	print("target_waypoints_np")
	print(target_waypoints_np)
	print("init_state")
	print(init_state)
	ver = verification(init_state, control, horizon, target_waypoints_np)
	return ver.optimization()


# ===============================
# main function
# ===============================

def main():
	pygame.init()
	# init font for text message
	pygame.font.init()
	# myfont = pygame.font.SysFont('Comic Sans MS', 30) # see below
	hud_dim = [800, 600]  # default: 800, 600 # collect data: 200, 88
	display = pygame.display.set_mode(
		(hud_dim[0], hud_dim[1]),
		pygame.HWSURFACE | pygame.DOUBLEBUF)
	font = get_font()
	clock = pygame.time.Clock()

	client = carla.Client('localhost', 2000)
	client.set_timeout(2.0)

	world = client.get_world()
	world.set_weather(carla.WeatherParameters.ClearNoon)
	blueprint_library = world.get_blueprint_library()
	# actor_list = world.get_actors() # can get actors like traffic lights, stop signs, and spectator
	global_actor_list = []
	save_dir_base = 'data/testing/'
	#MLP_model_path = "/home/depend/workspace/carla/PythonAPI/synchronous_mode/models/mlp/dest_start_SR0.5_models/mlp_model_nx=8_wps10_lr0.001_bs32_optimSGD_ep1000.pth"
	MLP_model_path = "/home/depend/workspace/carla/PythonAPI/synchronous_mode/models/mlp/dest_start_merge_models/mlp_dict_nx=8_wps5_lr0.001_bs32_optimSGD_ep100_ep200_ep300_ep400_ep500_ep600_ep700_ep800_ep900_ep1000.pth"



	# "/home/ruihan/UnrealEngine_4.22/carla/Dist/CARLA_0.9.5-428-g0ce908db/LinuxNoEditor/PythonAPI/synchronous_mode/models/mlp/dest_start_merge_models/mlp_model_nx=8_wps10_lr0.001_bs32_optimSGD_ep100_ep200_ep300_ep400_ep500_ep600_ep700_ep800_ep900_ep1000.pth"
	# "/home/ruihan/UnrealEngine_4.22/carla/Dist/CARLA_0.9.5-428-g0ce908db/LinuxNoEditor/PythonAPI/synchronous_mode/models/mlp/dest_start_merge_models/mlp_model_nx=8_wps5_lr0.001_bs32_optimSGD_ep100_ep200_ep300_ep400_ep500_ep600_ep700_ep800_ep900_ep1000.pth"
	# "/home/ruihan/UnrealEngine_4.22/carla/Dist/CARLA_0.9.5-428-g0ce908db/LinuxNoEditor/PythonAPI/synchronous_mode/models/mlp/dest_start_merge_models/mlp_model_nx=8_wps10_lr0.001_bs32_optimSGD_ep100_ep200_ep300_ep400_ep500_ep600_ep700_ep800_ep900_ep1000.pth"
	# "/home/ruihan/UnrealEngine_4.22/carla/Dist/CARLA_0.9.5-428-g0ce908db/LinuxNoEditor/PythonAPI/synchronous_mode/models/mlp/mlp_img_model_nx=8_wps10_lr0.001_bs64_optimAdam_ep1000.pth"
	model = torch.load(MLP_model_path)
	checkpoint = torch.load(MLP_model_path)
	model = mlp(nx = 5 + 3 * 5, ny = 2)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	try:
		m = world.get_map()
		spawn_points = m.get_spawn_points()
		print("total number of spawn_points", len(spawn_points))

		# TODO: randomly choose spawn_point and destiny to test performance on new routes
		i = random.randint(0, len(spawn_points))
		j = random.randint(0, len(spawn_points))
		while j == i:
			j = random.randint(0, len(spawn_points))
		
		i = 3
		j = 0# 2
		save_dir = os.path.join(save_dir_base, 'data_{}_{}/'.format(i, j))
		destiny = spawn_points[i]
		print(j, "car destiny", destiny.location)
		start_pose = spawn_points[j]
		print(i, "car start_pose", start_pose.location)

		# TODO: consider create a class of agent and use planner as in my_local_planner
		# waypoints_queue = deque(maxlen=20000)

		# use record and replay API
		# check /home/ruihan/.config/Epic/CarlaUE4/Saved path
		# client.start_recorder("record_testing_dest_{}_start_{}_wps10_1000.log".format(i, j))

		# set and spawn actors
		actor_list = []
		vehicle_bp = random.choice(blueprint_library.filter('vehicle.lincoln.mkz2017'))
		vehicle = world.spawn_actor(vehicle_bp, start_pose)
		vehicle.set_simulate_physics(True)
		actor_list.append(vehicle)

		# common camera locations:
		# x=1.6, z=1.7 for front
		# x=-5.5, z=2.8 for back

		camera_rgb_bp = blueprint_library.find('sensor.camera.rgb')
		camera_rgb_bp.set_attribute('image_size_x', str(hud_dim[0]))
		camera_rgb_bp.set_attribute('image_size_y', str(hud_dim[1]))

		camera_rgb = world.spawn_actor(
			camera_rgb_bp,
			carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
			attach_to=vehicle)
		actor_list.append(camera_rgb)


		camera_semseg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
		camera_semseg_bp.set_attribute('image_size_x', str(hud_dim[0]))
		camera_semseg_bp.set_attribute('image_size_y', str(hud_dim[1]))

		camera_semseg = world.spawn_actor(
			camera_semseg_bp,
			carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
			attach_to=vehicle)
		actor_list.append(camera_semseg)

		# set PD control to the vehicle, fine here as long as you don't use vehicle_agent.run_step
		target_speed = 30
		vehicle_agent = BasicAgent(vehicle, target_speed=target_speed)
		destiny_loc = destiny.location
		vehicle_agent.set_destination((destiny_loc.x, destiny_loc.y, destiny_loc.z))
		# vehicle.set_autopilot(True)

		print("local actor list", actor_list)

		img_width = 200
		img_height = 88
		num_wps = 5
		MIN_DISTANCE_PERCENTAGE = 0.9
		min_distance = target_speed/3.6*MIN_DISTANCE_PERCENTAGE
		target_waypoints = deque(maxlen=num_wps)
		safety_checking = False

		# Create a synchronous mode context.
		with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=20) as sync_mode:
			
			control = vehicle.get_control()

			while reach_destiny(destiny_loc, vehicle)>10:
				if should_quit():
					print('destroying local actors.')
					for actor in actor_list:
						actor.destroy()
					return
				clock.tick(20)

				t = vehicle.get_transform()
				v = vehicle.get_velocity()

				# Instead of using BasicAgent, query target_waypoint and feed in NN controller
				last_waypoint = m.get_waypoint(t.location)
				
				# Method 1. query the target-waypoints based on current location of each frame
				# # query target_waypoints based on current location
				# target_waypoints = []
				# for k in range(num_wps):
				# 	target_waypoint, road_option = compute_target_waypoint(last_waypoint, target_speed)
				# 	target_waypoints.append([target_waypoint, road_option])
				# 	last_waypoint = target_waypoint

				# target_waypoints_np = []
				# for wp_ro in target_waypoints:
				# 	target_waypoints_np.append(transform_to_arr(wp_ro[0].transform))
				# target_waypoints_np = np.array([target_waypoints_np]).flatten()

				# Method 2. Use target_waypoints buffer and pop out once reached
				# check if reach, pop out from the list (purge the queue of obsolete waypoints)
				print("before purge", len(target_waypoints))
				max_index = -1
				# print(list(enumerate(target_waypoints)))
				for num, waypoint in enumerate(target_waypoints):
					if distance_vehicle(waypoint, t) < min_distance:
						max_index = num
				if max_index >= 0:
					for num in range(max_index+1):
						target_waypoints.popleft()
				print("after purge", len(target_waypoints))

				for k in range(len(target_waypoints), num_wps):
					# append waypoint one step ahead
					target_waypoint, road_option = compute_target_waypoint(last_waypoint, target_speed)
					target_waypoints.append(target_waypoint)
					last_waypoint = target_waypoint
					# spawn a trolley to visualilze target_waypoint
					# actor_list.append(spawn_trolley(world, blueprint_library, x=target_waypoint.transform.location.x, y=target_waypoint.transform.location.y))
				print("after refill", len(target_waypoints))

				target_waypoints_np = []
				for wp_ro in target_waypoints:
					target_waypoints_np.append(transform_to_arr(wp_ro.transform))
				target_waypoints_np = np.array([target_waypoints_np]).flatten()

				cur_speed = np.linalg.norm(np.array([v.x, v.y]))
				# save the long state for data collection
				# state = np.hstack((cur_speed, transform_to_arr(t), target_speed, transform_to_arr(target_waypoint.transform)))
				full_state = np.hstack((cur_speed, transform_to_arr(t), target_speed, target_waypoints_np)).flatten() # shape(188,)
				# print("full state")
				# print(full_state)
				# parse the state for NN testing
				unnormalized = np.hstack((full_state[0:3], full_state[5], full_state[7]))
				state = np.hstack((full_state[0]/max_speed_val, full_state[1]/max_pos_val, full_state[2]/max_pos_val, full_state[5]/max_yaw_val, full_state[7]/max_speed_val))

				for j in range(num_wps): # concatenate x, y, yaw of future_wps
					unnormalized = np.hstack((unnormalized, full_state[8+j*6], full_state[9+j*6], full_state[12+j*6]))
					state = np.hstack((state, full_state[8+j*6]/max_pos_val, full_state[9+j*6]/max_pos_val, full_state[12+j*6]/max_yaw_val))
					# actor_list.append(spawn_trolley(world, blueprint_library, x=full_state[8+j*6], y=full_state[9+j*6], z=10))

				# print("unnormalized")
				# print(unnormalized)
				# # calculate the relative coordinates
				# state[1] = state[1] + spawn_points[0].location.x - start_pose.location.x
				# state[5] = state[5] + spawn_points[0].location.x - start_pose.location.x
				# state[2] = state[2] + spawn_points[0].location.y - start_pose.location.y
				# state[6] = state[6] + spawn_points[0].location.y - start_pose.location.y
				
				# Advance the simulation and wait for the data.
				snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=5.0) # extend timeout=2.0
				image_semseg.convert(carla.ColorConverter.CityScapesPalette)
				fps = round(1.0 / snapshot.timestamp.delta_seconds)
				frame_number = image_semseg.frame_number
				# save the data
				# image_semseg.save_to_disk(save_dir+'{:08d}'.format(frame_number))
				
				if '_img_' in MLP_model_path:
					# process the img data as a tensor (without saving and reading in again), see _parse_image in manual_control.py
					print("process img online")
					array = np.frombuffer(image_semseg.raw_data, dtype=np.dtype("uint8"))
					array = np.reshape(array, (image_semseg.height, image_semseg.width, 4))
					array = array[:, :, :3]
					array = array[:, :, ::-1]
					print(array.shape)
					control = get_nn_img_controller(array, state, model, device, img_height, img_width)

				else:
					control = get_nn_controller(state, model, device)
				
				pred_traj = []
				if safety_checking:
					# check if the traj is safe
					# safe, unsafe_time, pred_traj = mpc_verify(world, blueprint_library, model, device, m, full_state, control, pred_traj)
					result = mpc_verify_2(world, blueprint_library, model, device, m, full_state, control, pred_traj)

					print("message")
					print(result.message)
					print("output")
					print(result.x)

					if result.success:
						print("NN control is safe", control.throttle, control.steer)
					else:
						print("NN is unsafe, use MPC instead")
						#TODO
						# control = compute_mpc_safe_control()



					# if not safe:
					# 	print("mpc predicts unsafe traj", unsafe_time)
					# 	textsurface = font.render('mpc predicts unsafe traj at {} step'.format(unsafe_time), False, (255, 255, 255))
					# 	# output most conservative action
					# 	DISCOUNTED_RATIO = 1.5
					# 	control.throttle, control.steer = control.throttle/DISCOUNTED_RATIO, control.steer/DISCOUNTED_RATIO
					# 	# raise ValueError("stop here")
					# else:
					# 	print("safe continue")
					# 	textsurface = font.render('Safe. Continue', False, (255, 255, 255))

				print("control", control.throttle, control.steer)
				vehicle.apply_control(control)
				# save the data
				# path = save_dir+'{:08d}_ctv'.format(frame_number) 
				# x = np.hstack((np.array([control.throttle, control.steer]), state))
				# print("control and measurement", x)
				# save in two formats, one separate to work with img, one appended to csv file
				# np.save(path, x)
				# with open(csv_dir, 'a+') as csvFile:
				#     writer = csv.writer(csvFile)
				#     writer.writerow(x)

				# Draw the display.
				draw_image(display, image_rgb)
				draw_image(display, image_semseg, blend=True)
				# Render the text messafes
				display.blit(
					font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
					(8, 10))
				display.blit(
					font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
					(8, 28))
				display.blit(
					font.render('control: [{}, {}]'.format(control.throttle, control.steer), 
						True, (255, 255, 255)), (8,46))
				if safety_checking:
					display.blit(textsurface,(8, 64))
				pygame.display.flip()
				# wait()
				if len(pred_traj):
					# print("destroy pred_traj")
					for actor in pred_traj:
						actor.destroy()


			print('destroying local actors.')
			for actor in actor_list:
				actor.destroy()


	finally:

		print('destroying global actors.')
		for actor in global_actor_list:
			actor.destroy()

		pygame.quit()
		# record API
		# client.stop_recorder()
		print('done.')



if __name__ == '__main__':

	try:

		main()

	except KeyboardInterrupt:
		print('\nCancelled by user. Bye!')
