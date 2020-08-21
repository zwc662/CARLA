#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import traceback
import copy
import csv
import pickle
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

sys.path.append('../carla/')
import carla
from utils import *

import agents
import random

try:
    import pygame
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


from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.set_default_dtype(torch.float32)
dtype = torch.float32
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
save_dir_base = 'data/testing/'

prev_epoch = 94999
from_epoch = 97000
to_epoch = from_epoch + 2000

# Best minimally deviating"
checkpoint_path = "IJCAI/100/" + \
        "mlp_dict_nx=20_wps5_spd_30_lr0.0001_bs32_optimSGD_ep{}_friction_safe_150cm_train_ep{}.pth".format(\
        prev_epoch, from_epoch)
stat_type = "minimally_deviating"

checkpoint_path = "mlp_dict_nx=20_wps5_spd_30_lr0.0001_bs32_optimSGD_ep63999_ep66000.pth"
stat_type = "initial"

""" Good initial neural controllers
        "mlp_dict_nx=20_wps5_spd_30_lr0.0001_bs32_optimSGD_ep90999_friction_safe_150cm_train_ep91000.pth"
        "mlp_dict_nx=20_wps5_spd_30_lr0.0001_bs32_optimSGD_ep91999_friction_safe_150cm_train_ep94000.pth"
"""
""" Partially good neural controller
        "mlp_dict_nx=20_wps5_spd_30_lr0.0001_bs32_optimSGD_ep93999_friction_safe_150cm_train_ep95000.pth"
"""

#checkpoint_path = "mlp_dict_nx=20_wps5_spd_30_lr0.0001_bs32_optimSGD_ep70999_friction_safe_150cm_train_ep75000.pth"

#Naive
#No friction bad
#checkpoint_path = "IJCAI/mlp_dict_nx=20_wps5_spd_30_lr0.0001_bs32_optimSGD_ep70999_friction_safe_150cm_train_ep76000.pth"
#OK
#checkpoint_path = "mlp_dict_nx=20_wps5_spd_30_lr0.0001_bs32_optimSGD_ep90999_friction_safe_150cm_train_ep91000.pth"
#stat_type = "naive"

#Initial
#checkpoint_path = "IJCAI/mlp_dict_nx=20_wps5_spd_30_lr0.0001_bs32_optimSGD_ep63999_ep66000.pth"

#Minimal Deivating
#checkpoint_path = "mlp_dict_nx=20_wps5_spd_30_lr0.0001_bs32_optimSGD_ep94999_friction_safe_150cm_train_ep95000.pth"
#stat_type = "minimally_deviating"

model_path = "./checkpoints/" + checkpoint_path

epoch = 70000
no_interference = False 
save_data = False #True


from model_predicative_control_new import MPCController
from agents.navigation.my_basic_agent import BasicAgent
from ilqr.ilqr import ILQRController
#from synchronous_mode_client_control_test_NN import spawn_trolley

from agents.navigation.my_local_planner import _retrieve_options, RoadOption
from agents.navigation.my_local_planner import *
from agents.tools.misc import distance_vehicle, draw_waypoints
from NN_controller import mlp
#from NN_controller_img import MyCoil

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

    def tick(self, timeout, vehicle=None, control=None):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout, vehicle, control) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout, vehicle=None, control=None):
        # counter = 0
        while True:
            data = sensor_queue.get(timeout=timeout)
            # if vehicle is not None:
            #     print(counter, "control", control.throttle, control.steer)
            #     vehicle.apply_control(control)
            #     counter += 1
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

def get_event():
    global should_auto
    global should_quit
    global should_save
    global should_slip

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                should_quit = True
            elif event.key == pygame.K_a:
                if should_auto is False:
                    should_auto = True
                elif should_auto is True:
                    should_auto = False
            elif event.key == pygame.K_s:
                if should_save is False:
                    should_save = True
                elif should_save is True:
                    should_save = False
            elif event.key == pygame.K_f:
                if should_slip is False:
                    should_slip = True
                elif should_slip is True:
                    should_slip = False
    


# =================================
# RH: customized methods
# ===============================
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_ilqr_control(measurements, controller, target_waypoints, us_init = None, avoidances = None, verbose = False):
    # build a new trajectory including the measurements, dynamics model states and controls
    trajectory_dict = {"nn": False, "measurements": [], "waypoints": [], "xs": [], "us": []} 
    trajectory = AttrDict(trajectory_dict)

    number_waypoints = controller.steps_ahead
    # Build target waypoints array
    target_waypoints_np = []
    for waypoint in list(target_waypoints)[:number_waypoints]:
        target_waypoints_np.append(transform_to_arr(waypoint.transform)[0:2])
    target_waypoints_np = np.array(target_waypoints_np)
    assert target_waypoints_np.shape == (number_waypoints, 2)

    # for ILQRController
    xs, us = controller.control(target_waypoints_np, measurements, us_init, avoidances)
    # print("one_log_dict in run_carla_client")x
    # print(one_log_dict)
    control = carla.VehicleControl()
    # control.throttle, control.steer = one_log_dict['throttle'], one_log_dict['steer']
    control.throttle, control.steer = np.clip(us[0][0], 0, 1), np.clip(us[0][1], -1, 1)

    trajectory.us = us.tolist()
    trajectory.xs = xs.tolist()
    trajectory.waypoints = target_waypoints
    trajectory.measurements.append(measurements)
    for i in range(1, controller.steps_ahead):
        trajectory.measurements.append(build_measurements(xs[i], trajectory.measurements[-1]))

    if verbose:
        print("output from ilqr_control", control.throttle, control_ilqr.steer)

    return control, trajectory

def verify_in_lane(trajectory, horizon, threshold = 2.0, m = None):
    unsafe = 0
    for i in range(horizon):
        i_vehicle_loc = trajectory.measurements[i].t.location
        if m is not None:
            i_waypoint_loc = m.get_waypoint(i_vehicle_loc)
        else:
            i_waypoint_loc = trajectory.waypoints[i].transform.location
        
        dist = np.linalg.norm([i_waypoint_loc.x - i_vehicle_loc.x, i_waypoint_loc.y - i_vehicle_loc.y])
        if dist > threshold:
            unsafe += 1           
            #print("!!!!!!!!! {} unsafe!!!!!!!!!".format("nn" if trajectory['nn'] else 'safe controller'))
            #print("Unsafe at t {}, vehicle {}, waypoint {}, disrtnace {}".format(i, i_vehicle_loc, i_waypoint_loc, dist))
    return unsafe      

def verify_avoidance(trajectory, horizon, avoidances, threshold = 1.0):
    unsafe = False
    for i in range(horizon):
        i_vehicle_loc = trajectory.measurements[i].t.location
        if len(avoidances) == 1:
            i_avoidance_loc = avoidances[0]
        else:
            i_avoidance_loc = avoidances[i]
    
        dist = np.linalg.norm([i_avoidance_loc.x - i_vehicle_loc.x, i_avoidance_loc.y - i_vehicle_loc.y])
        if dist < threshold:
            if not unsafe:
                unsafe = (i, i_vehicle_loc, i_avoidance_loc, dist)
                return unsafe      
            print("!!!!!!!!! {} unsafe!!!!!!!!!".format("nn" if trajectory['nn'] else 'safe controller'))
            print("At {}, vehicle {}, waypoint {}, disrtnace {}".format(i, i_vehicle_loc, i_avoidance_loc, dist))
    return unsafe      


    
def get_nn_controller(state, model, device = 'cpu', verbose = False):
    state = torch.from_numpy(state).float().to(device)
    state = state.view(1, -1)

    # print("state", state.size(), state)
    output = model(state) #RuntimeError: size mismatch, m1: [10 x 1], m2: [10 x 100] at /pytorch/aten/src/TH/generic/THTensorMath.cpp:961
    output = output.data.cpu().numpy()[0]
    if verbose:
        print("output from nn_controller", output)
    control = carla.VehicleControl()
    control.throttle, control.steer = output[0].item(), output[1].item()
    # control = carla.VehicleControl(throttle=output[0], steer=output[1])
    return control


def get_nn_prediction(m, measurements, controller, model, horizon, target_speed, nn_number_waypoints):
    # build a new trajectory including the measurements, dynamics model states and controls
    trajectory_dict = {"nn": True, "measurements": [], "waypoints": [], "xs": [], "us": [], "states": []} 
    trajectory = AttrDict(trajectory_dict)

    # get current measurements for horizon length of time
    i_measurements = measurements
    i_target_waypoints = deque(maxlen=nn_number_waypoints)
    i_min_distance = controller.dt * target_speed/3.6* 0.9
    for i in range(horizon):
        # Get closest waypoint for the current measurement
        i_current_waypoint = m.get_waypoint(i_measurements.t.location)
        query_target_waypoints(i_current_waypoint, \
                target_speed, nn_number_waypoints, i_target_waypoints, \
                m = m, measurements = i_measurements, min_distance = i_min_distance)

        # Build state input for neural controller
        i_unnormalized_state, i_state = build_nn_state(i_measurements, \
                target_speed, nn_number_waypoints, i_target_waypoints)
        # Get control output from neurla controller
        i_control = get_nn_controller(i_state, model)
    
        # Build state and action for predictive model
        controller.get_state(i_measurements)
        i_x = np.asarray([controller.measurements['posx'], \
                controller.measurements['posy'], \
                controller.measurements['theta'], \
                controller.measurements['v']])
        i_u = np.asarray([i_control.throttle, i_control.steer])

        # Add state and action to trajectory
        trajectory.measurements.append(i_measurements)
        trajectory.waypoints.append(i_current_waypoint)
        trajectory.xs.append(i_x)
        trajectory.us.append(i_u)
        trajectory.states.append(i_state)

        # Predict next state 
        i_x_nxt = controller.dynamics.f(i_x, i_u, i).flatten()
        #print("Predicted transition:", i, x, u, x_nxt)
        # Build measurement for next state
        i_measurements = build_measurements(i_x_nxt, i_measurements)
    return trajectory
   

def build_measurements(x, measurements_prev):
    location = carla.Location(x = x[0], y =x[1], z = measurements_prev.t.location.z)

    rotation = carla.Rotation(pitch = measurements_prev.t.rotation.pitch, \
            yaw = x[2] * 180/np.pi, \
            roll = measurements_prev.t.rotation.roll)

    t = carla.Transform(location, rotation)

    v = carla.Vector3D(x = x[3] * np.cos(x[2]), y = x[3] * np.sin(x[2]), z = 0.0)

    measurements_dict = {"v": v, "t": t}
    measurements = AttrDict(measurements_dict)

    return measurements
    


def build_nn_state(measurements, target_speed, nn_number_waypoints, target_waypoints):
    # get current nn state
    t = measurements['t']
    v = measurements['v']
    
    target_waypoints_np = []
    for waypoint in list(target_waypoints)[:nn_number_waypoints]:
        target_waypoints_np.append(transform_to_arr(waypoint.transform))
    target_waypoints_np = np.array([target_waypoints_np]).flatten()

    current_speed = np.linalg.norm(np.array([v.x, v.y]))

    full_state = np.hstack((current_speed, transform_to_arr(t), target_speed, target_waypoints_np)).flatten() # shape(188,)
		
    unnormalized_state = np.hstack((full_state[0:3], full_state[5], full_state[7]))
    state = np.hstack((full_state[0]/max_speed_val, full_state[1]/max_pos_val, full_state[2]/max_pos_val, full_state[5]/max_yaw_val, full_state[7]/max_speed_val))
    for j in range(nn_number_waypoints): # concatenate x, y, yaw of future_wps
        unnormalized_state = np.hstack((unnormalized_state, full_state[8+j*6], full_state[9+j*6], full_state[12+j*6]))
        state = np.hstack((state, full_state[8+j*6]/max_pos_val, full_state[9+j*6]/max_pos_val, full_state[12+j*6]/max_yaw_val))
    return unnormalized_state, state


def compute_target_waypoint(last_waypoint, target_speed):
    local_sampling_radius = 0.5
    sampling_radius = 0.05 * target_speed/3.6 # km/h -> m/s
    #sampling_radius = local_sampling_radius
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
        target_waypoint = next_waypoints[road_options_list.index(road_option)]
    # print("road_option", road_option)
    return target_waypoint, road_option

def query_target_waypoints(current_waypoint, target_speed, number_waypoints, target_waypoints, road_option = None, **kwargs):
    # Method 1. query the target-waypoints based on current location of each frame
    # # query target_waypoints based on current location
    if 'measurements' not in kwargs.keys():
        target_waypoints.clear()
    else:
        # Method 2. use the buffer target_waypoints to store the waypoints
        # Once a waypoint is reached pop one waypoint out and push a new waypoint in
        m = kwargs['m']
        measurements = kwargs['measurements']
        min_distance = kwargs['min_distance']
        from_waypoint = m.get_waypoint(measurements.t.location)
        max_index = -1
        for num, waypoint in enumerate(list(target_waypoints)):
            if distance_vehicle(waypoint, from_waypoint.transform) < min_distance:
                max_index = num
        if max_index >= 0:
            for num in range(1 + max_index):
                target_waypoints.popleft()
    if len(target_waypoints) > 0:
        from_waypoint = target_waypoints[-1]
    else:
        from_waypoint = current_waypoint
    for k in range(number_waypoints - len(target_waypoints)):
        target_waypoint, road_option = compute_target_waypoint(from_waypoint, target_speed)
        target_waypoints.append(target_waypoint)
        from_waypoint = target_waypoint

    
def leave_start_position(start_pose_loc, vehicle):
    veh_loc = vehicle.get_transform().location
    dist_vec = np.array([start_pose_loc.x-veh_loc.x, start_pose_loc.y-veh_loc.y])
    print("dist[from start]", dist_vec, np.linalg.norm(dist_vec))
    return np.linalg.norm(dist_vec)


def reach_destination(destination_loc, vehicle):
    veh_loc = vehicle.get_transform().location
    dist_vec = np.array([destination_loc.x-veh_loc.x, destination_loc.y-veh_loc.y])
    print("dist[to end]", dist_vec, np.linalg.norm(dist_vec))
    return np.linalg.norm(dist_vec)

def transform_to_arr(tf):
    return np.array([tf.location.x, tf.location.y, tf.location.z, \
            tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll])



    

# ===============================
# main function
# ===============================

def main():
    target_waypoints_bak = []
    # Initialize pygame
    pygame.init()

    # Initialize actors, display configuration and clock
    actor_list = []
    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    # Build connection between server and client
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    # Get the world class
    world = client.get_world()
    # actor_list = world.get_actors() # can get actors like traffic lights, stop signs, and spectator



            #carla.Transform(carla.Location(x=64, y = 70, z=15), carla.Rotation(pitch=-80, yaw = 0)))
    # set a clear weather
    # weather = carla.WeatherParameters(cloudyness=0.0, precipitation=0.0, sun_altitude_angle=90.0)
    # world.set_weather(weather)
    world.set_weather(carla.WeatherParameters.ClearNoon)

    # Some global variables
    global should_auto
    should_auto = False
    global should_save
    should_save = False
    global should_slip
    should_slip = False
    global should_quit
    should_quit = False


    try:
        m = world.get_map()
        spawn_points = m.get_spawn_points()
        print("total number of spawn_points", len(spawn_points))

        # Initialize spawn configuration
        spawn_config = 1
        start_pose, destination = choose_spawn_destination(m, spawn_config = spawn_config)
        # Manually choose spawn location
        #start_pose = carla.Transform(location=carla.Location(x=-6.446170, y=-79.055023))
        print("start_pose")
        print(start_pose)


        # Manually choose destination
        destination = carla.Transform(location = carla.Location(x = 121.678581, y=61.944809, z=-0.010011))
        destination = carla.Transform(location = carla.Location(x = 121.678581, y=61.944809, z=-0.010011))
        #?????destination = carla.Transform(location=carla.Location(x=-2.419357, y=204.005676, z=1.843104))
        print("destination")
        print(destination)

        # Find the first waypoint equal to spawn location        
        print("Start waypoint", start_pose.location)
        start_waypoint = m.get_waypoint(start_pose.location)
        print("End waypoint", destination.location)
        end_waypoint = m.get_waypoint(destination.location)


        # Get blueprint library
        blueprint_library = world.get_blueprint_library()

        # set a constant vehicle
        vehicle_temp = random.choice(blueprint_library.filter('vehicle.lincoln.mkz2017'))
        vehicle = world.spawn_actor(vehicle_temp, start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(True)

        # Set a on-vehicle rgb camera
        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=5.8), carla.Rotation(pitch=-35)),
            attach_to=vehicle)
        """
        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=73, y = 73, z=15), carla.Rotation(pitch=-90, yaw = 0)))
            #starting position
            #carla.Transform(carla.Location(x=110, y =55, z=15), carla.Rotation(pitch=-45, yaw = 180)))
            #friction 
            # Angle
            #carla.Transform(carla.Location(x=58, y = 72, z=15), carla.Rotation(pitch=-45, yaw = -40)))
            # bird's view friction
            #carla.Transform(carla.Location(x=64, y = 70, z=15), carla.Rotation(pitch=-80, yaw = 0)))
            # towards center
            #carla.Transform(carla.Location(x=73, y = 73, z=15), carla.Rotation(pitch=-90, yaw = 0)))
        """


        actor_list.append(camera_rgb)


        # Set a on-vehicle perceptron module
        camera_semseg = world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_semseg)

        # Create a basic agent in the vehicle
        target_speed = 30
        vehicle_agent = BasicAgent(vehicle, target_speed=target_speed)
        destination_loc = destination.location
        vehicle_agent.set_destination((destination_loc.x, destination_loc.y, destination_loc.z))

        # Initialize an NN controller from file 
        nn_number_waypoints = 5
        model = mlp(nx = (5+3*nn_number_waypoints), ny = 2)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(dtype).to(device)
        model.eval()
        print("Model loaded")
        
        # Initialize an ILQR controller
        controller = ILQRController(target_speed, steps_ahead = 100, dt = 0.05, l = 1.0, half_width = 1.5)
        ilqr_number_waypoints = controller.steps_ahead

        """ Collect data to file"""
        #csv_dir = build_file_base(epoch, timestr, spawn_config, nn_number_waypoints, target_speed, info = 'friction_safe_150cm_run') if save_data else None
        #csv_dir = "{}_friction_{}.csv".format(stat_type, not no_interference) if save_data else None
        csv_dir = "{}_SC_friction_{}.csv".format(stat_type, not no_interference) if save_data else None

        # Create a synchronous mode context.
        MIN_DISTANCE_PERCENTAGE = 0.9
        MIN_DISTANCE = target_speed/3.6 * MIN_DISTANCE_PERCENTAGE

        nn_min_distance = MIN_DISTANCE
        nn_target_waypoints = deque(maxlen=nn_number_waypoints)

        ilqr_min_distance = 0.0
        ilqr_target_waypoints = deque(maxlen=ilqr_number_waypoints)
    
        visual_number_waypoints = 1
        visual_target_waypoints = deque(maxlen=visual_number_waypoints)

        max_distance = 1.5
        #vehicle.set_autopilot(True)

        # Start simulation
        with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=30) as sync_mode:
            nn_trajectory = []
            ilqr_trajectory = [] 
            tot_episodes = 0
            tot_time = 0
            spd_flg = 0
            top_spd = 0
            bottom_spd = 10
            avg_spd = 0
            avg_spd_var = 0
            avg_dist = 0
            dists = []
            avg_dist_var = 0
            tot_unsafe_time = 0
            tot_unsafe_episodes = 0
            tot_interference_episodes = 0
            tot_interference_time = 0
            tot_loss = 0
            time_begin = None
            avg_inf_time = 0
            tot_inf_time = 0
            avg_verif_time = 0
            tot_verif_time = 0
            avg_int_time = 0
            tot_int_time = 0

            while True:
                get_event()
                
                # Quit the game once ESC is pressed
                if should_quit:
                    return

                # Spawn Trigger Friction once f key is pressed
                avoidances = [carla.Location(64, 62, 0)]
                #avoidances = [carla.Location(83, 50, 0)]
                if should_slip:
                    for avoidance in avoidances:
                        friction_bp = world.get_blueprint_library().find('static.trigger.friction')
                        friction_bp, friction_transform, friction_box = config_friction(friction_bp, \
                            location = avoidance, \
                            extent = carla.Location(6., 2., 0.2),\
                            color = (255, 127, 0))
                        world.debug.draw_box(**friction_box)
                        frictioner = world.spawn_actor(friction_bp, friction_transform)
                        actor_list.append(frictioner)
                    should_slip = False


                
                clock.tick(30)

                # Get the current measurements
                t = vehicle.get_transform()
                v = vehicle.get_velocity()

                measurements_dict = {"v": v, "t": t} 
                measurements = AttrDict(measurements_dict)
                print("Location:", measurements.t.location)
                

                spd = np.linalg.norm([v.x, v.y], ord = 2)
                print("Velocity:", spd)
                if top_spd <= spd:
                    top_spd = spd
                    print("New top speed: {}".format(top_spd))
                if spd_flg > 0 and bottom_spd >= spd:
                    bottom_spd = spd
                    print("New bottom speed: {}".format(bottom_spd))

                nn_min_distance = MIN_DISTANCE
                ilqr_min_distance = spd * controller.dt
                

                # get last waypoint
                current_waypoint = m.get_waypoint(t.location)
                current_distance = distance_vehicle(current_waypoint, t)
                dists.append(current_distance)
                avg_dist += current_distance
                avg_dist_var += current_distance**2
                if current_distance > max_distance:
                    tot_unsafe_episodes += 1
                    tot_unsafe_time += controller.dt
                
                """ (visualize) Query ground true future waypoints
                """
                query_target_waypoints(current_waypoint, \
                        target_speed, visual_number_waypoints, visual_target_waypoints, \
                        m = m, measurements = measurements, min_distance = nn_min_distance * 0.1)
                for visual_target_waypoint in visual_target_waypoints:
                    world.debug.draw_box(**config_waypoint_box(visual_target_waypoint, color = (0, 255, 0)))

                """ Collect true waypoints with PD controller
                """
                # Run PD controllerr
                control_auto, target_waypoint = vehicle_agent.run_step(debug=False)
                # Draw PD controller target waypoints
                #world.debug.draw_box(**config_waypoint_box(target_waypoint, color = (0, 0, 0)))

                if should_save:
                    # Start saving target waypoint to waypoint file
                    print("Stored target waypoint {}".format(\
                            [target_waypoint.transform.location.x, target_waypoint.transform.location.y]))
                    target_waypoints_bak.append([target_waypoint.transform.location.x, target_waypoint.transform.location.y])

                # Run other controllers  
                if not should_auto:
                    # Draw nn controller waypoints
                    #for horizon_waypoint in target_waypoints:
                    #    world.debug.draw_box(**config_waypoint_box(target_waypoint, color = (255, 0, 0)))

                    # Run constant control 
                    # control = carla.VehicleControl(throttle=1, steer=0)


                    """ Predict future nn trajectory and draw
                    """
                    inf_time_begin = time.clock()
                    nn_trajectory_pred = get_nn_prediction(m, measurements, controller, model, \
                            ilqr_number_waypoints, target_speed, nn_number_waypoints)
                    inf_time_end = time.clock()
                    avg_inf_time = inf_time_end - inf_time_begin
                    tot_inf_time += 1


                    state = nn_trajectory_pred.states[0]
                    # Draw predcted measurements
                    
                    #for i_measurements in nn_trajectory_pred.measurements:
                    #    world.debug.draw_box(**config_measurements_box(i_measurements,\
                    #            color = (255, 0, 0)))
                    us_init = np.array(nn_trajectory_pred.us[:ilqr_number_waypoints - 1])
                    print("Verify predicted nn trajectory")
                    unsafe = verify_in_lane(nn_trajectory_pred, ilqr_number_waypoints, max_distance)
                    #unsafe = verify_avoidance(nn_trajectory_pred, number_waypoints, avoidances, 1.0)
                    if unsafe < ilqr_number_waypoints or no_interference:
                        """ Get output from neural controller
                        """
                        # Visualize
                        if stat_type == "minimally_deviating":
                            color = (255, 0, 255)
                        elif stat_type == "initial":
                            color = (255, 0, 0)
                        elif stat_type == "naive":
                            color = (0, 255, 255)
                        world.debug.draw_box(**config_measurements_box(measurements, color = color))

                        if unsafe: 
                            print("!!!!!!!!!!!!!!\nUNSAFE NN operation number {}".format(unsafe))
                            print("!!!!!!!!!!!!!!")

                        # Query future waypoints
                        query_target_waypoints(current_waypoint, \
                                target_speed, nn_number_waypoints, nn_target_waypoints, \
                                m = m, measurements = measurements, min_distance = nn_min_distance)
                        # Draw future waypoints
                        #for nn_target_waypoint in nn_target_waypoints:
                        #    world.debug.draw_box(**config_waypoint_box(nn_target_waypoint, color = (0, 255, 0)))
                        # Build state for neural controller
                        unnormalized_state, state = build_nn_state(measurements, \
                            target_speed, nn_number_waypoints, nn_target_waypoints)
                        # Get nn control
                        control_nn = get_nn_controller(state, model, device)
                        control = control_nn
                        nn_trajectory.append([tot_episodes, control.throttle, control.steer, spd])
                    else:
                        print("!!!!!!!!!!!!!!\nUNSAFE NN operation number {}".format(unsafe))
                        print("!!!!!!!!!!!!!!")

                        """ Run ilqr controller
                        """
                        # Visualize
                        world.debug.draw_box(**config_measurements_box(measurements, color = (0, 0, 255)))

                        us_init = 0.0 * us_init
                        # Query future waypoints
                        query_target_waypoints(current_waypoint, \
                            target_speed, ilqr_number_waypoints, ilqr_target_waypoints, \
                            m = m, measurements = measurements, min_distance = ilqr_min_distance)
                        # Draw future waypoints
                        #for ilqr_target_waypoint in ilqr_target_waypoints:
                        #    world.debug.draw_box(**config_waypoint_box(ilqr_target_waypoint, color = (0, 0, 255)))
                        # Get ilqr control
                        verif_time_begin = time.clock()
                        control_ilqr, ilqr_trajectory_pred = get_ilqr_control(measurements, \
                                controller, ilqr_target_waypoints, avoidances = avoidances, us_init = us_init)
                        verif_time_end = time.clock()
                        avg_verif_time += verif_time_end - verif_time_begin
                        tot_verif_time += 1

                        control = control_ilqr

                        print("Verify predicted ilqr trajectory")
                        unsafe = verify_in_lane(ilqr_trajectory_pred, ilqr_number_waypoints, max_distance)
                        #unsafe = verify_avoidance(ilqr_trajectory_pred, number_waypoints, avoidances, 2.0)
                        if unsafe:
                            print("!!!!!!!!!!!!!!!!!!!!!!!!!\nUNSAFE ilqr operation number {}".format(unsafe))
                            print("!!!!!!!!!!!!!!!!!!!!!!!!!")
                        
                        tot_interference_episodes += 1
                        tot_interference_time += controller.dt

                        ilqr_trajectory.append([tot_episodes, control.throttle, control.steer, spd])

                

                else:
                    print("Auto pilot takes it from here")
                    print("output from PD_control", control_auto.throttle, control_auto.steer)
                    control = control_auto 
                
                # Apply control to the actuator
                vehicle.apply_control(control)
                print("[Apply control]", control.throttle, control.steer)

                # Store vehicle information
                if leave_start_position(start_pose.location, vehicle) >= 2 or tot_episodes >= 10:
                    # If having left the start position
                    tot_episodes += 1
                    tot_time += clock.get_time()/1e3
                    
                    #controller.dt = clock.get_time()/1e3
                    #controller.steps_ahead = int(5./controller.dt)
                    #ilqr_number_waypoints = controller.steps_ahead
                    #print("Update ILQR steps ahead {}, step length {}".format(controller.steps_ahead,\
                    #        controller.dt))

                    
                    if spd >= 8.:
                        spd_flg += 1
                        if time_begin is None:
                            time_begin = time.clock()
                    avg_spd += spd
                    avg_spd_var += (spd - target_speed/3.6)**2 
                    #nn_trajectory.append([control.throttle, control.steer, spd])

                    # Collect state-control training data 
                    #    world.debug.draw_box(**config_measurements_box(i_measurements,\
                    #            color = (255, 0, 0)))
                    if csv_dir is not None:
                        y_x = np.hstack((np.array([control.throttle, control.steer]), state))
                        with open(csv_dir, 'a+') as csvFile:
                            writer = csv.writer(csvFile)
                            writer.writerow(y_x)

                    # If reaching destination 
                    if reach_destination(destination_loc, vehicle) < 2.0 or (control_auto.throttle == 0 and control_auto.steer == 0.0):
                        if tot_episodes >= 10:
                            raise Exception("Endgame")
                            print("It is not the end!!!??")
                            return

                    print(">>>>Episode: {}".format(tot_episodes))
                    time_end = time.clock()
                    





                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0, vehicle=vehicle, control=control)
                
                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                # Draw the display.
                draw_image(display, image_rgb)
                #draw_image(display, image_semseg, blend=True)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()

                # raise ValueError("stop here")


            print('destroying local actors.')
            for actor in actor_list:
                actor.destroy()
            
                

    except Exception as e:
        print(e)
        #if tot_episodes <= 10:
        #    os.remove(csv_dir)
        traceback.print_exc()

    finally:

        """ Store waypoints to file
        if len(target_waypoints_bak) is not 0:
            print("Store {} waypoints".format(len(target_waypoints_bak)))
            pickle.dump(target_waypoints_bak, open('./wps_at_plant_rotary_02.pt', 'wb'))
        else:
            print("Did not store target waypoints (length={})".format(len(target_waypoints_bak)))
        """
        if should_quit:
            os.remove(csv_dir)

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        avg_spd /= tot_episodes 
        avg_spd_var /= tot_episodes
        avg_spd_var = avg_spd_var
        avg_dist /= tot_episodes
        avg_dist_var /= tot_episodes
        avg_dist_var = avg_dist_var

        diff_dist_var = np.var(np.diff(dists, axis = 0), axis = 0)
        print(">>>>>>>>>>> Total episodes: {} episodes||pygame time: {}s||Computer time: {}s <<<<<<<<<<<<<<".format(
            tot_episodes, tot_time, time_end - time_begin))
        print(">>>>>>>>>>>>>Total unsafe episodes: {}||{}s<<<<<<<<<<<<<<<<".format(tot_unsafe_episodes,\
                tot_unsafe_time))
        print(">>>>>>>>>>>>>>Total interference episodes: {}||{}s<<<<<<<<<".format(tot_interference_episodes,\
                tot_unsafe_time))
        print(">>>>>>>>>>>>>Average Distance: {}<<<<<<<<<<<<<<<<<<<<<".format(avg_dist))
        print(">>>>>>>>>>>>>Average Distance Variance: {}<<<<<<<<<<<<<<<<<<<<<".format(avg_dist_var))
        print(">>>>>>>>>>>>>Derivative Distance Variance: {}<<<<<<<<<<<<<<<<<<<<<".format(diff_dist_var))
        print(">>>>>>>>>>>>>>Top speed: {} <<<<<<<<<<<<<<<<<<<<<<".format(top_spd))
        print(">>>>>>>>>>>>>>Bottom speed: {} <<<<<<<<<<<<<<<<<<<<<<".format(bottom_spd))
        print(">>>>>>>>>>>>>>Average speed: {}<<<<<<<<<<<<<<<<<<<".format(avg_spd))
        print(">>>>>>>>>>>>>Average speed Variance: {}<<<<<<<<<<<<<<<<<<<<<".format(avg_spd_var))
        print(">>>>>>>>>>>>>Average inference time: {}<<<<<<<<<<<<<<<<<<<<<".format(avg_inf_time/tot_inf_time))
        print(">>>>>>>>>>>>>Average verification time: {}<<<<<<<<<<<<<<<<".format(avg_verif_time/tot_verif_time))

        pygame.quit()
        print('done.')

        #pickle.dump((nn_trajectory, ilqr_trajectory), open("./trajectory_" + timestr + ".p", 'wb'))
        pickle.dump((nn_trajectory), open("./bak/trajectory_safe" + timestr + ".p", 'wb'))
    



if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
