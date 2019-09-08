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

import carla
from utils import *

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
model_path = "./checkpoints/" + \
        "mlp_dict_nx=20_wps5_spd_30_lr0.0001_bs32_optimSGD_ep63999_ep66000.pth"
        #"mlp_dict_nx=20_wps5_spd_30_lr0.0001_bs32_optimSGD_ep2799_ep8000.pth"


from model_predicative_control_new import MPCController
from agents.navigation.my_basic_agent import BasicAgent
from ilqr.ilqr import ILQRController
from synchronous_mode_client_control_test_NN import spawn_trolley

from agents.navigation.my_local_planner import _retrieve_options, RoadOption
from agents.navigation.my_local_planner import *
from agents.tools.misc import distance_vehicle, draw_waypoints
from NN_controller import mlp
from NN_controller_img import MyCoil

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


def get_ilqr_control(measurements, controller, target_waypoints):
    number_waypoints = controller.steps_ahead
    # Build target waypoints array
    target_waypoints_np = []
    for waypoint in list(target_waypoints)[:number_waypoints]:
        target_waypoints_np.append(transform_to_arr(waypoint.transform)[0:2])
    target_waypoints_np = np.array(target_waypoints_np)
    assert target_waypoints_np.shape == (number_waypoints, 2)

    # for ILQRController
    xs, us = controller.control(target_waypoints_np, measurements)
    # print("one_log_dict in run_carla_client")x
    # print(one_log_dict)
    control = carla.VehicleControl()
    # control.throttle, control.steer = one_log_dict['throttle'], one_log_dict['steer']
    control.throttle, control.steer = us[0][0], -us[0][1]

    return control

    
def get_nn_controller(state, model, device = 'cpu'):
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


def get_prediction(m, measurements, controller, model, horizon, target_speed, number_waypoints):
    # build a new trajectory including the measurements, dynamics model states and controls
    trajectory_dict = {"measurements": [], "waypoints": [], "xs": [], "us": []} 
    trajectory = AttrDict(trajectory_dict)

    # get current measurements for horizon length of time
    i_measurements = copy.deepcopy(measurements)
    for i in range(horizon):
        # Get closest waypoint for the current measurement
        current_waypoint = m.get_waypoint(i_measurements.t.location)
        next_waypoint = current_waypoint.random.choice(waypoint.next(1.5))

        # Build state input for neural controller
        state = build_nn_state(i_measurements, next_waypoint, target_speed, number_waypoints)
        # Get control output from neurla controller
        control = get_nn_controller(state, model)
    
        # Build state and action for predictive model
        x = controller.get_state(i_measurements)
        u = np.asarray([control.throttle, control.steer])

        # Add state and action to trajectory
        trajectory.i_measurements.append(i_measurements)
        trajectory.waypoints.append(current_waypoint)
        trajectory.xs.append(x)
        trajectory.us.append(u)

        # Predict next state 
        x_nxt = controller.dynamics.f(x, u, i)
        # Build measurement for next state
        i_measurements = build_measurements(x_nxt, i_measurements)
    return trajectory
   

def build_measurements(x, measurements_prev):
    location_dict = {"x": x[0], "y": x[1], "z": measurements_prev.t.location.z}
    location = AttrDict(location_dict)

    rotation_dict = {"pitch": measurements_prev.t.rotation.pitch, "yaw": x[2] * 180/np.pi, \
            "roll": measurements_prev.rotation.roll}
    rotation = AttrDict(rotation_dict)

    t_dict = {"location": location, "rotatino": rotation}
    t = AttrDict(t_dict)

    v_dict = {"x": x[3] * np.cos(x[2]), "y": x[3] * np.sin(x[2])}
    v = AttrDict(v_dict)

    measurements_dict = {"v": v, "t": t}
    measurements = AttrDict(measurements_dict)

    return measurements
    


def build_nn_state(measurements, target_speed, number_waypoints, target_waypoints):
    # get current nn state
    t = measurements['t']
    v = measurements['v']
    
    target_waypoints_np = []
    for waypoint in list(target_waypoints)[:number_waypoints]:
        target_waypoints_np.append(transform_to_arr(waypoint.transform))
    target_waypoints_np = np.array([target_waypoints_np]).flatten()

    current_speed = np.linalg.norm(np.array([v.x, v.y]))

    full_state = np.hstack((current_speed, transform_to_arr(t), target_speed, target_waypoints_np)).flatten() # shape(188,)
		
    unnormalized_state = np.hstack((full_state[0:3], full_state[5], full_state[7]))
    state = np.hstack((full_state[0]/max_speed_val, full_state[1]/max_pos_val, full_state[2]/max_pos_val, full_state[5]/max_yaw_val, full_state[7]/max_speed_val))
    for j in range(number_waypoints): # concatenate x, y, yaw of future_wps
        unnormalized_state = np.hstack((unnormalized_state, full_state[8+j*6], full_state[9+j*6], full_state[12+j*6]))
        state = np.hstack((state, full_state[8+j*6]/max_pos_val, full_state[9+j*6]/max_pos_val, full_state[12+j*6]/max_yaw_val))
    return unnormalized_state, state


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
        measurements = kwargs['measurements']
        min_distance = kwargs['min_distance']
        max_index = -1
        for num, waypoint in enumerate(list(target_waypoints)):
            if distance_vehicle(waypoint, measurements.t) < min_distance:
                max_index = num
        if max_index >= 0:
            for num in range(1 + max_index):
                target_waypoints.popleft()
    if len(target_waypoints) > 0:
        last_waypoint = target_waypoints[-1]
    else:
        last_waypoint = current_waypoint
    for k in range(number_waypoints - len(target_waypoints)):
        target_waypoint, road_option = compute_target_waypoint(last_waypoint, target_speed)
        target_waypoints.append(target_waypoint)
        last_waypoint = target_waypoint

    
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
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
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
        number_waypoints = 5
        model = mlp(nx = (5+3*number_waypoints), ny = 2)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(dtype).to(device)
        model.eval()
        print("Model loaded")
        
        # Initialize an ILQR controller
        controller = ILQRController(target_speed, steps_ahead = 20)
        number_waypoints = controller.steps_ahead

        """ Collect data to file"""
        csv_dir = None
        #csv_dir = build_file_base(timestr, spawn_config, number_waypoints, target_speed)

        # Create a synchronous mode context.
        MIN_DISTANCE_PERCENTAGE = 0.9
        min_distance = target_speed/3.6*MIN_DISTANCE_PERCENTAGE
        target_waypoints = deque(maxlen=number_waypoints)
        #vehicle.set_autopilot(True)

        # Start simulation
        with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=30) as sync_mode:
            i_episode = 0
            i_time = 0
            top_spd = 0
            avg_spd = 0
            while True:
                get_event()
                
                # Quit the game once ESC is pressed
                if should_quit:
                    return

                # Spawn Trigger Friction once f key is pressed
                if should_slip:
                    friction_bp = world.get_blueprint_library().find('static.trigger.friction')
                    friction_transform, friction_box = config_friction(friction_bp, \
                            location = carla.Location(62, 61, 0), \
                            extent = carla.Location(700., 700., 700.))
                    frictioner = world.spawn_actor(friction_bp, friction_transform)
                    actor_list.append(frictioner)
                    world.debug.draw_box(**friction_box)
                    should_slip = False
                
                clock.tick(30)

                # Get the current measurements
                t = vehicle.get_transform()
                v = vehicle.get_velocity()
                measurements_dict = {"v": v, "t": t} 
                
                measurements = AttrDict(measurements_dict)
                print("Location:", measurements.t.location)

                spd = np.linalg.norm([measurements.v.x, measurements.v.y], ord = 2)
                print("Velocity:", spd)
                if top_spd <= spd:
                    top_spd = spd
                    print("New top speed: {}".format(top_spd))
                

                # get last waypoint
                current_waypoint = m.get_waypoint(t.location)

                query_target_waypoints(current_waypoint, \
                            target_speed, number_waypoints, target_waypoints, \
                            measurements = measurements, min_distance = min_distance)


                # Draw future waypoints
                for target_waypoint in target_waypoints:
                    world.debug.draw_box(**config_waypoint_box(target_waypoint, color = (0, 255, 0)))

                # Run PD controllerr
                control_auto, target_waypoint = vehicle_agent.run_step(debug=False)
                # Draw PD controller target waypoints
                world.debug.draw_box(**config_waypoint_box(target_waypoint, color = (0, 0, 255)))

                if should_save:
                    # Start saving target waypoint to waypoint file
                    print("Stored target waypoint {}".format(\
                            [target_waypoint.transform.location.x, target_waypoint.transform.location.y]))
                    target_waypoints_bak.append([target_waypoint.transform.location.x, target_waypoint.transform.location.y])

                # Run other controllers  
                if not should_auto:
                    """ Run ilqr controller
                    """
                    control_ilqr = get_ilqr_control(measurements, controller, target_waypoints)
                    print("output from ilqr_control", control_ilqr.throttle, control_ilqr.steer)
                    control = control_ilqr
                
                    # Draw nn controller waypoints
                    #for horizon_waypoint in target_waypoints:
                    #    world.debug.draw_box(**config_waypoint_box(target_waypoint, color = (255, 0, 0)))

                    # Run constant control 
                    # control = carla.VehicleControl(throttle=1, steer=0)

                    """ Get output from neural controller
                    # Build state for neural controller
                    unnormalized_state, state = build_nn_state(measurements, \
                            target_speed, number_waypoints, target_waypoints)
                    # Get nn control
                    control_nn = get_nn_controller(state, model, device)
                    control = control_nn
                    """
                else:
                    print("Auto pilot takes it from here")
                    print("output from PD_control", control_auto.throttle, control_auto.steer)
                    control = control_auto 
                
                # Apply control to the actuator
                vehicle.apply_control(control)

                # Store vehicle information
                if leave_start_position(start_pose.location, vehicle) >= 2 or i_episode >= 10:
                    # If having left the start position
                    i_episode += 1
                    i_time += clock.get_time()
                    print("Episode: {}".format(i_episode))
                    avg_spd += spd

                    # Collect state-PDcontrol training data 
                    y_x = np.hstack((np.array([control.throttle, control.steer]), state))

                    if csv_dir is not None:
                        with open(csv_dir, 'a+') as csvFile:
                            writer = csv.writer(csvFile)
                            writer.writerow(y_x)

                    # If reaching destination 
                    if reach_destination(destination_loc, vehicle) < 2.0 or (control_auto.throttle == 0 and control_auto.steer == 0.0):
                        if i_episode >= 10:
                            raise Exception("Endgame")
                            print("It is not the end!!!??")
                            return




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
        #if i_episode <= 10:
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

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        avg_spd /= i_episode
        print(">>>>>>>>>>> Total Time: {} episodes||{}s <<<<<<<<<<<<<<".format(i_episode, i_time/1e3))
        print(">>>>>>>>>>>>>>Top speed: {} <<<<<<<<<<<<<<<<<<<<<<".format(top_spd))
        print(">>>>>>>>>>>>>>Average speed: {}<<<<<<<<<<<<<<<<<<<".format(avg_spd))

        pygame.quit()
        print('done.')

    



if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
