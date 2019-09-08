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
import math

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random

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
from model_predicative_control_new import MPCController
from agents.navigation.my_basic_agent import BasicAgent
from agents.navigation.my_local_planner import _retrieve_options, RoadOption


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
    # print("sampling_radius", sampling_radius)
    # print("local_sampling_radius", local_sampling_radius)
    next_waypoints = list(last_waypoint.next(local_sampling_radius))
    # print("length of next_waypoints", len(next_waypoints))
    # print(next_waypoints)

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
    return target_waypoint, road_option


def wait():
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                return


def spawn_trolley(world, blueprint_library, x=0, y=0, z=10):
     # spawn a trolley to visualilze target_waypoint
    trolley_bp = random.choice(blueprint_library.filter('static.prop.shoppingtrolley')) # vehicle.toyota.prius static.prop.shoppingtrolley
    trolley_tf = carla.Transform(location=carla.Location(x=x, y=y, z=z))
    # print("trolley_bp", trolley_bp, trolley_tf.location)
    trolley = world.spawn_actor(trolley_bp, trolley_tf)
    trolley.set_simulate_physics(False)
    return trolley
    
# ===============================
# main function
# ===============================

def main(i, j):
    pygame.init()
    hud_dim = [200, 88]  # default: 800, 600 # collect data: 200, 88
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
    save_dir_base = 'data/dest_start_SR0.5/' # collect data after shortening the horizon

    try:
        m = world.get_map()
        spawn_points = m.get_spawn_points() # get_spawn_points will return the same list each time called
        # print("total number of spawn_points", len(spawn_points)) # 265
        destiny = spawn_points[i]
        print(i, "car destiny", destiny.location)
        start_pose = spawn_points[j]
        print(j, "car start_pose", start_pose.location)



        # save_dir = save_dir_base + 'data_{}_{}/'.format(i, j)
        save_dir = os.path.join(save_dir_base, 'data_{}_{}/'.format(i, j))
        csv_dir = save_dir+"all_{}_{}.csv".format(i, j)
        # print("save_dir", save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(csv_dir, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(["throttle", "steer", \
                             "cur_speed", "cur_x", "cur_y", "cur_z", "cur_pitch", "cur_yaw", "cur_roll", \
                             "target_speed", "target_x", "target_y", "target_z", "target_pitch", "target_yaw", "target_roll"])
  
        actor_list = []
        # set and spawn actor
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.lincoln.mkz2017'))
        vehicle = world.spawn_actor(vehicle_bp, start_pose)
        vehicle.set_velocity(carla.Vector3D(x=0, y=5))
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
            carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)


        camera_semseg_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        camera_semseg_bp.set_attribute('image_size_x', str(hud_dim[0]))
        camera_semseg_bp.set_attribute('image_size_y', str(hud_dim[1]))

        camera_semseg = world.spawn_actor(
            camera_semseg_bp,
            carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_semseg)

        # set PD control to the vehicle
        target_speed = 30
        vehicle_agent = BasicAgent(vehicle, target_speed=target_speed)
        destiny_loc = destiny.location
        vehicle_agent.set_destination((destiny_loc.x, destiny_loc.y, destiny_loc.z))
        # vehicle.set_autopilot(True)

        # print("local actor list", actor_list)
        num_wps = 30
        last_dist = math.inf
        eps = 1e-3

        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=20) as sync_mode:
            stuck_counter = 0
            dist  = reach_destiny(destiny_loc, vehicle)
            while dist > 10:
                if should_quit():
                    print('destroying local actors.')
                    for actor in actor_list:
                        actor.destroy()
                    return
                # if abs(dist - last_dist) < eps:
                #     stuck_counter += 1
                #     if stuck_counter >= 100:
                #         print("stuck for too long, return")
                #         stuck_counter = 0
                #         return
                clock.tick(20)

                t = vehicle.get_transform()
                v = vehicle.get_velocity()

                last_waypoint = m.get_waypoint(t.location)
                target_waypoints = []

                actor_list.append(spawn_trolley(world, blueprint_library, t.location.x, t.location.y))
                # wait() # wait for KEYDOWN to be pressed

                for k in range(num_wps):
                    target_waypoint, road_option = compute_target_waypoint(last_waypoint, target_speed)
                    target_waypoints.append([target_waypoint, road_option])
                    last_waypoint = target_waypoint
                    # actor_list.append(spawn_trolley(world, blueprint_library, x=target_waypoint.transform.location.x, y=target_waypoint.transform.location.y))
                    # wait()

                target_waypoints_np = []
                for wp_ro in target_waypoints:
                    target_waypoints_np.append(transform_to_arr(wp_ro[0].transform))
                target_waypoints_np = np.array([target_waypoints_np]).flatten()

                # BasicAgent use PD control
                # query both control output and future_wps from Local LocalPlanner
                control, target_waypoint = vehicle_agent.run_step(debug=False)
                if target_waypoint is None:
                    break
                # control = get_mpc_control(world, vehicle, m)
                # control = carla.VehicleControl(throttle=1, steer=0)
                # print("current location", t)
                # print("target_waypoint", target_waypoint)
                # print("control", control.throttle, control.steer)
                vehicle.apply_control(control)
                cur_speed = np.linalg.norm(np.array([v.x, v.y]))

                # print("cur_speed", cur_speed)
                # print(target_waypoints_np)

                state = np.hstack((cur_speed, transform_to_arr(t), target_speed, target_waypoints_np)).flatten() # shape(188,)
                # print("state", state.shape)
                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)
                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)
                frame_number = image_semseg.frame_number
                
                # save the data
                # print("frame_number")
                image_semseg.save_to_disk(save_dir+'{:08d}'.format(frame_number))

                path = save_dir+'{:08d}_ctv'.format(frame_number) 
                x = np.hstack((np.array([control.throttle, control.steer]), state))
                # print("control and measurement", x)
                # save in two formats, one separate to work with img, one appended to csv file
                np.save(path, x)
                with open(csv_dir, 'a+') as csvFile:
                    writer = csv.writer(csvFile)
                    writer.writerow(x)
                last_dist = dist
                # Draw the display.
                draw_image(display, image_rgb)
                draw_image(display, image_semseg, blend=True)
                display.blit(
                    font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                    (8, 28))
                pygame.display.flip()

            print('destroying local actors.')
            for actor in actor_list:
                actor.destroy()


    finally:
        print('destroying local actors.')
        for actor in actor_list:
            actor.destroy()

        print('destroying global actors.')
        for actor in global_actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:
        # put i, j outside
        num_spawn_points = 265
        for i in range(num_spawn_points):
            # if i % 10 == 0:
            for j in range(num_spawn_points):
                if j == i:
                    continue
                main(i, j)
                
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
