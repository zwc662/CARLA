
#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import pickle

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
from ilqr.ilqr import ILQRController
from synchronous_mode_client_control_test_NN import spawn_trolley

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

def get_mpc_control(world, vehicle, m, controller, actor_list, blueprint_library):
    # TODO: add more params in the dict, 
    # which can be automatically converted (to replace args)
    params_dict = {'target_speed': 10}
    params = AttrDict(params_dict)

    # get current "measurements"
    t = vehicle.get_transform()
    v = vehicle.get_velocity()
    c = vehicle.get_control()
    measurements_dict = {"v": v,
                         "t": t} # TODO: create a dict that return the world&vehicle data similar as 0.8 API
    measurements = AttrDict(measurements_dict)
    
    cur_wp = m.get_waypoint(t.location)

    local_interval = 0.5
    horizon = 9
    # initiate a series of waypoints
    future_wps = []
    future_wps.append(cur_wp)

    for i in range(horizon):
        # TODO: check whether "next" works here
        future_wps.append(random.choice(future_wps[-1].next(local_interval)))

    # save data for testing
    print("future_wps")
    print(future_wps)
    print("measurements")
    print(measurements)
    data = []
    for waypoint in future_wps:
        data.append({"waypoint": transform_to_arr(waypoint.transform)})
        actor_list.append(spawn_trolley(world, blueprint_library, x=waypoint.transform.location.x, y=waypoint.transform.location.y))
    print("transform")
    data.append({"measurements.t": transform_to_arr(measurements.t)})
    print(transform_to_arr(measurements.t))
    print("velocity")
    data.append({"measurements.v": np.array([measurements.v.x, measurements.v.y, measurements.v.z])})
    print(np.array([measurements.v.x, measurements.v.y, measurements.v.z]))
    print("full data")
    print(data)
    # for MPCController
    # one_log_dict = controller.control(future_wps, measurements)
    # for ILQRController
    xs, us = controller.control(future_wps, measurements)
    # print("one_log_dict in run_carla_client")x
    # print(one_log_dict)
    control = carla.VehicleControl()
    # control.throttle, control.steer = one_log_dict['throttle'], one_log_dict['steer']
    control.throttle, control.steer = us[0][0], us[0][1]

    return control
    

def reach_destiny(destiny_loc, vehicle):
    veh_loc = vehicle.get_transform().location
    dist_vec = np.array([destiny_loc.x-veh_loc.x, destiny_loc.y-veh_loc.y])
    print("dist", dist_vec, np.linalg.norm(dist_vec))
    return np.linalg.norm(dist_vec)

def transform_to_arr(tf):
    return np.array([tf.location.x, tf.location.y, tf.location.z, tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll])


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
    actor_list = []

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
        j = 2# 2
        # destiny = spawn_points[i]
        # print(j, "car destiny", destiny.location)
        # start_pose = spawn_points[j]
        wp = m.get_waypoint(carla.Location(x=110, y = 60))
        start_pose = wp.transform
        # print(i, "car start_pose", start_pose.location)

        # start_pose = random.choice(spawn_points)
        print("car start_pose", start_pose.location)
        waypoint = m.get_waypoint(start_pose.location)

        blueprint_library = world.get_blueprint_library()

        actor_list = []

        # set a constant vehicle
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.lincoln.mkz2017'))
        vehicle = world.spawn_actor(vehicle_bp, start_pose)
        actor_list.append(vehicle)
        vehicle.set_simulate_physics(True)

        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)

        camera_semseg = world.spawn_actor(
            blueprint_library.find('sensor.camera.semantic_segmentation'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_semseg)

        vehicle_agent = BasicAgent(vehicle, target_speed=30)
        destiny = spawn_points[94]
        destiny_loc = destiny.location
        vehicle_agent.set_destination((destiny_loc.x, destiny_loc.y, destiny_loc.z))
        target_speed = 10
        # controller = MPCController(target_speed)
        controller = ILQRController(target_speed)
        # try to spawn a static actor at destination
        # for attr in blueprint:
        #     if attr.is_modifiable:
        #         blueprint.set_attribute(attr.id, random.choice(attr.recommended_values))
        # trolley_bp = random.choice(blueprint_library.filter('static.prop.shoppingtrolley')) # vehicle.toyota.prius static.prop.shoppingtrolley
        # trolley_tf = carla.Transform(location=carla.Location(x=destiny_loc.x, y=destiny_loc.y, z=5))
        # print("trolley_bp", trolley_bp, trolley_tf.location)

        # trolley = world.spawn_actor(trolley_bp, trolley_tf)
        # actor_list.append(trolley)
        # trolley.set_simulate_physics(False)

        # pt_loc = carla.Location(x=destiny_loc.x, y=destiny_loc.y, z=0.5)
        # print("debug", world.debug)
        # world.debug.draw_string(pt_loc, 'O')
        # world.debug.draw_point(pt_loc, 100)

        # vehicle.set_autopilot(True)


        # set and spawn actors
        num_wps = 180
        wps_list = []
        last_waypoint = m.get_waypoint(start_pose.location)
        dist = 1

        for i in range(num_wps):
            wp = random.choice(last_waypoint.next(dist))
            wps_list.append(wp)
            last_waypoint = wp

        print("waypoints")
        # print(wps_list)
        for wp in wps_list:
            actor_list.append(spawn_trolley(world, blueprint_library, x=wp.transform.location.x, y=wp.transform.location.y))

        wps_list_np = []
        for wp in wps_list:
            wps_list_np.append([wp.transform.location.x, wp.transform.location.y])

        wps_file = "wps_at_plant.pt"

        pickle.dump(wps_list_np, open(wps_file, 'wb'))

        load_wps = pickle.load(open(wps_file, 'rb'))
        print(load_wps)

        recovered_wps = []
        for wp in load_wps:
            recovered_wps.append(m.get_waypoint(carla.Location(x=wp[0], y=wp[1])))
        print("recovered_wps")
        print(recovered_wps)

  
        # Create a synchronous mode context.
        with CarlaSyncMode(world, camera_rgb, camera_semseg, fps=30) as sync_mode:
            while True:
                if should_quit():
                    return
                if reach_destiny(destiny_loc, vehicle)<0:
                    return
                
                clock.tick(30)

                t = vehicle.get_transform()
                print("vehicle location")
                print(t.location)
                # Advance the simulation and wait for the data.
                snapshot, image_rgb, image_semseg = sync_mode.tick(timeout=2.0)
                
                image_semseg.convert(carla.ColorConverter.CityScapesPalette)
                fps = round(1.0 / snapshot.timestamp.delta_seconds)

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

                # raise ValueError("stop here")

            print('destroying local actors.')
            for actor in actor_list:
                actor.destroy()




    finally:

        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()

        pygame.quit()
        print('done.')


if __name__ == '__main__':

    try:

        main()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
