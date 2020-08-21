import os
import sys

import carla
import pickle
import csv


def config_waypoint_box(target_waypoint, color = (255, 0, 0)):
    extent_wp = carla.Location(0.1, 0.1, 0.1)
    
    if target_waypoint is None:
        box = carla.BoundingBox(carla.Location(), extent_wp)
        rotation=carla.Rotation()
    else:
        box=carla.BoundingBox(target_waypoint.transform.location, extent_wp)
        rotation=target_waypoint.transform.rotation
    life_time= 0.3 * 10
    thickness=0.01 * 1
    color = carla.Color(r=color[0],g=color[1],b=color[2])
    config = {'box': box, \
            'rotation': rotation, \
            'life_time': life_time, \
            'thickness': thickness, 
            'color': color
            }
    return config

def config_measurements_box(measurements, color = (255, 0, 0)):
    extent_m = carla.Location(0.1, 0.1, 0.1)
    
    box=carla.BoundingBox(measurements.t.location, extent_m)
    rotation=measurements.t.rotation
    life_time= .1 * 1
    thickness=0.01 * 1
    color = carla.Color(r=color[0],g=color[1],b=color[2])
    config = {'box': box, \
            'rotation': rotation, \
            'life_time': life_time, \
            'thickness': thickness, 
            'color': color
            }
    return config

def choose_spawn_destination(m, spawn_config, **kwargs):
    """ m: map
        spwan_confif: 0 or 1
        **kwargs(optional):
            spawn_points: list of spawn points
            start_i: index of start spawn point
            end_i: index of end spawn point
    """

    # Spawn configuration
    if spawn_config == 0:
        # Choose from spawn points
        start_i = kwargs['start_i']
        end_i = kwargs['end_i']
        
        if start_i is None:
            print("Spawn from random point")
            start_pose = random.choice(spawn_points)
        else:
            print("Spawn from {}".format(start_i))
            start_pose = kwargs['spawn_points'][start_i]

        if end_i is None:
            print("Go to random destination")
            destination = random.choice(spawn_points)
        else:
            print("Go to {}".format(end_i))
            destination = kwargs['spawn_points'][end_i]

    else:
        #Load spawn point from waypoint file
        wps_file = "wps_at_plant_rotary_0" + str(spawn_config) + ".pt"
        load_wps = pickle.load(open(wps_file, 'rb'))
        print("Spawn from loaded waypoint file %d" % spawn_config)

        # Extract the waypoints
        recovered_wps = []
        for wp in load_wps:
            recovered_wps.append(m.get_waypoint(carla.Location(x=wp[0], y=wp[1])))
        #print("recovered waypoints")
        #print(recovered_wps)

        start_pose = recovered_wps[0].transform
        destination = recovered_wps[-1].transform

    return start_pose, destination

def config_friction(friction_bp, location, extent, scale = 0.001, color = (255, 0, 0)):
    """ Config and spawn a friction event """
    # friction from blueprint
    # Defind bounding box and location
    friction_extent = extent
    friction_bp.set_attribute('friction', str(scale))
    friction_bp.set_attribute('extent_x', str(100 * friction_extent.x))
    friction_bp.set_attribute('extent_y', str(100 * friction_extent.y))
    friction_bp.set_attribute('extent_z', str(100 * friction_extent.z))

    friction_transform = carla.Transform()
    friction_transform.location = location

    box=carla.BoundingBox(friction_transform.location, friction_extent)
    rotation=friction_transform.rotation
    life_time=20
    thickness=0.2
    color=carla.Color(r=color[0],g=color[1],b=color[2])
    
    config = {'box': box, \
            'rotation': rotation, \
            'life_time': life_time, \
            'thickness': thickness, 
            'color': color
            }

    return friction_bp, friction_transform, config
    
def build_file_base(epoch, timestr, spawn_config, number_waypoints, target_speed, **kwargs):
    if 'info' not in kwargs.keys():
        kwargs['info'] = ''
    save_dir_base = "/home/depend/workspace/carla/PythonAPI/synchronous_mode/datasets/"
    csv_dir = save_dir_base 

    if spawn_config == 0:
        csv_dir = csv_dir + "nx_{}_ny_2_from_{}_to_{}_spd_{}_ep_{}_{}_".format(5 + number_waypoints * 3, \
                kwargs['start_i'], kwargs['end_i'], target_speed, epoch, kwargs['info']) +\
                timestr + '.csv'
    else:
        csv_dir = csv_dir + "nx_{}_ny_2_from_{}_wpfile_spd_{}_ep_{}_{}_".format(5 + number_waypoints * 3, \
                spawn_config, target_speed, epoch, kwargs['info']) + \
                timestr + '.csv'

    if not os.path.exists(save_dir_base):
        os.makedirs(save_dir_base)
    with open(csv_dir, 'w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(["throttle", "steer", \
                         "cur_speed", "cur_x", "cur_y", "cur_z", "cur_pitch", "cur_yaw", "cur_roll", \
                         "target_speed", "target_x", "target_y", "target_z", "target_pitch", "target_yaw", "target_roll"])
    return csv_dir
