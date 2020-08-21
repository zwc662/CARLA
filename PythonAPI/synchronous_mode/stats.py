import numpy as np
import csv

# set MACROS for data normalization
max_pos_val = 500
max_yaw_val = 180
max_speed_val = 40


traj_paths = ["./initial_friction_False", "./initial_friction_True", "./initial_SC_friction_True", "./naive_friction_True", "./minimally_deviating_friction_True"]

def deriv(array):
    diff = np.diff(array, axis = 0)
    angle_ = np.reshape(np.arctan(diff[:, 1]/diff[:, 0]), [diff.shape[0], 1])
    deriv = np.concatenate((np.cos(angle_), np.sin(angle_)), axis = 1)
    return deriv

def poly_fit(array, deg):
    poly = np.polyfit(np.arange(np.shape(array)[0]), array, deg, full = True)
    poly_res = np.sum(poly[1])
    poly_res = np.sqrt(poly_res)
    return poly_res


for traj_path in traj_paths:
    pos = []
    wpt = []
    vel = []
    acc = []

    with open(traj_path + ".csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        print(traj_path)
        for row in csv_reader:
            row = [float(i) for i in row]
            pos.append(np.asarray([row[1 + 2], row[2 + 2]]) * max_pos_val)
            vel.append(np.asarray([row[0 + 2], row[0 + 2]]) * max_speed_val)
            wpt.append(np.asarray([row[5 + 2], row[6 + 2]]) * max_pos_val)

        """
        pos = np.stack(pos[-450: ], axis = 0)
        vel_direc = deriv(pos)
        pos = pos - np.mean(pos, axis = 0)

        vel = np.stack(vel[-np.shape(vel_direc)[0]:], axis = 0)
        vel = vel * vel_direc
        acc_dirc = deriv(vel)
        vel = vel - np.mean(vel, axis = 0)

        acc = acc_dirc
        acc = acc - np.mean(acc, axis = 0)
        

        pos_var = np.sum(np.var(pos, axis = 0))
        vel_var = np.sum(np.var(np.diff(pos, axis = 0), axis = 0))
        acc_var = np.sum(np.var(np.diff(pos, n = 2, axis = 0), axis = 0))
        print("Position smoothness: {}\nVelocity smoothness: {}\n Acceleration smoothness: {}".format(
                pos_var, vel_var, acc_var))
        """

        
        vel = np.stack(vel[:], axis = 0)
        print("Velocity 0: {}\nVelocity 1: {}\nVelocity 2: {}\n".format(np.mean(vel, axis = 0), np.var(np.diff(vel, axis = 0), axis = 0), np.var(np.diff(np.diff(vel, axis = 0), axis = 0), axis = 0)))
        dist = np.linalg.norm(np.stack(wpt[:], axis = 0) - np.stack(pos[:], axis = 0), ord = 2, axis = 1)
        #dist = dist[175:400]
        print("Distance 0: {}\nDistance 1: {}\nDistance 2: {}\n".format(np.mean(dist, axis = 0), np.var(np.diff(dist, axis = 0), axis = 0), np.var(np.diff(np.diff(dist, axis = 0), axis = 0), axis = 0)))
        continue

        for deg in range(5): 
            poly_pos_res = poly_fit(pos, deg)
            poly_vel_res = poly_fit(vel, deg)
            poly_acc_res = poly_fit(acc, deg)


            print("Position smoothness: {}\nVelocity smoothness: {}\n Acceleration smoothness: {}".format(
                poly_pos_res, poly_vel_res, poly_acc_res))

