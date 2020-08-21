import numpy as np
import csv

# set MACROS for data normalization
max_pos_val = 500
max_yaw_val = 180
max_speed_val = 40


traj_paths = ["./initial_friction_False", "./initial_friction_True", "./naive_friction_False", "./minimally_deviating_friction_False"]

for traj_path in traj_paths:
    pos = []
    vel = []

    with open(traj_path + ".csv") as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            row = [float(i) for i in row]
            pos.append(np.asarray([row[1] * max_pos_val, row[2] * max_pos_val]))
            vel.append(vel * max_speed_val)
        pos = np.stack(pos)
        vel = np.stack(vel)

        pos_smoothness = np.var(np.diff(pos, axis = 0))
        vel_smoothness = np.var(np.diff(vel, axis = 0))
        print(traj_path)
        print("Position smoothness: {}\nVelocity smoothness: {}\n".format(pos_smoothness, vel_smoothness))

