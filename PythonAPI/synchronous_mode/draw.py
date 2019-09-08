import pickle
import matplotlib.pyplot as plt
import numpy as np

traj_1 = pickle.load(open('trajectory_20190905-121931.p', 'rb'))
traj_23  = pickle.load(open('trajectory_20190905-122757.p', 'rb'))

def draw(traj, traj_nn, traj_mpc):
    epi = []
    st = []
    acc = []
    vel = []

    epi_nn = []
    st_nn = []
    acc_nn = []
    vel_nn = []

    epi_mpc = []
    st_mpc = []
    acc_mpc = []
    vel_mpc = []
    

    print(len(traj))
    for i in range(len(traj)):
        epi.append(traj[i][0])
        acc.append(traj[i][1])
        st.append(traj[i][2])
        vel.append(traj[i][3])


    for i in range(len(traj_nn)):
        epi_nn.append(traj_nn[i][0])
        acc_nn.append(np.clip(traj_nn[i][1], 0.0, 1.0))
        st_nn.append(np.clip(traj_nn[i][2], -1.0, 1.0))
        vel_nn.append(traj_nn[i][3])

    for i in range(len(traj_mpc)):
        epi_mpc.append(traj_mpc[i][0])
        acc_mpc.append(np.clip(traj_mpc[i][1], 0.0, 1.0))
        st_mpc.append(np.clip(traj_mpc[i][2], -1.0, 1.0))
        vel_mpc.append(traj_mpc[i][3])
    
    
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.scatter(epi_mpc, vel_mpc, s = 10, c = 'b')
    ax1.scatter(epi_nn, vel_nn, s = 10, c = 'r')
    ax2.scatter(epi, vel, s = 10, c = 'g')

    ax1.set_xlabel('Time Step', fontsize = 18)
    ax1.set_xlim(left = 0, right = 500)
    ax1.set_ylim(bottom = -2, top = 12)
    ax2.set_xlim(left = 0, right = 500)
    ax2.set_ylim(bottom = -2, top = 12)
    ax1.set_ylabel('Velocity (m/s) of Initial Policy', fontsize = 18)
    ax2.set_ylabel('Velocity (m/s) of Enhanced Policy', fontsize = 18)
    ax1.legend(["Initial Policy", "Safety Controller"], loc = 'lower left', fontsize = 18)
    ax2.legend(["Enhanced Policy"], loc = 'upper right', fontsize = 18)

    ax1.grid(True, axis = 'both')
    ax2.grid(True, axis = 'both')

    plt.show()


draw(traj_1[0], traj_23[0], traj_23[1])
