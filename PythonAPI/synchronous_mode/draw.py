import pickle
import matplotlib.pyplot as plt
import numpy as np

traj_1 = pickle.load(open('trajectory_20190905-121931.p', 'rb'))
traj_23  = pickle.load(open('trajectory_20190905-122757.p', 'rb'))
traj_4 = pickle.load(open('trajectory_safe20200122-010953.p', 'rb'))

def draw(traj, traj_nn, traj_mpc, traj_safe):
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
    
    epi_safe = []
    st_safe = []
    acc_safe = []
    vel_safe = []

    print(len(traj_nn) + len(traj_mpc))
    print(len(traj_safe))
    print(len(traj))
    print(traj[4])

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

    for i in range(len(traj)):
        epi_safe.append(traj_safe[i][0])
        acc_safe.append(traj_safe[i][1])
        st_safe.append(traj_safe[i][2])
        vel_safe.append(traj_safe[i][3])
    
    
    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.scatter(epi_mpc, vel_mpc, s = 10, c = 'b')
    ax1.scatter(epi_nn, vel_nn, s = 10, c = 'r')
    ax2.scatter(epi, vel,  s = 10, c = 'magenta')
    ax2.scatter(epi_safe, vel_safe, s = 10, c = 'cyan')


    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    ax1.set_xlabel('Time Step', fontsize = 20)
    ax1.set_xlim(left = 180, right = 350)
    ax1.set_ylim(bottom = -2, top = 12)
    ax2.set_xlim(left = 180, right = 350)
    ax2.set_ylim(bottom = -2, top = 12)
    ax1.set_ylabel('Velocity (m/s)', fontsize = 20)
    #ax2.set_ylabel('Velocity (m/s) of Enhanced Policy', fontsize = 18)
    ax1.legend(["Initial Policy", "Safety Controller"], loc = 'lower right', fontsize = 16)
    ax2.legend(["Minimally Deviating", "Naive"], loc = 'upper right', fontsize = 16)

    ax1.grid(True, axis = 'both')
    ax2.grid(True, axis = 'both')

    plt.show()


draw(traj_1[0], traj_23[0], traj_23[1], traj_4)
