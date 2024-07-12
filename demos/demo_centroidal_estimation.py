import numpy as np
import pinocchio as pin
from pathlib import Path
import matplotlib.pyplot as plt
from src.centroidal_state_ekf_force import ForceCentroidalEKF
from src.centroidal_state_ekf_torque import TorqueCentroidalEKF

def plot(x, y, z, x_legend, y_legend, z_legend, title):
    t = np.arange(sim_time)
    string = "XYZ"
    for i in range(3):
        plt.subplot(int("31" + str(i + 1)))
        plt.plot(t, x[:, i], "b--", label=x_legend, linewidth=0.85)
        plt.plot(t, y[:, i], "g", label=y_legend, linewidth=0.8)
        plt.plot(t, z[:, i], "r", label=z_legend, linewidth=0.8)
        plt.ylabel("_" + string[i] + "_")
        plt.grid()
    plt.legend(loc="upper right", shadow=True, fontsize="large")
    plt.xlabel("time(ms)")
    plt.suptitle(title)


if __name__ == "__main__":
    # Read the trajectory data - Trot motion of Unitree Go1
    path = Path.cwd()
    robot_q = np.loadtxt(path/"data"/"robot_q.dat", delimiter='\t', dtype=np.float32)
    robot_dq = np.loadtxt(path/"data"/"robot_dq.dat", delimiter='\t', dtype=np.float32)
    cent_states = np.loadtxt(path/"data"/"cent_state.dat", delimiter='\t', dtype=np.float32)
    contacts_schedule = np.loadtxt(path/"data"/"robot_cnt.dat", delimiter='\t', dtype=np.float32)
    ee_force = np.loadtxt(path/"data"/"ee_force.dat", delimiter='\t', dtype=np.float32)
    joint_torques = np.loadtxt(path/"data"/"robot_tau.dat", delimiter='\t', dtype=np.float32)

    # Initialize vectors for data collecting
    sim_time = robot_q.shape[1]
    com_pos_f = np.zeros((sim_time, 3), float)
    lin_mom_f = np.zeros((sim_time, 3), float)
    ang_mom_f = np.zeros((sim_time, 3), float)
    com_pos_t = np.zeros((sim_time, 3), float)
    lin_mom_t = np.zeros((sim_time, 3), float)
    ang_mom_t = np.zeros((sim_time, 3), float)
    
    # Create EKF instances
    robot_urdf = path/"files"/"go1.urdf"
    robot_config = path/"files"/"go1_config.yaml"
    
    # Centroidal state estimation with contact force measurements
    robot_cent_ekf_force = ForceCentroidalEKF(str(robot_urdf), robot_config)
    # Tuning parameters
    robot_cent_ekf_force.set_process_noise_cov(1e-1, 1e-1, 1e-1)
    R_ekf = np.diag(np.array([1e-2, 1e-2, 1e-2] + [1e-2, 1e-2, 1e-2] + [1e-4, 1e-4, 1e-4]))
    robot_cent_ekf_force.set_measurement_noise_cov(R_ekf)
    
    # Centroidal state estimation with joint torque measurements
    robot_cent_ekf_torque = TorqueCentroidalEKF(str(robot_urdf), robot_config)
    # Tuning parameters
    robot_cent_ekf_torque.set_process_noise_cov(1e-7, 1e-5, 1e-4)
    robot_cent_ekf_torque.set_measurement_noise_cov(1e-5, 1e-5, 1e-6)
    
    for i in range(sim_time):
        # Set the initial values of EKF
        if i == 0:
            robot_cent_ekf_force.set_mu_post(cent_states[:, 0])
            robot_cent_ekf_torque.set_mu_post(cent_states[:, 0])
        
        # Run the EKF
        robot_cent_ekf_force.update_filter(
            robot_q[:, i],
            robot_dq[:, i],
            contacts_schedule[:, i],
            ee_force[:, i]
        )
        
        robot_cent_ekf_torque.update_filter(
            robot_q[:, i],
            robot_dq[:, i],
            contacts_schedule[:, i],
            joint_torques[:, i],
        )
        
        # Read the values of estimated centroidla states from EKF
        com_pos_f[i, :], lin_mom_f[i, :], ang_mom_f[i, :] = robot_cent_ekf_force.get_filter_output()
        com_pos_t[i, :], lin_mom_t[i, :], ang_mom_t[i, :] = robot_cent_ekf_torque.get_filter_output()
        
    # Plot the results
    plt.figure("COM Position")
    plot(cent_states[:3,:].T, com_pos_f, com_pos_t, "Measured", "EKF_force", "EKF_torque","COM_Position (m)")

    plt.figure("COM Linear Momentum")
    plot(cent_states[3:6,:].T, lin_mom_f, lin_mom_t, "Measured", "EKF_force", "EKF_torque", "COM_Linear_Momentum (kgm/s)")

    plt.figure("COM Angular Momentum")
    plot(cent_states[6:,:].T, ang_mom_f, ang_mom_t, "Measured", "EKF_force", "EKF_torque", "COM_Angular_Momentum (kgm^2/s)",)
    
    plt.show()