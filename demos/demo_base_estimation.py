import numpy as np
import pinocchio as pin
from pathlib import Path
from src.base_state_ekf import EKF
import matplotlib.pyplot as plt

def plot(x, y, x_legend, y_legend, title):
    t = np.arange(sim_time)
    string = "XYZ"
    for i in range(3):
        plt.subplot(int("31" + str(i + 1)))
        plt.plot(t, x[:, i], "b", label=x_legend, linewidth=0.75)
        plt.plot(t, y[:, i], "r--", label=y_legend, linewidth=0.75)
        plt.ylabel("_" + string[i] + "_")
        plt.grid()
    plt.legend(loc="upper right", shadow=True, fontsize="large")
    plt.xlabel("time(ms)")
    plt.suptitle(title)


if __name__ == "__main__":
    # Read the trajectory data - Jumping motion
    path = Path.cwd()
    q = np.loadtxt(path/"data"/"robot_q.dat", delimiter='\t', dtype=np.float32)
    dq = np.loadtxt(path/"data"/"robot_dq.dat", delimiter='\t', dtype=np.float32)
    contacts_schedule = np.loadtxt(path/"data"/"robot_cnt.dat", delimiter='\t', dtype=np.float32)
    imu_lin_acc = np.loadtxt(path/"data"/"robot_imu.dat", delimiter='\t', dtype=np.float32)[:3, :]
    imu_ang_vel = np.loadtxt(path/"data"/"robot_imu.dat", delimiter='\t', dtype=np.float32)[3:, :]
    
    # Initialize vectors for data collecting
    sim_time = q.shape[1]
    base_pos = np.zeros((sim_time, 3), float)
    base_vel = np.zeros((sim_time, 3), float)
    base_rpy = np.zeros((sim_time, 3), float)
    base_pos_ekf = np.zeros((sim_time, 3), float)
    base_vel_ekf = np.zeros((sim_time, 3), float)
    base_rpy_ekf = np.zeros((sim_time, 3), float)
    
    # Create EKF instance
    robot_urdf = path/"files"/"go1.urdf"
    robot_config = path/"files"/"go1_config.yaml"
    robot_base_ekf = EKF(str(robot_urdf), robot_config)
    robot_base_ekf.set_meas_noise_cov(np.array([1e-5, 1e-5, 1e-5]))

    for i in range(sim_time):
        # Set the initial values of EKF
        if i == 0:
            robot_base_ekf.set_mu_post("ekf_frame_position", q[:3, 0])
            robot_base_ekf.set_mu_post("ekf_frame_velocity", dq[:3, 0])
            robot_base_ekf.set_mu_post("ekf_frame_orientation", pin.Quaternion(q[3:7, 0]))
        
        # Run the EKF
        robot_base_ekf.update_filter(
            imu_lin_acc[:, i],
            imu_ang_vel[:, i],
            contacts_schedule[:, i],
            q[7:, i], # joint_positions
            dq[6:, i],# joint_velocities
        )
        
        # Read the values of position, velocity and orientation of the base from robot trajectory (ground truth)
        base_pos[i, :] = q[:3, i]
        base_vel[i, :] = dq[:3, i]
        q_base = pin.Quaternion(q[3:7, i])
        base_rpy[i, :] = pin.utils.matrixToRpy(q_base.matrix())
        
        # Read the values of position, velocity and orientation of estimated states from EKF
        base_state = robot_base_ekf.get_filter_output()
        base_pos_ekf[i, :] = base_state.get("base_position")
        base_vel_ekf[i, :] = base_state.get("base_velocity")
        q_ekf = base_state.get("base_orientation")
        base_rpy_ekf[i, :] = pin.utils.matrixToRpy(q_ekf.matrix())
        
    # Plot the results
    plt.figure("Position")
    plot(base_pos, base_pos_ekf, "Sim_data", "EKF_data", "Base_Position (m)")

    plt.figure("Velocity")
    plot(base_vel, base_vel_ekf, "Sim_data", "EKF_data", "Base_Linear_Velocity (m/s)")

    plt.figure("Orientation")
    plot(
        base_rpy,
        base_rpy_ekf,
        "Sim_data",
        "EKF_data",
        "Base_Orientation(roll-pitch-yaw) (rad)",
    )
    plt.show()