"""Centroidal Extended Kalman Filter Class
License BSD-3-Clause
Author: Shahram Khorshidi
"""

import yaml
import numpy as np
import pinocchio as pin
from pathlib import Path
from numpy.linalg import inv


class ForceCentroidalEKF(object):
    def __init__(self, urdf_file, config_file):
        # Load configuration from YAML file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self._rmodel = pin.buildModelFromUrdf(urdf_file, pin.JointModelFreeFlyer())
        self.__rdata = self._rmodel.createData()
        robot_config = config.get('robot', {})
        self.__robot_mass = robot_config.get('mass')
        self.__nx = 3 * 3
        self.__dt = robot_config.get('dt', 0.001)  # Default dt 0.001
        self.__g_vector = np.array([0, 0, -9.81])
        self.__end_effectors_frame_names = robot_config.get('end_effectors_frame_names', [])
        self.__nb_ee = len(self.__end_effectors_frame_names)
        self.__mu_pre = np.zeros(self.__nx)
        self.__mu_post = np.zeros(self.__nx)
        self.__Sigma_pre = np.zeros((self.__nx, self.__nx))
        self.__Sigma_post = np.zeros((self.__nx, self.__nx))
        self.__Qc = np.eye(3 * self.__nb_ee, dtype=np.float64)
        self.__Rk = np.eye(self.__nx, dtype=np.float64)
        self.__Hk = np.eye(self.__nx)
        
    # Public methods
    def set_process_noise_cov(self, f_x, f_y, f_z):
        q = np.array(self.__nb_ee * [f_x, f_y, f_z])
        np.fill_diagonal(self.__Qc, q)

    def set_measurement_noise_cov(self, Rk):
        self.__Rk = Rk
    
    def set_mu_post(self, value):
        self.__mu_post = value
        
    def update_robot(self):
        pin.forwardKinematics(self.__rmodel, self.__rdata, self.q, self.dq)
        pin.framesForwardKinematics(self.__rmodel, self.__rdata, self.q)
        
    def compute_ee_position(self):
        self.ee_positions = []
        for i in range(self.__nb_ee):
            frame_index = self.__rmodel.getFrameId(
                self.__end_effectors_frame_names[i]
            )
            ee_position = self.__rdata.oMf[frame_index].translation
            self.ee_positions.append(ee_position)
    
    def update_force_momenta(self):
        self.ee_relative_pos = []
        self.total_force = np.zeros(3)
        self.total_momenta = np.zeros(3)
        com_post = self.__mu_post[0:3]
        for index in range(self.__nb_ee):
            self.total_force += self.contact[index] * self.ee_force[index]
            self.total_momenta += self.contact[index] * np.cross((self.ee_positions[index] - com_post), self.ee_force[index])
    
    def get_ee_relative_positions(self, q):
        ee_pos = np.zeros(12)
        com_measured = pin.centerOfMass(self.__rmodel, self.__rdata, q, False)
        for i in range(self.__nb_ee):
            ee_pos[3*i : 3*(i+1)] = self.contact[i] * (self.ee_positions[i] - com_measured)
        return ee_pos
    
    def integrate_model_euler(self):
        com_post = self.__mu_post[0:3]
        lin_mom_post = self.__mu_post[3:6]
        ang_mom_post = self.__mu_post[6:9]
        
        self.update_robot()
        self.compute_ee_position()
        self.update_force_momenta()
        
        # Com position
        self.__mu_pre[0:3] = com_post + ((1/self.__robot_mass) * lin_mom_post) * self.__dt
        
        # linear Momentum
        self.__mu_pre[3:6] = lin_mom_post + (self.total_force + self.__robot_mass * self.__g_vector) * self.__dt
        
        # Angular Momentum
        self.__mu_pre[6:9] = ang_mom_post + self.total_momenta * self.__dt
    
    # Define the derivative function of the state vector (used for RK4 integration)
    def derivative_fn(self, state):
        # Decompose the state vector back into its components
        com, lin_mom, ang_mom = state[:3], state[3:6], state[6:]
        
        # Compute the derivatives based on system's dynamics
        d_com = lin_mom / self.__robot_mass
        d_lin_mom = self.total_force + self.__robot_mass * self.__g_vector
        d_ang_mom = self.total_momenta
        
        # Combine the derivatives into a single vector
        derivatives = np.concatenate((d_com, d_lin_mom, d_ang_mom))
        return derivatives

    def integrate_model_rk4(self):
        # Define the state vector from the current state
        state_post = self.__mu_post
        
        # Update robot state and compute necessary dynamics
        self.update_robot()
        self.compute_ee_position()
        self.update_force_momenta()

        # RK4 integration steps
        k1 = self.__dt * self.derivative_fn(state_post)
        k2 = self.__dt * self.derivative_fn(state_post + 0.5 * k1)
        k3 = self.__dt * self.derivative_fn(state_post + 0.5 * k2)
        k4 = self.__dt * self.derivative_fn(state_post + k3)

        # Update the state vector with the RK4 approximation
        state_pre = state_post + (k1 + 2*k2 + 2*k3 + k4) / 6

        # Update the class attributes with the new estimated state
        self.__mu_pre = state_pre
        
    def compute_discrete_prediction_jacobian(self):
        Fc = np.zeros((self.__nx, self.__nx), dtype=np.float64)
        Fc[:3, 3:6] = (1/self.__robot_mass) * np.eye(3)
        Fc[6:9, :3] = pin.skew(self.total_force)
        Fk = np.eye(self.__nx) + Fc * self.__dt
        return Fk
        
    def compute_noise_jacobian(self):
        L = []
        com_measured = pin.centerOfMass(self.__rmodel, self.__rdata, self.q, False)
        for i in range(self.__nb_ee):
            Lc = np.zeros((self.__nx, 3), dtype=np.float64)
            Lc[3:6, :] = self.contact[i] * np.eye(3)
            Lc[6:9, :] = self.contact[i] * pin.skew(self.ee_positions[i] - com_measured)
            L.append(Lc)
        return np.hstack((L[0], L[1], L[2], L[3]))
    
    def construct_discrete_noise_covariance(self, Fk, Lc):
        Qk_left = Fk @ Lc
        Qk_right = Lc.T @ Fk.T
        return (Qk_left @ self.__Qc @ Qk_right) * self.__dt
    
    def prediction_step(self):
        Fk = self.compute_discrete_prediction_jacobian()
        Lc = self.compute_noise_jacobian()
        Qk = self.construct_discrete_noise_covariance(Fk, Lc)
        # Priori error covariance matrix
        self.__Sigma_pre = (Fk @ self.__Sigma_post @ Fk.T) + Qk
        
    def measurement_model(self):
        y_predicted = np.zeros(9, dtype=float)
        y_measured = np.zeros(9, dtype=float)
        
        y_predicted = self.__mu_pre
        
        pin.computeCentroidalMomentum(self.__rmodel, self.__rdata, self.q, self.dq)
        y_measured[0:3] = self.__rdata.com[0]
        y_measured[3:6] = self.__rdata.hg.linear
        y_measured[6:9] = self.__rdata.hg.angular
        
        measurement_error = y_measured - y_predicted
        return measurement_error
    
    def compute_innovation_covariance(self):
        return (self.__Hk @ self.__Sigma_pre @ self.__Hk.T) + self.__Rk
    
    def update_step(self):
        measurement_error = self.measurement_model()
        # Compute kalman gain
        Sk = self.compute_innovation_covariance()
        kalman_gain = (self.__Sigma_pre @ self.__Hk.T) @ inv(Sk)
        delta_x = kalman_gain @ measurement_error
        self.__Sigma_post = (np.eye(self.__nx) - (kalman_gain @ self.__Hk)) @ self.__Sigma_pre
        self.__mu_post = self.__mu_pre + delta_x
    
    def no_update_step(self):
        self.__mu_post = self.__mu_pre
        
    def update_filter(self, q, dq, contact_scedule, ee_force, integration_method="euler"):
        self.q = q
        self.dq = dq
        self.ee_force = ee_force
        self.contact = contact_scedule
        
        # Update robot state and compute necessary dynamics
        self.update_robot()
        self.compute_ee_position()
        self.update_force_momenta()
        if integration_method == "euler":
            self.integrate_model_euler()
        if integration_method == "rk4":
            self.integrate_model_rk4()
        self.prediction_step()
        self.update_step()
    
    def update_values(self, q, dq, contact_scedule, ee_force):
        self.q = q
        self.dq = dq
        self.ee_force = ee_force
        self.contact = contact_scedule
        
        # Update robot state and compute necessary dynamics
        self.update_robot()
        self.compute_ee_position()
        self.update_force_momenta()
    
    def get_filter_output(self):
        com_position = self.__mu_post[0:3]
        lin_momentum = self.__mu_post[3:6]
        ang_momentum = self.__mu_post[6:9]
        return com_position, lin_momentum, ang_momentum


if __name__ == "__main__":
    cur_dir = Path.cwd()
    robot_urdf = cur_dir/"files"/"go1.urdf"
    robot_config = cur_dir/"files"/"go1_config.yaml"
    solo_cent_ekf = ForceCentroidalEKF(str(robot_urdf), robot_config)
    robot_q = np.random.rand(19)
    robot_dq = np.random.rand(18)
    ee_force = np.random.rand(4, 3)
    solo_cent_ekf.update_filter(robot_q, robot_dq, ee_force)
    com, lin_mom, ang_mom = solo_cent_ekf.get_filter_output()