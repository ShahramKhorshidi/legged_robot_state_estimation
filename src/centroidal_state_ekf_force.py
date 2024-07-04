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
        self._rdata = self._rmodel.createData()
        robot_config = config.get('robot', {})
        self._robot_mass = robot_config.get('mass')
        self._nx = 3 * 3
        self._dt = robot_config.get('dt', 0.001)  # Default dt 0.001
        self._g_vector = np.array([0, 0, -9.81])
        self._end_effectors_frame_names = robot_config.get('end_effectors_frame_names', [])
        self._nb_ee = len(self._end_effectors_frame_names)
        self._mu_pre = np.zeros(self._nx)
        self._mu_post = np.zeros(self._nx)
        self._Sigma_pre = np.zeros((self._nx, self._nx))
        self._Sigma_post = np.zeros((self._nx, self._nx))
        self._Qc = np.eye(3 * self._nb_ee, dtype=np.float64)
        self._Rk = np.eye(self._nx, dtype=np.float64)
        self._Hk = np.eye(self._nx)
        
    # Public methods
    def set_process_noise_cov(self, f_x, f_y, f_z):
        q = np.array(self._nb_ee * [f_x, f_y, f_z])
        np.fill_diagonal(self._Qc, q)

    def set_measurement_noise_cov(self, Rk):
        self._Rk = Rk
    
    def set_mu_post(self, value):
        self._mu_post = value
        
    def update_robot(self):
        pin.forwardKinematics(  self._rmodel, self._rdata, self.q, self.dq)
        pin.framesForwardKinematics(  self._rmodel, self._rdata, self.q)
        
    def compute_ee_position(self):
        self.ee_positions = []
        for i in range(self._nb_ee):
            frame_index =   self._rmodel.getFrameId(
                self._end_effectors_frame_names[i]
            )
            ee_position = self._rdata.oMf[frame_index].translation
            self.ee_positions.append(ee_position)
    
    def update_force_momenta(self):
        self.ee_relative_pos = []
        self.total_force = np.zeros(3)
        self.total_momenta = np.zeros(3)
        com_post = self._mu_post[0:3]
        for index in range(self._nb_ee):
            self.total_force += self.contact[index] * self.ee_force[index]
            self.total_momenta += self.contact[index] * np.cross((self.ee_positions[index] - com_post), self.ee_force[index])
    
    def integrate_model_euler(self):
        com_post = self._mu_post[0:3]
        lin_mom_post = self._mu_post[3:6]
        ang_mom_post = self._mu_post[6:9]
        
        self.update_robot()
        self.compute_ee_position()
        self.update_force_momenta()
        
        # Com position
        self._mu_pre[0:3] = com_post + ((1/self._robot_mass) * lin_mom_post) * self._dt
        
        # linear Momentum
        self._mu_pre[3:6] = lin_mom_post + (self.total_force + self._robot_mass * self._g_vector) * self._dt
        
        # Angular Momentum
        self._mu_pre[6:9] = ang_mom_post + self.total_momenta * self._dt
    
    # Define the derivative function of the state vector (used for RK4 integration)
    def derivative_fn(self, state):
        # Decompose the state vector back into its components
        com, lin_mom, ang_mom = state[:3], state[3:6], state[6:]
        
        # Compute the derivatives based on system's dynamics
        d_com = lin_mom / self._robot_mass
        d_lin_mom = self.total_force + self._robot_mass * self._g_vector
        d_ang_mom = self.total_momenta
        
        # Combine the derivatives into a single vector
        derivatives = np.concatenate((d_com, d_lin_mom, d_ang_mom))
        return derivatives

    def integrate_model_rk4(self):
        # Define the state vector from the current state
        state_post = self._mu_post
        
        # Update robot state and compute necessary dynamics
        self.update_robot()
        self.compute_ee_position()
        self.update_force_momenta()

        # RK4 integration steps
        k1 = self._dt * self.derivative_fn(state_post)
        k2 = self._dt * self.derivative_fn(state_post + 0.5 * k1)
        k3 = self._dt * self.derivative_fn(state_post + 0.5 * k2)
        k4 = self._dt * self.derivative_fn(state_post + k3)

        # Update the state vector with the RK4 approximation
        state_pre = state_post + (k1 + 2*k2 + 2*k3 + k4) / 6

        # Update the class attributes with the new estimated state
        self._mu_pre = state_pre
        
    def compute_discrete_prediction_jacobian(self):
        Fc = np.zeros((self._nx, self._nx), dtype=np.float64)
        Fc[:3, 3:6] = (1/self._robot_mass) * np.eye(3)
        Fc[6:9, :3] = pin.skew(self.total_force)
        Fk = np.eye(self._nx) + Fc * self._dt
        return Fk
        
    def compute_noise_jacobian(self):
        L = []
        com_measured = pin.centerOfMass(  self._rmodel, self._rdata, self.q, False)
        for i in range(self._nb_ee):
            Lc = np.zeros((self._nx, 3), dtype=np.float64)
            Lc[3:6, :] = self.contact[i] * np.eye(3)
            Lc[6:9, :] = self.contact[i] * pin.skew(self.ee_positions[i] - com_measured)
            L.append(Lc)
        return np.hstack((L[0], L[1], L[2], L[3]))
    
    def construct_discrete_noise_covariance(self, Fk, Lc):
        Qk_left = Fk @ Lc
        Qk_right = Lc.T @ Fk.T
        return (Qk_left @ self._Qc @ Qk_right) * self._dt
    
    def prediction_step(self):
        Fk = self.compute_discrete_prediction_jacobian()
        Lc = self.compute_noise_jacobian()
        Qk = self.construct_discrete_noise_covariance(Fk, Lc)
        # Priori error covariance matrix
        self._Sigma_pre = (Fk @ self._Sigma_post @ Fk.T) + Qk
        
    def measurement_model(self):
        y_predicted = np.zeros(9, dtype=float)
        y_measured = np.zeros(9, dtype=float)
        
        y_predicted = self._mu_pre
        
        pin.computeCentroidalMomentum(self._rmodel, self._rdata, self.q, self.dq)
        y_measured[0:3] = self._rdata.com[0]
        y_measured[3:6] = self._rdata.hg.linear
        y_measured[6:9] = self._rdata.hg.angular
        
        measurement_error = y_measured - y_predicted
        return measurement_error
    
    def compute_innovation_covariance(self):
        return (self._Hk @ self._Sigma_pre @ self._Hk.T) + self._Rk
    
    def update_step(self):
        measurement_error = self.measurement_model()
        # Compute kalman gain
        Sk = self.compute_innovation_covariance()
        kalman_gain = (self._Sigma_pre @ self._Hk.T) @ inv(Sk)
        delta_x = kalman_gain @ measurement_error
        self._Sigma_post = (np.eye(self._nx) - (kalman_gain @ self._Hk)) @ self._Sigma_pre
        self._mu_post = self._mu_pre + delta_x
    
    def no_update_step(self):
        self._mu_post = self._mu_pre
        
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
    
    def get_filter_output(self):
        com_position = self._mu_post[0:3]
        lin_momentum = self._mu_post[3:6]
        ang_momentum = self._mu_post[6:9]
        return com_position, lin_momentum, ang_momentum


if __name__ == "__main__":
    cur_dir = Path.cwd()
    robot_urdf = cur_dir/"files"/"go1.urdf"
    robot_config = cur_dir/"files"/"go1_config.yaml"
    solo_cent_ekf = ForceCentroidalEKF(str(robot_urdf), robot_config)
    robot_q = np.random.rand(19)
    robot_dq = np.random.rand(18)
    ee_force = np.random.rand(4, 3)
    contacts_schedule = [True, True, True, True]
    solo_cent_ekf.update_filter(robot_q, robot_dq, contacts_schedule, ee_force)
    com, lin_mom, ang_mom = solo_cent_ekf.get_filter_output()