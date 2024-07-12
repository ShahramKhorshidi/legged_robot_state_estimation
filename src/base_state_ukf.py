"""Extended Kalman Filter Class
License BSD-3-Clause
Copyright (c) 2021, New York University and Max Planck Gesellschaft.
Authors: Shahram Khorshidi
"""

import yaml
import numpy as np
import pinocchio as pin
from pathlib import Path
from numpy.linalg import inv
from scipy.linalg import sqrtm
from scipy.linalg import block_diag
from scipy.linalg import cholesky
from base_state_ekf import BaseEKF


# class UKF(object):
#     def __init__(self, urdf_file, config_file):
#         # Load configuration from YAML file
#         with open(config_file, 'r') as file:
#             config = yaml.safe_load(file)
#         self._rmodel = pin.buildModelFromUrdf(urdf_file, pin.JointModelFreeFlyer())
#         self.__rdata = self._rmodel.createData()
#         robot_config = config.get('robot', {})
#         self.__robot_mass = robot_config.get('mass')
#         self.__nx = 3 * 3
#         self.__dt = robot_config.get('dt', 0.001)  # Default dt 0.001
#         self.__g_vector = np.array([0, 0, -9.81])
#         self.__end_effectors_frame_names = robot_config.get('end_effectors_frame_names')
#         self.__rot_base_to_imu = np.array(robot_config.get('rot_base_to_imu'))
#         self.__r_base_to_imu = np.array(robot_config.get('r_base_to_imu'))
#         self.__SE3_imu_to_base = pin.SE3(
#             self.__rot_base_to_imu.T, self.__r_base_to_imu
#         )

class BaseUKF(BaseEKF):
    def __init__(self, robot_config, dt=0.001, alpha=0.1, beta=2.0, kappa=0.0):
        super().__init__(robot_config, dt)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambd = alpha**2 * (self._nx + kappa) - self._nx

    def compute_sigma_points(self, mu, Sigma):
        n = len(mu)
        scale = np.sqrt(n + self.lambd)
        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = mu
        sigma_points[1:n+1] = mu + scale * cholesky(Sigma).T
        sigma_points[n+1:] = mu - scale * cholesky(Sigma).T
        return sigma_points

    def _prediction_step(self):
        mu_pre = np.zeros(self._nx)
        Sigma_pre = np.zeros((self._nx, self._nx))

        # Compute sigma points
        sigma_points = self.compute_sigma_points(self._mu_post, self._Sigma_post)

        # Propagate sigma points through the nonlinear process model
        for i in range(len(sigma_points)):
            sigma_point = sigma_points[i]
            mu_pre += sigma_point * self._dt
        mu_pre /= len(sigma_points)

        # Compute the predicted mean and covariance
        for i in range(len(sigma_points)):
            diff = sigma_points[i] - mu_pre
            Sigma_pre += np.outer(diff, diff)
        Sigma_pre /= len(sigma_points)

        self.__mu_pre = mu_pre
        self.__Sigma_pre = Sigma_pre + self._construct_continuous_noise_covariance()

    def _measurement_model(self, contacts_schedule, joint_positions, joint_velocities):
        Hk = np.zeros((3 * self._nb_ee, self._nx))
        predicted_frame_velocity = np.zeros(3 * self._nb_ee)
        measured_frame_velocity = np.zeros(3 * self._nb_ee)

        # Compute sigma points
        sigma_points = self.compute_sigma_points(self.__mu_pre, self.__Sigma_pre)

        # Propagate sigma points through the measurement model
        for i in range(len(sigma_points)):
            sigma_point = sigma_points[i]
            Hk_point, _ = super()._measurement_model(
                contacts_schedule, joint_positions, joint_velocities
            )
            predicted_frame_velocity += Hk_point @ sigma_point

        # Compute the predicted mean and covariance
        predicted_frame_velocity /= len(sigma_points)

        for i in range(len(sigma_points)):
            diff = Hk @ sigma_points[i] - predicted_frame_velocity
            measured_frame_velocity += np.outer(diff, diff)
        measured_frame_velocity /= len(sigma_points)

        return Hk, measured_frame_velocity + self._construct_discrete_measurement_noise_covariance()

    def _update_step(self, contacts_schedule, joint_positions, joint_velocities):
        Hk, measurement_error = self._measurement_model(
            contacts_schedule, joint_positions, joint_velocities
        )
        Rk = self._construct_discrete_measurement_noise_covariance()
        Sk = self._compute_innovation_covariance(Hk, Rk)
        K = (self.__Sigma_pre @ Hk.T) @ inv(Sk)  # kalman gain
        delta_x = K @ measurement_error
        self.__Sigma_post = (np.eye(self._nx) - (K @ Hk)) @ self.__Sigma_pre
        self.__mu_post = self.__mu_pre + delta_x

    def get_filter_output(self):
        return self.__mu_post


if __name__ == "__main__":
    cur_dir = Path.cwd()
    robot_urdf = cur_dir/"files"/"go1.urdf"
    robot_config = cur_dir/"files"/"go1_config.yaml"
    robot_base_ekf = UKF(str(robot_urdf), robot_config)
    robot_base_ekf.set_meas_noise_cov(np.array([1e-4, 1e-4, 1e-4]))
    f_tilde = np.random.rand(3)
    w_tilde = np.random.rand(3)
    contacts_schedule = [True, True, True, True]
    joint_positions = np.random.rand(12)
    joint_velocities = np.random.rand(12)
    robot_base_ekf.update_filter(
        f_tilde,
        w_tilde,
        contacts_schedule,
        joint_positions,
        joint_velocities,
    )
    base_state = robot_base_ekf.get_filter_output()