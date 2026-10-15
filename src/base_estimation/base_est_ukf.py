"""Unscented Kalman filter for legged-robot base-state estimation.

License BSD-3-Clause
Copyright (c) 2023, University of Bonn.
Author: Shahram Khorshidi

The filter uses the same state, inputs, and output convention as ``BaseEKF``.
The covariance lives in a 15-dimensional local error space while orientation
itself remains on SO(3). Sigma-point rotations are generated with the
exponential map and averaged with the logarithmic map.

The scaled unscented transform follows Wan and van der Merwe, "The Unscented
Kalman Filter for Nonlinear Estimation," 2000. The local retraction approach is
the standard manifold adaptation described by Brossard, Barrau, and Bonnabel,
"A Code for Unscented Kalman Filtering on Manifolds," 2020.
"""

import numpy as np
import pinocchio as pin
from pinocchio import Quaternion

from .base_est_ekf import BaseEKF, box_minus, box_plus


class BaseUKF(BaseEKF):
    """Manifold UKF for base pose, velocity, and IMU biases.

    The local error vector has the same order as :class:`BaseEKF`:

    ``delta_x = [delta_p, delta_v, delta_theta, delta_b_a, delta_b_omega]``.

    Position is expressed in the world frame, velocity in the selected EKF
    frame, and the orientation perturbation is right-multiplied on SO(3).
    """

    def __init__(
        self,
        urdf_file,
        robot_config,
        dt=0.001,
        alpha=0.3,
        beta=2.0,
        kappa=0.0,
    ):
        """Initializes the UKF.

        Args:
            urdf_file: Path to the robot URDF.
            robot_config: Path to the robot YAML configuration.
            dt (float): Discretization time.
            alpha (float): Sigma-point spread parameter.
            beta (float): Prior-distribution parameter. ``2`` is suitable for
                Gaussian state errors.
            kappa (float): Secondary sigma-point scaling parameter.
        """
        super().__init__(urdf_file, robot_config, dt)

        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if self._nx + kappa <= 0.0:
            raise ValueError("nx + kappa must be positive")

        self._alpha = float(alpha)
        self._beta = float(beta)
        self._kappa = float(kappa)
        self._lambda = self._alpha**2 * (self._nx + self._kappa) - self._nx
        self._sigma_scale = self._nx + self._lambda
        if self._sigma_scale <= 0.0:
            raise ValueError("UKF sigma-point scale must be positive")

        number_of_sigma_points = 2 * self._nx + 1
        self._weights_mean = np.full(
            number_of_sigma_points, 1.0 / (2.0 * self._sigma_scale)
        )
        self._weights_cov = self._weights_mean.copy()
        self._weights_mean[0] = self._lambda / self._sigma_scale
        self._weights_cov[0] = (
            self._weights_mean[0] + 1.0 - self._alpha**2 + self._beta
        )

        self._a_tilde = np.zeros(3)
        self._omega_tilde = np.zeros(3)
        self._omega_tilde_prev = np.zeros(3)
        self._has_previous_imu = False

    # Private methods
    @staticmethod
    def _copy_state(state):
        quaternion = Quaternion(state["ekf_frame_orientation"].coeffs())
        quaternion.normalize()
        return {
            "ekf_frame_position": state["ekf_frame_position"].copy(),
            "ekf_frame_velocity": state["ekf_frame_velocity"].copy(),
            "ekf_frame_orientation": quaternion,
            "imu_bias_acceleration": state["imu_bias_acceleration"].copy(),
            "imu_bias_orientation": state["imu_bias_orientation"].copy(),
        }

    def _state_box_plus(self, state, delta_x):
        """Applies a 15-dimensional local perturbation to a filter state."""
        rotation = state["ekf_frame_orientation"].matrix()
        quaternion = Quaternion(box_plus(rotation, delta_x[6:9]))
        quaternion.normalize()
        return {
            "ekf_frame_position": state["ekf_frame_position"] + delta_x[0:3],
            "ekf_frame_velocity": state["ekf_frame_velocity"] + delta_x[3:6],
            "ekf_frame_orientation": quaternion,
            "imu_bias_acceleration": (
                state["imu_bias_acceleration"] + delta_x[9:12]
            ),
            "imu_bias_orientation": (
                state["imu_bias_orientation"] + delta_x[12:15]
            ),
        }

    def _state_box_minus(self, state, reference):
        """Returns the local error that maps ``reference`` to ``state``."""
        delta_x = np.zeros(self._nx)
        delta_x[0:3] = (
            state["ekf_frame_position"] - reference["ekf_frame_position"]
        )
        delta_x[3:6] = (
            state["ekf_frame_velocity"] - reference["ekf_frame_velocity"]
        )
        delta_x[6:9] = box_minus(
            state["ekf_frame_orientation"].matrix(),
            reference["ekf_frame_orientation"].matrix(),
        )
        delta_x[9:12] = (
            state["imu_bias_acceleration"]
            - reference["imu_bias_acceleration"]
        )
        delta_x[12:15] = (
            state["imu_bias_orientation"]
            - reference["imu_bias_orientation"]
        )
        return delta_x

    @staticmethod
    def _covariance_square_root(covariance, scale):
        """Returns a stable square root of a symmetric PSD covariance."""
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tolerance = 1e-12 * max(1.0, np.max(np.abs(eigenvalues)))
        if eigenvalues[0] < -tolerance:
            raise ValueError("UKF covariance is not positive semidefinite")
        eigenvalues = np.maximum(eigenvalues, 0.0)
        return eigenvectors @ np.diag(np.sqrt(scale * eigenvalues))

    def _compute_sigma_points(self, mean, covariance):
        """Creates sigma points in the local error space and retracts them."""
        square_root = self._covariance_square_root(
            covariance, self._sigma_scale
        )
        sigma_points = [self._copy_state(mean)]
        for i in range(self._nx):
            sigma_points.append(self._state_box_plus(mean, square_root[:, i]))
            sigma_points.append(self._state_box_plus(mean, -square_root[:, i]))
        return sigma_points

    def _compute_state_mean(self, sigma_points):
        """Computes the weighted state mean, including the SO(3) mean."""
        mean = self._copy_state(sigma_points[0])
        for key in (
            "ekf_frame_position",
            "ekf_frame_velocity",
            "imu_bias_acceleration",
            "imu_bias_orientation",
        ):
            mean[key] = sum(
                weight * state[key]
                for weight, state in zip(self._weights_mean, sigma_points)
            )

        rotation_mean = mean["ekf_frame_orientation"].matrix()
        for _ in range(20):
            mean_error = sum(
                weight
                * box_minus(
                    state["ekf_frame_orientation"].matrix(), rotation_mean
                )
                for weight, state in zip(self._weights_mean, sigma_points)
            )
            if np.linalg.norm(mean_error) < 1e-12:
                break
            rotation_mean = box_plus(rotation_mean, mean_error)

        quaternion = Quaternion(rotation_mean)
        quaternion.normalize()
        mean["ekf_frame_orientation"] = quaternion
        return mean

    def _corrected_angular_velocity(self, state):
        omega = self._omega_tilde - state["imu_bias_orientation"]
        if self._ekf_in_imu_frame:
            return omega
        return self._SE3_imu_to_base.rotation @ omega

    def _propagate_sigma_point(self, state):
        """Propagates one sigma point through the nonlinear IMU model."""
        dt = self._dt
        gravity = self._g_vector
        rotation = state["ekf_frame_orientation"].matrix()
        velocity = state["ekf_frame_velocity"]
        bias_acceleration = state["imu_bias_acceleration"]
        omega_hat = self._corrected_angular_velocity(state)

        if self._ekf_in_imu_frame:
            acceleration_hat = self._a_tilde - bias_acceleration
        else:
            rotation_imu_to_base = self._SE3_imu_to_base.rotation
            lever_arm = self._SE3_imu_to_base.translation
            angular_acceleration = rotation_imu_to_base @ (
                self._omega_tilde - self._omega_tilde_prev
            ) / dt
            acceleration_hat = (
                rotation_imu_to_base @ (self._a_tilde - bias_acceleration)
                + np.cross(angular_acceleration, -lever_arm)
                + np.cross(omega_hat, np.cross(omega_hat, -lever_arm))
            )

        rotation_pre = box_plus(rotation, omega_hat * dt)
        quaternion = Quaternion(rotation_pre)
        quaternion.normalize()
        return {
            "ekf_frame_position": (
                state["ekf_frame_position"] + rotation @ velocity * dt
            ),
            "ekf_frame_velocity": (
                velocity
                + (
                    -pin.skew(omega_hat) @ velocity
                    + rotation.T @ gravity
                    + acceleration_hat
                )
                * dt
            ),
            "ekf_frame_orientation": quaternion,
            "imu_bias_acceleration": bias_acceleration.copy(),
            "imu_bias_orientation": state["imu_bias_orientation"].copy(),
        }

    def _construct_ukf_process_noise_covariance(self):
        """Constructs discrete process noise in the local error coordinates."""
        velocity = self._mu_pre["ekf_frame_velocity"]
        noise_rotation = (
            np.eye(3)
            if self._ekf_in_imu_frame
            else self._SE3_imu_to_base.rotation
        )
        noise_jacobian = np.zeros((self._nx, self._nx - 3))
        noise_jacobian[3:6, 0:3] = -noise_rotation
        noise_jacobian[3:6, 3:6] = -pin.skew(velocity) @ noise_rotation
        noise_jacobian[6:9, 3:6] = -noise_rotation
        noise_jacobian[9:12, 6:9] = np.eye(3)
        noise_jacobian[12:15, 9:12] = np.eye(3)
        continuous_noise = self._construct_continuous_noise_covariance()

        # BaseEKF stores accelerometer and gyroscope noise with an initial dt
        # factor. Remove it because the discretization below applies dt once.
        continuous_noise[0:3, 0:3] /= self._dt
        continuous_noise[3:6, 3:6] /= self._dt
        return (
            noise_jacobian
            @ continuous_noise
            @ noise_jacobian.T
            * self._dt
        )

    def _prediction_step(self):
        """Propagates the sigma points and computes the prior covariance."""
        sigma_points = self._compute_sigma_points(
            self._mu_post, self._Sigma_post
        )
        propagated_points = [
            self._propagate_sigma_point(state) for state in sigma_points
        ]
        self._mu_pre = self._compute_state_mean(propagated_points)

        covariance = np.zeros((self._nx, self._nx))
        for weight, state in zip(self._weights_cov, propagated_points):
            error = self._state_box_minus(state, self._mu_pre)
            covariance += weight * np.outer(error, error)

        self._omega_hat = self._corrected_angular_velocity(self._mu_pre)
        covariance += self._construct_ukf_process_noise_covariance()
        self._Sigma_pre = 0.5 * (covariance + covariance.T)

    def _measurement_model(
        self,
        sigma_points,
        contacts_schedule,
        joint_positions,
        joint_velocities,
    ):
        """Propagates sigma points through the no-slip leg-velocity model."""
        ee_positions, ee_velocities = self._compute_end_effectors_FK_quantities(
            joint_positions, joint_velocities
        )
        measurements = np.zeros(
            (len(sigma_points), 3 * self._nb_ee), dtype=float
        )

        for sigma_index, state in enumerate(sigma_points):
            velocity = state["ekf_frame_velocity"]
            omega_hat = self._corrected_angular_velocity(state)
            for contact_index in range(self._nb_ee):
                if not contacts_schedule[contact_index]:
                    continue
                row = slice(3 * contact_index, 3 * contact_index + 3)
                if self._ekf_in_imu_frame:
                    omega_base = self._SE3_imu_to_base.rotation @ omega_hat
                    base_motion = pin.Motion(
                        -ee_velocities[contact_index]
                        - pin.skew(omega_base) @ ee_positions[contact_index],
                        omega_base,
                    )
                    measured_velocity = self._SE3_base_to_imu.act(
                        base_motion
                    ).linear
                else:
                    measured_velocity = (
                        -ee_velocities[contact_index]
                        - pin.skew(omega_hat) @ ee_positions[contact_index]
                    )
                measurements[sigma_index, row] = velocity - measured_velocity
        return measurements

    def _update_step(
        self, contacts_schedule, joint_positions, joint_velocities
    ):
        """Updates the state with stance-foot velocity measurements."""
        sigma_points = self._compute_sigma_points(
            self._mu_pre, self._Sigma_pre
        )
        sigma_measurements = self._measurement_model(
            sigma_points,
            contacts_schedule,
            joint_positions,
            joint_velocities,
        )
        measurement_mean = self._weights_mean @ sigma_measurements

        measurement_dim = 3 * self._nb_ee
        innovation_covariance = np.zeros(
            (measurement_dim, measurement_dim)
        )
        cross_covariance = np.zeros((self._nx, measurement_dim))
        for index, weight in enumerate(self._weights_cov):
            state_error = self._state_box_minus(
                sigma_points[index], self._mu_pre
            )
            measurement_error = (
                sigma_measurements[index] - measurement_mean
            )
            innovation_covariance += weight * np.outer(
                measurement_error, measurement_error
            )
            cross_covariance += weight * np.outer(
                state_error, measurement_error
            )

        innovation_covariance += (
            self._construct_discrete_measurement_noise_covariance()
        )
        innovation_covariance = 0.5 * (
            innovation_covariance + innovation_covariance.T
        )
        kalman_gain = np.linalg.solve(
            innovation_covariance, cross_covariance.T
        ).T
        delta_x = kalman_gain @ (-measurement_mean)

        self._mu_post = self._state_box_plus(self._mu_pre, delta_x)
        covariance = (
            self._Sigma_pre
            - kalman_gain @ innovation_covariance @ kalman_gain.T
        )
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        if eigenvalues[0] < 0.0:
            covariance = (
                eigenvectors
                @ np.diag(np.maximum(eigenvalues, 0.0))
                @ eigenvectors.T
            )
        self._Sigma_post = 0.5 * (covariance + covariance.T)

    # Public methods
    def update_filter(
        self,
        a_tilde,
        omega_tilde,
        contacts_schedule,
        joint_positions,
        joint_velocities,
    ):
        """Updates the filter using IMU, contact, and encoder measurements."""
        self._a_tilde = np.asarray(a_tilde, dtype=float).reshape(3)
        self._omega_tilde = np.asarray(omega_tilde, dtype=float).reshape(3)
        if not self._has_previous_imu:
            # There is no angular-acceleration observation before the first
            # sample. Reusing the current rate avoids an artificial lever-arm
            # acceleration transient.
            self._omega_tilde_prev = self._omega_tilde.copy()
        contacts_schedule = np.asarray(contacts_schedule).reshape(-1)
        if contacts_schedule.size != self._nb_ee:
            raise ValueError(
                "contacts_schedule must contain one value per end effector"
            )

        self._prediction_step()
        self._update_step(
            contacts_schedule, joint_positions, joint_velocities
        )
        self._omega_tilde_prev = self._omega_tilde.copy()
        self._has_previous_imu = True

    def get_filter_output(self):
        """Returns the base states using the same convention as ``BaseEKF``."""
        return super().get_filter_output()

    def get_filter_covariance(self):
        """Returns a copy of the 15-dimensional local error covariance."""
        return self._Sigma_post.copy()
