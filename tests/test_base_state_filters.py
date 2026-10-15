"""Numerical checks for the manifold UKF and contact-aided InEKF."""

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pinocchio as pin


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from base_estimation.base_est_ukf import BaseUKF
from base_estimation.base_state_inekf import (
    BaseInEKF,
    ContactAidedInEKF,
    so3_gammas,
)


class BaseUKFTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.urdf = REPOSITORY_ROOT / "files" / "go1_description" / "go1.urdf"
        cls.config = (
            REPOSITORY_ROOT
            / "files"
            / "go1_description"
            / "go1_config.yaml"
        )

    def make_filter(self):
        return BaseUKF(str(self.urdf), self.config)

    def test_box_plus_and_minus_are_inverse(self):
        ukf = self.make_filter()
        rng = np.random.default_rng(4)
        delta = rng.normal(size=15) * 1e-2

        perturbed = ukf._state_box_plus(ukf._mu_post, delta)
        recovered = ukf._state_box_minus(perturbed, ukf._mu_post)

        np.testing.assert_allclose(recovered, delta, atol=1e-12, rtol=1e-10)

    def test_sigma_points_reconstruct_mean_and_covariance(self):
        ukf = self.make_filter()
        rng = np.random.default_rng(7)
        matrix = rng.normal(size=(15, 15))
        covariance = matrix @ matrix.T * 1e-5

        sigma_points = ukf._compute_sigma_points(ukf._mu_post, covariance)
        recovered_mean = ukf._compute_state_mean(sigma_points)
        mean_error = ukf._state_box_minus(recovered_mean, ukf._mu_post)

        recovered_covariance = np.zeros((15, 15))
        for weight, state in zip(ukf._weights_cov, sigma_points):
            error = ukf._state_box_minus(state, recovered_mean)
            recovered_covariance += weight * np.outer(error, error)

        np.testing.assert_allclose(mean_error, np.zeros(15), atol=1e-11)
        np.testing.assert_allclose(
            recovered_covariance, covariance, atol=1e-11, rtol=1e-8
        )

    def test_leg_velocity_sigma_transform_matches_measurement_jacobian(self):
        ukf = self.make_filter()
        q = np.loadtxt(REPOSITORY_ROOT / "data" / "robot_q.dat", delimiter="\t")
        dq = np.loadtxt(REPOSITORY_ROOT / "data" / "robot_dq.dat", delimiter="\t")
        imu = np.loadtxt(
            REPOSITORY_ROOT / "data" / "robot_imu.dat", delimiter="\t"
        )
        ukf._omega_tilde = imu[3:, 0]
        covariance = np.diag(np.linspace(1.0, 2.0, 15)) * 1e-7
        sigma_points = ukf._compute_sigma_points(ukf._mu_post, covariance)
        contacts = np.ones(ukf._nb_ee, dtype=bool)
        sigma_measurements = ukf._measurement_model(
            sigma_points, contacts, q[7:, 0], dq[6:, 0]
        )
        measurement_mean = ukf._weights_mean @ sigma_measurements

        cross_covariance = np.zeros((15, 3 * ukf._nb_ee))
        for index, weight in enumerate(ukf._weights_cov):
            state_error = ukf._state_box_minus(
                sigma_points[index], ukf._mu_post
            )
            measurement_error = (
                sigma_measurements[index] - measurement_mean
            )
            cross_covariance += weight * np.outer(
                state_error, measurement_error
            )

        positions, _ = ukf._compute_end_effectors_FK_quantities(
            q[7:, 0], dq[6:, 0]
        )
        measurement_jacobian = np.zeros((3 * ukf._nb_ee, 15))
        for index, position in enumerate(positions):
            rows = slice(3 * index, 3 * index + 3)
            measurement_jacobian[rows, 3:6] = np.eye(3)
            measurement_jacobian[rows, 12:15] = pin.skew(position)

        np.testing.assert_allclose(
            cross_covariance,
            covariance @ measurement_jacobian.T,
            atol=1e-13,
            rtol=1e-8,
        )

    def test_go1_updates_remain_on_so3_with_psd_covariance(self):
        ukf = self.make_filter()
        q = np.loadtxt(REPOSITORY_ROOT / "data" / "robot_q.dat", delimiter="\t")
        dq = np.loadtxt(REPOSITORY_ROOT / "data" / "robot_dq.dat", delimiter="\t")
        contact = np.loadtxt(
            REPOSITORY_ROOT / "data" / "robot_cnt.dat", delimiter="\t"
        )
        imu = np.loadtxt(
            REPOSITORY_ROOT / "data" / "robot_imu.dat", delimiter="\t"
        )

        ukf.set_mu_post("ekf_frame_position", q[:3, 0])
        ukf.set_mu_post("ekf_frame_velocity", dq[:3, 0])
        ukf.set_mu_post("ekf_frame_orientation", pin.Quaternion(q[3:7, 0]))
        ukf.set_meas_noise_cov(np.full(3, 1e-1))

        for i in range(50):
            ukf.update_filter(
                imu[:3, i],
                imu[3:, i],
                contact[:, i],
                q[7:, i],
                dq[6:, i],
            )

        state = ukf.get_filter_output()
        rotation = state["base_orientation"].matrix()
        covariance = ukf.get_filter_covariance()
        self.assertTrue(np.all(np.isfinite(covariance)))
        self.assertLess(np.linalg.norm(rotation.T @ rotation - np.eye(3)), 1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance)[0], -1e-12)

    def test_imu_frame_mode_uses_extrinsic_transform(self):
        ukf = self.make_filter()
        q = np.loadtxt(REPOSITORY_ROOT / "data" / "robot_q.dat", delimiter="\t")
        dq = np.loadtxt(REPOSITORY_ROOT / "data" / "robot_dq.dat", delimiter="\t")
        contact = np.loadtxt(
            REPOSITORY_ROOT / "data" / "robot_cnt.dat", delimiter="\t"
        )
        imu = np.loadtxt(
            REPOSITORY_ROOT / "data" / "robot_imu.dat", delimiter="\t"
        )
        ukf.set_ekf_in_imu_frame(True)
        ukf.update_filter(
            imu[:3, 0],
            imu[3:, 0],
            contact[:, 0],
            q[7:, 0],
            dq[6:, 0],
        )

        state = ukf.get_filter_output()
        self.assertTrue(np.all(np.isfinite(state["base_position"])))
        self.assertTrue(np.all(np.isfinite(state["base_velocity"])))
        rotation = state["base_orientation"].matrix()
        self.assertLess(np.linalg.norm(rotation.T @ rotation - np.eye(3)), 1e-12)


class ContactAidedInEKFTest(unittest.TestCase):
    def make_core(self):
        core = ContactAidedInEKF(
            rotation=pin.exp(np.array([0.2, -0.1, 0.3])),
            velocity=np.array([0.4, -0.2, 0.1]),
            position=np.array([0.3, 0.5, 0.7]),
            gyro_bias=np.array([0.01, -0.02, 0.015]),
            accel_bias=np.array([0.1, -0.05, 0.03]),
        )
        core.correct_contacts(
            {0: np.array([0.2, 0.1, -0.6])}, {0: True}
        )
        return core

    @staticmethod
    def invariant_error(true_state, estimated_state):
        error_matrix = true_state.X @ np.linalg.inv(estimated_state.X)
        rotation_error = pin.log3(error_matrix[:3, :3])
        left_jacobian = so3_gammas(rotation_error)[1]
        error = np.zeros(estimated_state.error_dim)
        error[:3] = rotation_error
        for i in range(error_matrix.shape[1] - 3):
            rows = slice(3 + 3 * i, 6 + 3 * i)
            error[rows] = np.linalg.solve(
                left_jacobian, error_matrix[:3, 3 + i]
            )
        group_dim = estimated_state.group_dim
        error[group_dim : group_dim + 3] = (
            true_state.bg - estimated_state.bg
        )
        error[group_dim + 3 :] = true_state.ba - estimated_state.ba
        return error

    def test_process_matrix_matches_invariant_error_dynamics(self):
        estimated = self.make_core()
        true_state = copy.deepcopy(estimated)
        rng = np.random.default_rng(11)
        initial_error = rng.normal(size=estimated.error_dim) * 1e-7
        true_state._inject(initial_error)

        dt = 0.001
        process_matrix, process_noise = estimated._continuous_matrices(dt)
        transition, _ = estimated._discretize(
            process_matrix, process_noise, dt
        )
        acceleration = np.array([0.3, -0.2, 9.7])
        angular_velocity = np.array([0.2, -0.15, 0.1])
        estimated.propagate(acceleration, angular_velocity, dt)
        true_state.propagate(acceleration, angular_velocity, dt)

        actual_error = self.invariant_error(true_state, estimated)
        predicted_error = transition @ initial_error
        np.testing.assert_allclose(
            actual_error, predicted_error, atol=1e-11, rtol=1e-5
        )

    def test_contact_jacobian_matches_invariant_residual(self):
        estimated = self.make_core()
        true_state = copy.deepcopy(estimated)
        rng = np.random.default_rng(17)
        error = rng.normal(size=estimated.error_dim) * 1e-7
        true_state._inject(error)

        measured_position = true_state.R.T @ (
            true_state._feet[:, 0] - true_state.p
        )
        residual = (
            estimated.R @ measured_position
            + estimated.p
            - estimated._feet[:, 0]
        )
        measurement_matrix = np.zeros((3, estimated.error_dim))
        measurement_matrix[:, 6:9] = -np.eye(3)
        measurement_matrix[:, 9:12] = np.eye(3)

        np.testing.assert_allclose(
            residual, measurement_matrix @ error, atol=1e-12, rtol=1e-5
        )


class BaseInEKFTest(unittest.TestCase):
    def test_go1_contact_updates_and_switching_contacts(self):
        urdf = REPOSITORY_ROOT / "files" / "go1_description" / "go1.urdf"
        config = (
            REPOSITORY_ROOT
            / "files"
            / "go1_description"
            / "go1_config.yaml"
        )
        q = np.loadtxt(REPOSITORY_ROOT / "data" / "robot_q.dat", delimiter="\t")
        dq = np.loadtxt(REPOSITORY_ROOT / "data" / "robot_dq.dat", delimiter="\t")
        contact = np.loadtxt(
            REPOSITORY_ROOT / "data" / "robot_cnt.dat", delimiter="\t"
        )
        imu = np.loadtxt(
            REPOSITORY_ROOT / "data" / "robot_imu.dat", delimiter="\t"
        )

        inekf = BaseInEKF(str(urdf), config)
        inekf.set_initial_state(
            q[:3, 0],
            dq[:3, 0],
            q[3:7, 0],
            omega_tilde=imu[3:, 0],
        )
        inekf.initialize_contacts(
            contact[:, 0], q[7:, 0], omega_tilde=imu[3:, 0]
        )
        for i in range(1, 250):
            inekf.update_filter(
                imu[:3, i],
                imu[3:, i],
                contact[:, i],
                q[7:, i],
                dq[6:, i],
            )

        state = inekf.get_filter_output()
        covariance = inekf.get_covariance()
        rotation = state["base_orientation"].matrix()
        self.assertEqual(
            set(state), {"base_position", "base_velocity", "base_orientation"}
        )
        self.assertTrue(np.all(np.isfinite(covariance)))
        self.assertLess(np.linalg.norm(rotation.T @ rotation - np.eye(3)), 1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance)[0], -1e-12)
        expected_contacts = set(np.flatnonzero(contact[:, 249]))
        self.assertEqual(set(inekf.core.contact_ids), expected_contacts)


if __name__ == "__main__":
    unittest.main()
