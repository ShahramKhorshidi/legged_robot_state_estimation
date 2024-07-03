"""Centroidal Extended Kalman Filter Class
License BSD-3-Clause
Copyright (c) 2022, New York University and Max Planck Gesellschaft.
Author: Shahram Khorshidi
"""

import numpy as np
import pinocchio as pin
from numpy.linalg import inv, pinv


class TorqueCentroidalEKF(object):
    """Centroidal EKF class for estimation of the centroidal states (center of mass position, linear momentum, and angular momentum) of the robot.

    Attributes:
        nx : int
            Dimension of the error state vector.
        dt : float
            Discretization time.
        robot_mass : float
            Robot total mass.
        nb_ee : int
            Number of end_effectors
        nv : int
            Dimension of the generalized velocities.
        centroidal_state : dict
            Centroidal states estimated by EKF.
        Qc : np.array(9,9)
            Continuous process noise covariance.
        Rk : np.array(9,9)
            Discrete measurement noise covariance.
        b : np.array(18,18)
            Selection matrix (separating actuated/unactuated DoFs).
        p : np.array(18,18)
            Null space projector.
    """

    def __init__(self, robot_config, dt=0.001):
        self.__rmodel = robot_config.robot_model
        self.__rdata = self.__rmodel.createData()
        self.__nx = 3 * 3
        self.__dt = dt
        self.__end_effectors_frame_names = robot_config.end_effector_names
        self.__endeff_ids = [
            self.__rmodel.getFrameId(name)
            for name in self.__end_effectors_frame_names
        ]
        self.__robot_mass = robot_config.mass

        self.__nb_ee = len(self.__end_effectors_frame_names)
        self.__nv = self.__rmodel.nv
        self.__centroidal_state = dict.fromkeys(
            [
                "com_position",
                "linear_momentum",
                "angular_momentum",
            ]
        )

        self.__mu_pre = np.zeros(self.__nx, dtype=float)
        self.__mu_post = np.zeros(self.__nx, dtype=float)
        self.__sigma_pre = np.zeros((self.__nx, self.__nx), dtype=float)
        self.__sigma_post = np.zeros((self.__nx, self.__nx), dtype=float)
        self.__Qc = np.eye(self.__nx, dtype=float)
        self.__Rk = np.eye(9, dtype=float)

        self.__b = np.zeros((self.__nv, self.__nv))
        self.__b[6:, 6:] = np.eye((self.__nv - 6))
        self.__p = np.zeros((self.__nv, self.__nv), dtype=float)
        self.__p_prev = np.zeros((self.__nv, self.__nv), dtype=float)
        self.__pdot = np.zeros((self.__nv, self.__nv), dtype=float)
        self.__contact_schedule = []

        # Initialize the filter
        self.__init_filter()

    # Private methods
    def __init_filter(self):
        self.set_process_noise_cov(1e-7, 1e-5, 1e-4)
        self.set_measurement_noise_cov(1e-5, 1e-5, 1e-5)
        self.__Hk = np.eye(9)

    def __update_robot(self, q):
        """Updates frames placement, and joint jacobians.

        Args:
            q (ndarray): Robot configuration.
        """
        pin.forwardKinematics(self.__rmodel, self.__rdata, q)
        pin.computeJointJacobians(self.__rmodel, self.__rdata, q)
        pin.framesForwardKinematics(self.__rmodel, self.__rdata, q)

    def __compute_J_c(self, q):
        """Returns Jacobian of (m) feet in contact.

        Args:
            q (ndarray): Robot configuration.

        Returns:
            np.array(3*m,18)
        """
        self.__update_robot(q)
        i = 0
        for value in self.__contact_schedule:
            if value:
                i += 1
        J_c = np.zeros((3 * i, self.__nv))
        j = 0
        for index in range(self.__nb_ee):
            if self.__contact_schedule[index]:
                frame_id = self.__endeff_ids[index]
                J_c[0 + j : 3 + j, :] = pin.getFrameJacobian(
                    self.__rmodel,
                    self.__rdata,
                    frame_id,
                    pin.LOCAL_WORLD_ALIGNED,
                )[0:3, :]
                j += 3
        return J_c

    def __compute_null_space_projection(self, q):
        """Returns null space projector.

        Args:
            q (ndarray): Robot configuration.

        Returns:
            np.array(18,18)
        """
        J_c = self.__compute_J_c(q)
        p = np.eye((self.__nv)) - pinv(J_c) @ J_c
        return p

    def __compute_p_dot(self):
        """Returns null space projector time derivative.

        Returns:
            np.array(18,18)
        """
        p_dot = (1.0 / self.__dt) * (self.__p - self.__p_prev)
        self.__p_prev = self.__p
        return p_dot

    def __compute_constraint_consistent_mass_matrix_inv(self, q, p):
        """
        Returns:
            np.array(18,18)
        """
        mass_matrix = pin.crba(self.__rmodel, self.__rdata, q)
        m_c = p @ mass_matrix + np.eye(self.__nv) - p
        return inv(m_c)

    def __compute_nonlinear_terms_h(self, q, dq):
        nonlinear_terms_h = pin.nonLinearEffects(
            self.__rmodel, self.__rdata, q, dq
        )
        return nonlinear_terms_h

    # Centroidal Extended Kalman Filter
    def integrate_model(self, q, dq, tau):
        """Calculates the 'a priori" estimate of the state vector.

        Args:
            q (ndarray): Robot configuration.
            dq (ndarray): Robot velocity.
            tau (ndarray): Joint torques.
        """
        self.__p = self.__compute_null_space_projection(q)
        self.p_dot = self.__compute_p_dot()
        h_g_dot = self.dynamic_model_h_g_dot(q, dq, tau)
        h_g = self.centroidal_momenta_h_g(q, dq)

        self.__mu_pre[0:3] = (
            self.__mu_post[0:3] + (1 / self.__robot_mass) * h_g[:3] * self.__dt
        )
        self.__mu_pre[3:9] = self.__mu_post[3:9] + h_g_dot * self.__dt

    def dynamic_model_h_g_dot(self, q, dq, tau):
        p = self.__compute_null_space_projection(q)
        m_c_inv = self.__compute_constraint_consistent_mass_matrix_inv(q, p)
        nonlinear_terms_h = self.__compute_nonlinear_terms_h(q, dq)
        torque_input = np.concatenate((np.zeros(6), tau), axis=0)
        A_g_dot = pin.computeCentroidalMapTimeVariation(
            self.__rmodel, self.__rdata, q, dq
        )
        A_g = self.__rdata.Ag
        h_g_dot = (A_g @ m_c_inv) @ (
            self.__pdot.dot(dq)
            - p.dot(nonlinear_terms_h)
            + p @ self.__b @ torque_input
        ) + (A_g_dot.dot(dq))
        return h_g_dot

    def centroidal_momenta_h_g(self, q, dq):
        h_g = np.zeros(6, dtype=float)
        pin.computeCentroidalMomentum(self.__rmodel, self.__rdata, q, dq)
        h_g[0:3] = self.__rdata.hg.linear
        h_g[3:6] = self.__rdata.hg.angular
        return h_g

    def com_position(self, q):
        com = pin.centerOfMass(self.__rmodel, self.__rdata, q, False)
        return com

    def compute_discrete_prediction_jacobain(self, q, dq, tau):
        """Returns the discrete prediction jacobian.

        Args:
            q (ndarray): Robot configuration.
            dq (ndarray): Robot velocity.
            tau (ndarray): Joint torques.

        Returns:
            np.array(9,9)
        """
        delta = 1e-6
        A_g_dot = pin.computeCentroidalMapTimeVariation(
            self.__rmodel, self.__rdata, q, dq
        )
        A_g = np.copy(self.__rdata.Ag)
        A_g_inv_com = pinv(A_g[:3, :])
        A_g_inv = pinv(A_g)

        Fc = np.zeros((9, 9), dtype=float)

        # d_com/d_com
        Fc[0:3, 3:6] = (1 / self.__robot_mass) * np.eye(3)

        # d_Fc/d_com
        vec = np.array([0, 0, 0])
        for i in range(3):
            vec[i] = 1
            delta_cx = delta * vec
            delta_dq = A_g_inv_com @ delta_cx
            q_plus = pin.integrate(self.__rmodel, q, delta_dq)
            partial_com = self.com_position(q_plus) - self.com_position(q)
            partial_F_com = self.dynamic_model_h_g_dot(
                q_plus, dq, tau
            ) - self.dynamic_model_h_g_dot(q, dq, tau)
            Fc[3:9, i] = partial_F_com * (1 / partial_com[i])
            vec = np.array([0, 0, 0])

        # d_Fc/d_h
        delta_vec = np.array([0, 0, 0, 0, 0, 0])
        for i in range(6):
            delta_vec[i] = 1
            delta_h = delta * delta_vec
            delta_dq = A_g_inv @ delta_h
            dq_plus = dq + delta_dq
            partial_h_g = self.centroidal_momenta_h_g(
                q, dq_plus
            ) - self.centroidal_momenta_h_g(q, dq)
            partial_F = self.dynamic_model_h_g_dot(
                q, dq_plus, tau
            ) - self.dynamic_model_h_g_dot(q, dq, tau)
            # d_F/d_h[i]
            Fc[3:9, i + 3] = partial_F * (1 / partial_h_g[i])
            delta_vec = np.array([0, 0, 0, 0, 0, 0])

        Fk = np.eye(self.__nx) + Fc * self.__dt
        return Fk

    def construct_discrete_noise_covariance(self, Qc, Fk):
        return (Fk @ Qc @ Fk.T) * self.__dt

    def prediction_step(self, q, dq, tau):
        """Calculates the 'a priori error covariance matrix' in the prediction step.

        Args:
            q (ndarray): Robot configuration.
            dq (ndarray): Robot velocity.
            tau (ndarray): Joint torques.
        """
        Fk = self.compute_discrete_prediction_jacobain(q, dq, tau)
        Qk = self.construct_discrete_noise_covariance(self.__Qc, Fk)
        # Priori error covariance matrix
        self.__sigma_pre = (Fk @ self.__sigma_post @ Fk.T) + Qk

    def measurement_model(self, q, dq):
        """Returns the measurement residual.

        Args:
            q (ndarray): Robot configuration.
            dq (ndarray): Robot velocity.

        Returns:
            ndarray: Measurement residual.
        """
        y_predicted = np.zeros(9, dtype=float)
        y_measured = np.zeros(9, dtype=float)
        y_predicted = self.__mu_pre

        pin.computeCentroidalMomentum(self.__rmodel, self.__rdata, q, dq)
        y_measured[0:3] = self.__rdata.com[0]
        y_measured[3:6] = self.__rdata.hg.linear
        y_measured[6:9] = self.__rdata.hg.angular

        measurement_error = y_measured - y_predicted

        return measurement_error

    def update_step(self, q, dq):
        """Calculates the 'a posteriori' state and error covariance matrix.

        Args:
            q (ndarray): Robot configuration.
            dq (ndarray): Robot velocity.
        """
        measurement_error = self.measurement_model(q, dq)
        # Compute kalman gain
        kalman_gain = (self.__sigma_pre @ self.__Hk.T) @ inv(
            (self.__Hk @ self.__sigma_pre @ self.__Hk.T) + self.__Rk
        )
        delta_x = kalman_gain @ measurement_error
        self.__sigma_post = (
            np.eye(self.__nx) - (kalman_gain @ self.__Hk)
        ) @ self.__sigma_pre
        self.__mu_post = self.__mu_pre + delta_x

    # Public methods
    def set_process_noise_cov(self, q_c, q_l, q_k):
        q = np.concatenate(
            [np.array(3 * [q_c]), np.array(3 * [q_l]), np.array(3 * [q_k])]
        )
        np.fill_diagonal(self.__Qc, q)

    def set_measurement_noise_cov(self, r_c, r_l, r_k):
        r = np.concatenate(
            [np.array(3 * [r_c]), np.array(3 * [r_l]), np.array(3 * [r_k])]
        )
        np.fill_diagonal(self.__Rk, r)

    def set_mu_post(self, value):
        self.__mu_post = value

    def update_filter(self, q, dq, contact_schedule, tau):
        """Updates the filter.

        Args:
            q (ndarray): Robot configuration.
            dq (ndarray): Robot velocity.
            contacts_schedule (list): Contact schedule of the feet.
            tau (ndarray): Joint torques.
        """
        self.__contact_schedule = contact_schedule
        self.integrate_model(q, dq, tau)
        self.prediction_step(q, dq, tau)
        self.update_step(q, dq)

    def get_filter_output(self):
        """Returns the centroidal states, estimated by the EKF.

        Returns:
            dict
        """
        self.__centroidal_state["com_position"] = self.__mu_post[0:3]
        self.__centroidal_state["linear_momentum"] = self.__mu_post[3:6]
        self.__centroidal_state["angular_momentum"] = self.__mu_post[6:9]
        return self.__centroidal_state


if __name__ == "__main__":
    robot_config = Solo12Config()
    solo_cent_ekf = TorqueCentroidalEKF(robot_config)
    robot_configuration = np.random.rand(19)
    robot_velocity = np.random.rand(18)
    joint_torques = np.random.rand(12)
    contacts_schedule = [1, 1, 1, 1]
    solo_cent_ekf.update_filter(
        robot_configuration,
        robot_velocity,
        contacts_schedule,
        joint_torques,
    )

    centroidal_state = solo_cent_ekf.get_filter_output()
