"""Centroidal Extended Kalman Filter Class
License BSD-3-Clause
Copyright (c) 2022, New York University and Max Planck Gesellschaft.
Author: Shahram Khorshidi
"""

import yaml
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

    def __init__(self, urdf_file, config_file):
        # Load configuration from YAML file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self._rmodel = pin.buildModelFromUrdf(urdf_file, pin.JointModelFreeFlyer())
        self._rdata = self._rmodel.createData()
        robot_config = config.get('robot', {})
        self._nx = 3 * 3
        self._dt = robot_config.get('dt', 0.001)  # Default dt: 0.001
        self._end_effectors_frame_names = robot_config.get('end_effectors_frame_names', [])
        self._endeff_ids = [
              self._rmodel.getFrameId(name)
            for name in self._end_effectors_frame_names
        ]
        self._robot_mass = robot_config.get('mass')

        self._nb_ee = len(self._end_effectors_frame_names)
        self._nv =   self._rmodel.nv
        self._mu_pre = np.zeros(self._nx, dtype=float)
        self._mu_post = np.zeros(self._nx, dtype=float)
        self._sigma_pre = np.zeros((self._nx, self._nx), dtype=float)
        self._sigma_post = np.zeros((self._nx, self._nx), dtype=float)
        self._Qc = np.eye(self._nx, dtype=float)
        self._Rk = np.eye(9, dtype=float)

        self._S = np.zeros((self._nv-6, self._nv))
        self._S[:, 6:] = np.eye(self._nv-6)
        self._p = np.zeros((self._nv, self._nv), dtype=float)
        self._p_prev = np.zeros((self._nv, self._nv), dtype=float)
        self._pdot = np.zeros((self._nv, self._nv), dtype=float)
        self._contact_schedule = []

        # Initialize the filter
        self._init_filter()

    # Private methods
    def _init_filter(self):
        self.set_process_noise_cov(1e-7, 1e-5, 1e-4)
        self.set_measurement_noise_cov(1e-5, 1e-5, 1e-5)
        self.__Hk = np.eye(9)

    def _update_robot(self, q):
        """Updates frames placement, and joint jacobians.

        Args:
            q (ndarray): Robot configuration.
        """
        pin.forwardKinematics(  self._rmodel, self._rdata, q)
        pin.computeJointJacobians(  self._rmodel, self._rdata, q)
        pin.framesForwardKinematics(  self._rmodel, self._rdata, q)

    def _compute_J_c(self, q):
        """Returns Jacobian of (m) feet in contact.

        Args:
            q (ndarray): Robot configuration.

        Returns:
            np.array(3*m,18)
        """
        self._update_robot(q)
        i = 0
        for value in self._contact_schedule:
            if value:
                i += 1
        J_c = np.zeros((3 * i, self._nv))
        j = 0
        for index in range(self._nb_ee):
            if self._contact_schedule[index]:
                frame_id = self._endeff_ids[index]
                J_c[0 + j : 3 + j, :] = pin.getFrameJacobian(
                      self._rmodel,
                    self._rdata,
                    frame_id,
                    pin.LOCAL_WORLD_ALIGNED,
                )[0:3, :]
                j += 3
        return J_c

    def _compute_null_space_projection(self, q):
        """Returns null space projector.

        Args:
            q (ndarray): Robot configuration.

        Returns:
            np.array(18,18)
        """
        J_c = self._compute_J_c(q)
        p = np.eye((self._nv)) - pinv(J_c) @ J_c
        return p

    def _compute_p_dot(self):
        """Returns null space projector time derivative.

        Returns:
            np.array(18,18)
        """
        p_dot = (1.0 / self._dt) * (self._p - self._p_prev)
        self._p_prev = self._p
        return p_dot

    def _compute_constraint_consistent_mass_matrix_inv(self, q, p):
        """
        Returns:
            np.array(18,18)
        """
        mass_matrix = pin.crba(  self._rmodel, self._rdata, q)
        m_c = p @ mass_matrix + np.eye(self._nv) - p
        return inv(m_c)

    def _compute_nonlinear_terms_h(self, q, dq):
        nonlinear_terms_h = pin.nonLinearEffects(
              self._rmodel, self._rdata, q, dq
        )
        return nonlinear_terms_h

    # Centroidal Extended Kalman Filter
    def _integrate_model(self, q, dq, tau):
        """Calculates the 'a priori" estimate of the state vector.

        Args:
            q (ndarray): Robot configuration.
            dq (ndarray): Robot velocity.
            tau (ndarray): Joint torques.
        """
        self._p = self._compute_null_space_projection(q)
        self.p_dot = self._compute_p_dot()
        h_g_dot = self._dynamic_model_h_g_dot(q, dq, tau)
        h_g = self._centroidal_momenta_h_g(q, dq)

        self._mu_pre[0:3] = (
            self._mu_post[0:3] + (1 / self._robot_mass) * h_g[:3] * self._dt
        )
        self._mu_pre[3:9] = self._mu_post[3:9] + h_g_dot * self._dt

    def _dynamic_model_h_g_dot(self, q, dq, tau):
        p = self._compute_null_space_projection(q)
        m_c_inv = self._compute_constraint_consistent_mass_matrix_inv(q, p)
        nonlinear_terms_h = self._compute_nonlinear_terms_h(q, dq)
        A_g_dot = pin.computeCentroidalMapTimeVariation(
              self._rmodel, self._rdata, q, dq
        )
        A_g = self._rdata.Ag
        h_g_dot = (A_g @ m_c_inv) @ (
            self._pdot.dot(dq)
            - p.dot(nonlinear_terms_h)
            + p @ self._S.T @ tau
        ) + (A_g_dot.dot(dq))
        return h_g_dot

    def _centroidal_momenta_h_g(self, q, dq):
        h_g = np.zeros(6, dtype=float)
        pin.computeCentroidalMomentum(  self._rmodel, self._rdata, q, dq)
        h_g[0:3] = self._rdata.hg.linear
        h_g[3:6] = self._rdata.hg.angular
        return h_g

    def _com_position(self, q):
        com = pin.centerOfMass(  self._rmodel, self._rdata, q, False)
        return com

    def _compute_discrete_prediction_jacobain(self, q, dq, tau):
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
              self._rmodel, self._rdata, q, dq
        )
        A_g = np.copy(self._rdata.Ag)
        A_g_inv_com = pinv(A_g[:3, :])
        A_g_inv = pinv(A_g)

        Fc = np.zeros((9, 9), dtype=float)

        # d_com/d_com
        Fc[0:3, 3:6] = (1 / self._robot_mass) * np.eye(3)

        # d_Fc/d_com
        vec = np.array([0, 0, 0])
        for i in range(3):
            vec[i] = 1
            delta_cx = delta * vec
            delta_dq = A_g_inv_com @ delta_cx
            q_plus = pin.integrate(  self._rmodel, q, delta_dq)
            partial_com = self._com_position(q_plus) - self._com_position(q)
            partial_F_com = self._dynamic_model_h_g_dot(
                q_plus, dq, tau
            ) - self._dynamic_model_h_g_dot(q, dq, tau)
            Fc[3:9, i] = partial_F_com * (1 / partial_com[i])
            vec = np.array([0, 0, 0])

        # d_Fc/d_h
        delta_vec = np.array([0, 0, 0, 0, 0, 0])
        for i in range(6):
            delta_vec[i] = 1
            delta_h = delta * delta_vec
            delta_dq = A_g_inv @ delta_h
            dq_plus = dq + delta_dq
            partial_h_g = self._centroidal_momenta_h_g(
                q, dq_plus
            ) - self._centroidal_momenta_h_g(q, dq)
            partial_F = self._dynamic_model_h_g_dot(
                q, dq_plus, tau
            ) - self._dynamic_model_h_g_dot(q, dq, tau)
            # d_F/d_h[i]
            Fc[3:9, i + 3] = partial_F * (1 / partial_h_g[i])
            delta_vec = np.array([0, 0, 0, 0, 0, 0])

        Fk = np.eye(self._nx) + Fc * self._dt
        return Fk

    def _construct_discrete_noise_covariance(self, Qc, Fk):
        return (Fk @ Qc @ Fk.T) * self._dt

    def _prediction_step(self, q, dq, tau):
        """Calculates the 'a priori error covariance matrix' in the prediction step.

        Args:
            q (ndarray): Robot configuration.
            dq (ndarray): Robot velocity.
            tau (ndarray): Joint torques.
        """
        Fk = self._compute_discrete_prediction_jacobain(q, dq, tau)
        Qk = self._construct_discrete_noise_covariance(self._Qc, Fk)
        # Priori error covariance matrix
        self._sigma_pre = (Fk @ self._sigma_post @ Fk.T) + Qk

    def _measurement_model(self, q, dq):
        """Returns the measurement residual.

        Args:
            q (ndarray): Robot configuration.
            dq (ndarray): Robot velocity.

        Returns:
            ndarray: Measurement residual.
        """
        y_predicted = np.zeros(9, dtype=float)
        y_measured = np.zeros(9, dtype=float)
        y_predicted = self._mu_pre

        pin.computeCentroidalMomentum(self._rmodel, self._rdata, q, dq)
        y_measured[0:3] = self._rdata.com[0]
        y_measured[3:6] = self._rdata.hg.linear
        y_measured[6:9] = self._rdata.hg.angular

        measurement_error = y_measured - y_predicted

        return measurement_error

    def _update_step(self, q, dq):
        """Calculates the 'a posteriori' state and error covariance matrix.

        Args:
            q (ndarray): Robot configuration.
            dq (ndarray): Robot velocity.
        """
        measurement_error = self._measurement_model(q, dq)
        # Compute kalman gain
        kalman_gain = (self._sigma_pre @ self.__Hk.T) @ inv(
            (self.__Hk @ self._sigma_pre @ self.__Hk.T) + self._Rk
        )
        delta_x = kalman_gain @ measurement_error
        self._sigma_post = (
            np.eye(self._nx) - (kalman_gain @ self.__Hk)
        ) @ self._sigma_pre
        self._mu_post = self._mu_pre + delta_x

    # Public methods
    def set_process_noise_cov(self, q_c, q_l, q_k):
        q = np.concatenate(
            [np.array(3 * [q_c]), np.array(3 * [q_l]), np.array(3 * [q_k])]
        )
        np.fill_diagonal(self._Qc, q)

    def set_measurement_noise_cov(self, r_c, r_l, r_k):
        r = np.concatenate(
            [np.array(3 * [r_c]), np.array(3 * [r_l]), np.array(3 * [r_k])]
        )
        np.fill_diagonal(self._Rk, r)

    def set_mu_post(self, value):
        self._mu_post = value

    def update_filter(self, q, dq, contact_schedule, tau):
        """Updates the filter.

        Args:
            q (ndarray): Robot configuration.
            dq (ndarray): Robot velocity.
            contacts_schedule (list): Contact schedule of the feet.
            tau (ndarray): Joint torques.
        """
        self._contact_schedule = contact_schedule
        self._integrate_model(q, dq, tau)
        self._prediction_step(q, dq, tau)
        self._update_step(q, dq)

    def get_filter_output(self):
        """Returns the centroidal states, estimated by the EKF.

        Returns:
            tuple of np.arrays
        """
        com_position = self._mu_post[0:3]
        lin_momentum = self._mu_post[3:6]
        ang_momentum = self._mu_post[6:9]
        return com_position, lin_momentum, ang_momentum
    
