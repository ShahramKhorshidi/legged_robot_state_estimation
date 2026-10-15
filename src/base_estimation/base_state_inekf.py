"""Invariant Extended Kalman Filter Class
License BSD-3-Clause
Copyright (c) 2025, University of Bonn.
Author: Shahram Khorshidi
"""
# BSD 3-Clause License
# Copyright (c) 2018-2019, Ross Hartley
# All rights reserved.
#
# Python adaptation of the contact-inertial equations and algorithms in
# RossHartley/invariant-ekf, with a new Pinocchio robot adapter.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.

"""Contact-aided right-invariant EKF, with IMU biases and switching contacts.

References
----------
Hartley et al., "Contact-Aided Invariant Extended Kalman Filtering for Legged
Robot State Estimation", RSS 2018, https://arxiv.org/abs/1805.10410.
https://github.com/RossHartley/invariant-ekf
Source inspected at commit ef16e8a1df72f9272111a488880e3fe9d161f59f.

Two classes in this single file
------------------------------
ContactAidedInEKF: robot-independent NumPy/SciPy core; accepts IMU measurements
and contact point positions expressed in the IMU frame.
BaseInEKF: Pinocchio/PyYAML adapter for the supplied BaseEKF's robot interface.

Quick integration
-----------------
    from src.base_estimation.base_state_inekf import BaseInEKF
    ekf = BaseInEKF(urdf_path, yaml_path, dt=0.001,
                   sigma_encoder=1e-3, sigma_kinematics=0.003,
                   sigma_contact=0.01)
    # Mean is at sample-zero time. Register contacts without advancing time:
    ekf.initialize_contacts(contact0, q_joint0, omega_tilde=gyro0)
    # Samples below are at successive times; dt is the elapsed time.
    ekf.update_filter(accel, gyro, contact_mask, q_joint, dq_joint)
    estimate = ekf.get_filter_output()
    # estimate['base_position']: world position, shape (3,)
    # estimate['base_velocity']: BASE-FRAME linear velocity, shape (3,)
    # estimate['base_orientation']: Pinocchio Quaternion (coeffs: xyzw)

For a moving initial state use set_initial_state(..., omega_tilde=gyro0) BEFORE
initialize_contacts. That gyro is needed to convert base velocity to IMU-point
velocity when the IMU has a lever arm. Joint velocities are accepted only for
call compatibility; the contact update uses positions and analytic Jacobians.
No contact-force sensor, numerical joint acceleration, or gyro derivative is used.

State and mathematics
---------------------
The core always tracks the IMU origin. R=R_WI; v and p are WORLD velocity and
position of the IMU. d_i are estimated WORLD contact positions. Biases are in
IMU axes. X = [[R, v, p, d_1, ..., d_N], [0, I]] is in SE_{N+2}(3).
Error/covariance order is [rotation, velocity, position, d_1,...,d_N, bg, ba],
dimension 15+3N. It is NOT the old quaternion EKF's 15-state covariance order.
ContactAidedInEKF.P is the full invariant covariance, including all correlations.

Use the correction error X_true=Exp(e_X) X_hat, b_true=b_hat+e_b. To first order
it is the negative of the paper's (log(X_hat X_true^-1), b_hat-b_true), with the
SAME covariance and A/H matrices. The nonzero continuous A blocks are
    A[v,R]=skew(g), A[p,v]=I, A[R,bg]=-R, A[v,ba]=-R,
    A[x,bg]=-skew(x) R, for x=v,p,d_i.
Noise is mapped by diag(Adjoint(X), I_6). Bias augmentation is the paper's
"imperfect InEKF": do not claim the bias-free autonomous-error theorem for it.

For a measured IMU-to-foot position z_i,
    residual_i = R z_i + p - d_i,
    H_i = [0,0,-I,...,+I_at_d_i,...,0,0],
    N_ij = R Cov(z_i,z_j) R.T.
The update is X_plus=Exp(K residual) X and b_plus=b+delta_b. It rotates/translates
ALL group columns, not just R. Covariance uses Hartley's first-order invariant
Riccati/Joseph correction, without an additional quaternion-only ESKF reset.

New contacts are initialized with d_i=p+R z_i and their correct cross-covariance;
that same measurement is NOT immediately reused as an independent correction.
Departing contacts are marginalized by selecting the remaining covariance block.
Full cross-foot measurement covariance is supported, including the correlation
between a newly augmented contact and an existing contact corrected at that time.

Discretization and scope
------------------------
Defaults: exact constant-input IMU mean integration (SO(3) Gamma matrices), and
exact discretization of the FROZEN linear A,Q model using A^4=0 and a four-point
Gauss integral. This is not exact integration of time-varying A/Q. Set BOTH
mean_integration='reference' and covariance_discretization='reference' for the
C++ repository's mean integration, Phi=I+A dt, Qd=Phi Q Phi.T dt.
All propagation/measurement Jacobians are analytic; no model finite differences.

Noise settings
--------------
* YAML imu_accel_thermal_noise / imu_gyro_thermal_noise: continuous noise-density
  amplitudes by default; their squares are covariance intensities. If they are
  per-sample stds instead, set imu_noise_is_density=False; intensity=variance*dt.
  This latter conversion assumes independent uniformly sampled IMU measurements.
* YAML imu_accel_bias_noise / imu_gyro_bias_noise: bias random-walk amplitudes
  per sqrt(second). A single dt integration is applied, not an extra dt factor.
* sigma_contact: isotropic contact position random-walk amplitude, m/sqrt(s).
  It models small stance-foot drift, not arbitrary sliding or moving terrain.
* sigma_encoder: per-sample stds in the joint TANGENT coordinate units.
  encoder_covariance can instead supply a full matrix, including joint coupling.
* sigma_kinematics: per-sample isotropic position model std in metres. Foot
  covariance = J_encoder Cov(encoder) J_encoder.T + sigma_kinematics**2 I.
  This is a position measurement; do not reuse old FOOT-VELOCITY noise tuning.
* All numeric defaults are starting values, not calibration for your robot.

Point-contact modes are supplied externally. False contacts, large slip, IMU
saturation and bad extrinsics remain model errors; optional per-foot NIS gating
can reject inconsistent corrections. Absolute position and global yaw are not
measured. No inertial-parameter estimation is included in this base-state module.
The core requires NumPy/SciPy; BaseInEKF additionally requires PyYAML/Pinocchio.
Python >=3.10. The old EKF's private mutators are not ported; use set_initial_state.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import time

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.stats import chi2


def _vector(value, n, name):
    out = np.asarray(value, dtype=float).reshape(-1)
    if out.size != n or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain {n} finite values")
    return out.copy()


def _std(value, n, name, *, positive=False):
    value = np.asarray(value, dtype=float)
    out = np.full(n, float(value)) if value.ndim == 0 else _vector(value, n, name)
    if not np.all(np.isfinite(out)) or np.any(out < 0) or (positive and np.any(out <= 0)):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return out


def _covariance(value, n, name):
    out = np.asarray(value, dtype=float)
    if out.shape != (n, n) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite {n} by {n} matrix")
    if not np.allclose(out, out.T, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    out = (out + out.T) * 0.5
    if n and np.linalg.eigvalsh(out)[0] < -1e-12 * max(1., np.linalg.norm(out, 2)):
        raise ValueError(f"{name} must be positive semidefinite")
    return out.copy()


def _rotation(value, name):
    out = np.asarray(value, dtype=float)
    if (out.shape != (3, 3) or not np.all(np.isfinite(out)) or
            not np.allclose(out.T @ out, np.eye(3), atol=1e-8) or
            not np.isclose(np.linalg.det(out), 1., atol=1e-8)):
        raise ValueError(f"{name} must be a proper rotation matrix")
    return out.copy()


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def so3_gammas(phi):
    """Return Gamma_0=Exp, Gamma_1=left Jacobian, Gamma_2 (stable near zero)."""
    phi = np.asarray(phi, dtype=float)
    t2 = float(phi @ phi)
    W = skew(phi)
    W2 = W @ W
    if t2 < 1e-4:
        s = 1. - t2/6. + t2*t2/120. - t2**3/5040.
        c = .5 - t2/24. + t2*t2/720. - t2**3/40320.
        b = 1./6. - t2/120. + t2*t2/5040. - t2**3/362880.
        d = 1./24. - t2/720. + t2*t2/40320. - t2**3/3628800.
    else:
        t = np.sqrt(t2)
        s = np.sin(t)/t
        c = (1.-np.cos(t))/t2
        b = (t-np.sin(t))/(t*t2)
        d = (t2+2.*np.cos(t)-2.)/(2.*t2*t2)
    I = np.eye(3)
    return I+s*W+c*W2, I+c*W+b*W2, .5*I+b*W+d*W2


def exp_se_k3(error):
    """Exponential on SE_K(3); order [rotation, translation_column_1, ...]."""
    error = np.asarray(error, dtype=float).reshape(-1)
    if error.size < 3 or error.size % 3 or not np.all(np.isfinite(error)):
        raise ValueError("Group error must be finite with length 3+3*K")
    count = error.size//3 - 1
    R, J, _ = so3_gammas(error[:3])
    X = np.eye(3+count)
    X[:3, :3] = R
    X[:3, 3:] = J @ error[3:].reshape(count, 3).T
    return X


def adjoint_se_k3(X):
    count = X.shape[0]-3
    out = np.zeros((3+3*count, 3+3*count))
    R = X[:3, :3]
    out[:3, :3] = R
    for i in range(count):
        rows = slice(3+3*i, 6+3*i)
        out[rows, :3] = skew(X[:3, 3+i]) @ R
        out[rows, rows] = R
    return out


def _rows(ids, selected):
    """Indices of selected three-vector measurements in an ordered ID list."""
    return np.array([3*ids.index(i)+j for i in selected for j in range(3)], dtype=int)


class ContactAidedInEKF:
    """Robot-independent bias-augmented right-invariant contact EKF.

    All positions and velocities in the state are WORLD expressed. Measurement
    positions are IMU expressed. IDs in correct_contacts() are integer foot IDs.
    Calls are chronological; each IMU/kinematic sample is consumed once.
    """

    def __init__(self, *, rotation=None, velocity=None, position=None,
                 gyro_bias=None, accel_bias=None, covariance=None,
                 gravity=(0., 0., -9.81), gyro_noise_std=.01, accel_noise_std=.1,
                 gyro_bias_noise_std=1e-5, accel_bias_noise_std=1e-4,
                 contact_noise_std=.01, kinematics_noise_std=.003,
                 imu_noise_is_density=True, mean_integration="zoh",
                 covariance_discretization="polynomial", contact_gate_probability=None):
        self.g = _vector(gravity, 3, "gravity")
        self.Qg = np.diag(_std(gyro_noise_std, 3, "gyro_noise_std")**2)
        self.Qa = np.diag(_std(accel_noise_std, 3, "accel_noise_std")**2)
        self.Qbg = np.diag(_std(gyro_bias_noise_std, 3, "gyro_bias_noise_std")**2)
        self.Qba = np.diag(_std(accel_bias_noise_std, 3, "accel_bias_noise_std")**2)
        self.Qcontact = float(_std(contact_noise_std, 1, "contact_noise_std")[0])**2 * np.eye(3)
        self.sigma_kinematics = float(_std(kinematics_noise_std, 1, "kinematics_noise_std", positive=True)[0])
        self.imu_noise_is_density = bool(imu_noise_is_density)
        if mean_integration not in ("zoh", "reference"):
            raise ValueError("mean_integration must be 'zoh' or 'reference'")
        if covariance_discretization not in ("polynomial", "reference"):
            raise ValueError("covariance_discretization must be 'polynomial' or 'reference'")
        self.mean_integration = mean_integration
        self.covariance_discretization = covariance_discretization
        self.gate_probability = contact_gate_probability
        if contact_gate_probability is not None and not 0 < contact_gate_probability < 1:
            raise ValueError("contact_gate_probability must be None or between zero and one")
        self._gate_threshold = None if contact_gate_probability is None else chi2.ppf(contact_gate_probability, 3)
        # Four-point Gauss-Legendre integrates degree-six covariance polynomials.
        nodes, weights = np.polynomial.legendre.leggauss(4)
        self._quadrature = tuple(zip((nodes+1.)*.5, weights*.5))
        self.reset(rotation=np.eye(3) if rotation is None else rotation,
                   velocity=np.zeros(3) if velocity is None else velocity,
                   position=np.zeros(3) if position is None else position,
                   gyro_bias=np.zeros(3) if gyro_bias is None else gyro_bias,
                   accel_bias=np.zeros(3) if accel_bias is None else accel_bias,
                   covariance=covariance)

    @property
    def contact_ids(self):
        return tuple(self._ids)

    @property
    def group_dim(self):
        return 9+3*len(self._ids)

    @property
    def error_dim(self):
        return self.group_dim+6

    @property
    def X(self):
        out = np.eye(5+len(self._ids))
        out[:3, :3] = self.R
        out[:3, 3] = self.v
        out[:3, 4] = self.p
        out[:3, 5:] = self._feet
        return out

    def reset(self, *, rotation, velocity, position, gyro_bias=None, accel_bias=None, covariance=None):
        """Set a new IMU state and discard all contacts. Covariance is invariant 15x15."""
        R = _rotation(rotation, "rotation")
        v, p = _vector(velocity, 3, "velocity"), _vector(position, 3, "position")
        bg = _vector(np.zeros(3) if gyro_bias is None else gyro_bias, 3, "gyro_bias")
        ba = _vector(np.zeros(3) if accel_bias is None else accel_bias, 3, "accel_bias")
        if covariance is None:
            # An uncertain initial estimate, not the all-zero covariance of the old EKF.
            std = np.r_[np.full(3, .1), np.full(3, .1), np.full(3, .1),
                        np.full(3, .01), np.full(3, .1)]
            P = np.diag(std**2)
        else:
            P = _covariance(covariance, 15, "initial covariance")
        self.R, self.v, self.p, self.bg, self.ba = R, v, p, bg, ba
        self._ids, self._feet, self.P = [], np.empty((3, 0)), P
        self.last_diagnostics = {"used_contacts": [], "new_contacts": [], "removed_contacts": [],
                                 "rejected_contacts": [], "normalized_innovation_squared": np.nan}

    def _continuous_matrices(self, dt):
        n, b = self.error_dim, self.group_dim
        A = np.zeros((n, n))
        A[3:6, :3] = skew(self.g)
        A[6:9, 3:6] = np.eye(3)
        A[:3, b:b+3] = -self.R
        A[3:6, b+3:b+6] = -self.R
        X = self.X
        for column in range(3, X.shape[1]):
            j = 3*column-6
            A[j:j+3, b:b+3] = -skew(X[:3, column]) @ self.R
        noise = np.zeros((n, n))
        scale = 1. if self.imu_noise_is_density else dt
        noise[:3, :3] = scale*self.Qg
        noise[3:6, 3:6] = scale*self.Qa
        for i in range(len(self._ids)):
            j = 9+3*i
            noise[j:j+3, j:j+3] = self.Qcontact
        noise[b:b+3, b:b+3] = self.Qbg
        noise[b+3:b+6, b+3:b+6] = self.Qba
        G = np.eye(n)
        G[:b, :b] = adjoint_se_k3(X)
        return A, G @ noise @ G.T

    def _discretize(self, A, Q, dt):
        I = np.eye(A.shape[0])
        B1 = A*dt
        if self.covariance_discretization == "reference":
            F = I+B1
            return F, dt*(F@Q@F.T)
        # A is nilpotent of index at most 4, even with bias/contact augmentation.
        B2 = .5*(B1@B1)
        B3 = (B2@B1)/3.
        F = I+B1+B2+B3
        Qd = np.zeros_like(Q)
        for u, weight in self._quadrature:
            Fu = I+u*B1+u*u*B2+u**3*B3
            Qd += dt*weight*(Fu@Q@Fu.T)
        return F, .5*(Qd+Qd.T)

    def propagate(self, a_tilde, omega_tilde, dt):
        """Propagate once across dt using raw specific force and angular rate in IMU axes.

        Accelerometer input includes gravity's specific-force contribution: at
        rest with an upright IMU, a_tilde is approximately [0,0,+9.81], not zero.
        Inputs are held constant over this interval. Set timestamps accordingly.
        """
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        a = _vector(a_tilde, 3, "a_tilde")-self.ba
        w = _vector(omega_tilde, 3, "omega_tilde")-self.bg
        A, Q = self._continuous_matrices(dt)
        F, Qd = self._discretize(A, Q, dt)
        Rstep, Gamma1, Gamma2 = so3_gammas(w*dt)
        if self.mean_integration == "reference":
            acceleration = self.R@a + self.g
            pnew = self.p+dt*self.v+.5*dt*dt*acceleration
            vnew = self.v+dt*acceleration
        else:
            pnew = self.p+dt*self.v+dt*dt*(self.R@Gamma2@a+.5*self.g)
            vnew = self.v+dt*(self.R@Gamma1@a+self.g)
        self.R, self.v, self.p = self.R@Rstep, vnew, pnew
        self.P = F@self.P@F.T+Qd
        self.P = .5*(self.P+self.P.T)

    def _remove_contacts(self, keep_ids):
        removed = [i for i in self._ids if i not in keep_ids]
        if not removed:
            return removed
        columns = [k for k, i in enumerate(self._ids) if i in keep_ids]
        keep = np.array([*range(9), *[9+3*k+j for k in columns for j in range(3)],
                         *range(self.group_dim, self.error_dim)], dtype=int)
        self.P = self.P[np.ix_(keep, keep)]
        self._feet = self._feet[:, columns].copy()
        self._ids = [self._ids[k] for k in columns]
        return removed

    def _augment_contacts(self, new_ids, positions, covariance):
        """Add all new contacts jointly, preserving cross-contact sensor covariance."""
        if not new_ids:
            return
        b, n, count = self.group_dim, self.error_dim, len(new_ids)
        F = np.zeros((n+3*count, n))
        F[:b, :b] = np.eye(b)
        F[b+3*count:, b:] = np.eye(6)
        G = np.zeros((n+3*count, 3*count))
        points = []
        for k, foot in enumerate(new_ids):
            rows = slice(b+3*k, b+3*k+3)
            F[rows, 6:9] = np.eye(3)
            # With our true-minus-estimate correction error: e_d=e_p-R*n_z.
            G[rows, 3*k:3*k+3] = -self.R
            points.append(self.p+self.R@positions[foot])
        self.P = F@self.P@F.T+G@covariance@G.T
        self._feet = np.column_stack([self._feet, *points])
        self._ids.extend(new_ids)

    def _inject(self, delta):
        b = self.group_dim
        increment = exp_se_k3(delta[:b])
        updated = increment @ self.X
        self.R = updated[:3, :3].copy()
        self.v, self.p = updated[:3, 3].copy(), updated[:3, 4].copy()
        self._feet = updated[:3, 5:].copy()
        self.bg += delta[b:b+3]
        self.ba += delta[b+3:b+6]

    def correct_contacts(self, positions, contacts, covariance=None):
        """Fuse one contact/FK sample and handle stance transitions.

        positions: {integer_foot_id: measured_IMU_to_foot_position}, in IMU axes.
        contacts: {integer_foot_id: bool}; a COMPLETE current contact schedule.
        Absent/false IDs are removed; all true IDs require a position measurement.
        covariance: full 3N x 3N PER-SAMPLE position covariance in IMU axes,
        ordered by SORTED active IDs; defaults to independent sigma_kinematics.
        New-contact measurements augment the state; only existing contacts are
        used for the correction at this timestamp. Do not submit a sample twice.
        """
        if not isinstance(positions, Mapping) or not isinstance(contacts, Mapping):
            raise TypeError("positions and contacts must be mappings keyed by integer foot ID")
        for key, value in contacts.items():
            if not isinstance(key, (int, np.integer)) or value not in (False, True, 0, 1):
                raise ValueError("Contact IDs must be integers and indicators must be Boolean/0/1")
        active = sorted(int(i) for i, on in contacts.items() if on)
        if any(i not in positions for i in active):
            raise ValueError("Every active contact needs an IMU-frame position measurement")
        z = {i: _vector(positions[i], 3, f"foot {i} position") for i in active}
        count = len(active)
        C = (np.eye(3*count)*self.sigma_kinematics**2 if covariance is None else
             _covariance(covariance, 3*count, "kinematic covariance"))
        removed = self._remove_contacts(active)
        old = [i for i in active if i in self._ids]
        new = [i for i in active if i not in self._ids]
        old_rows, new_rows = _rows(active, old), _rows(active, new)
        old_group_dim = self.group_dim
        self._augment_contacts(new, z, C[np.ix_(new_rows, new_rows)])
        diag = {"used_contacts": [], "new_contacts": new, "removed_contacts": removed,
                "rejected_contacts": [], "normalized_innovation_squared": np.nan,
                "per_contact_nis": {}, "measurement_rows": 0}
        if not old:
            self.last_diagnostics = diag
            return
        m = 3*len(old)
        H = np.zeros((m, self.error_dim))
        residual = np.empty(m)
        W = np.kron(np.eye(len(old)), self.R)
        N = W@C[np.ix_(old_rows, old_rows)]@W.T
        # New-contact errors can be correlated with this sample's old-contact
        # measurement noise. Keep U=Cov(e, noise) in the Kalman/Joseph update.
        U = np.zeros((self.error_dim, m))
        if new:
            Wnew = np.kron(np.eye(len(new)), self.R)
            U[old_group_dim:old_group_dim+3*len(new)] = -Wnew@C[np.ix_(new_rows, old_rows)]@W.T
        accepted_rows = []
        for k, foot in enumerate(old):
            rows = slice(3*k, 3*k+3)
            col = self._ids.index(foot)
            H[rows, 6:9] = -np.eye(3)
            H[rows, 9+3*col:12+3*col] = np.eye(3)
            residual[rows] = self.R@z[foot]+self.p-self._feet[:, col]
            if self._gate_threshold is not None:
                h, u = H[rows], U[:, rows]
                S = h@self.P@h.T+N[rows, rows]+h@u+u.T@h.T
                factor = cho_factor(.5*(S+S.T), lower=True)
                nis = float(residual[rows]@cho_solve(factor, residual[rows]))
                diag["per_contact_nis"][foot] = nis
                if nis > self._gate_threshold:
                    diag["rejected_contacts"].append(foot)
                    continue
            accepted_rows.extend(range(3*k, 3*k+3))
            diag["used_contacts"].append(foot)
        if accepted_rows:
            rows = np.asarray(accepted_rows, dtype=int)
            H, residual = H[rows], residual[rows]
            N, U = N[np.ix_(rows, rows)], U[:, rows]
            S = H@self.P@H.T+N+H@U+U.T@H.T
            factor = cho_factor(.5*(S+S.T), lower=True)
            K = cho_solve(factor, (self.P@H.T+U).T).T
            delta = K@residual
            I_KH = np.eye(self.error_dim)-K@H
            P = (I_KH@self.P@I_KH.T + K@N@K.T
                 - I_KH@U@K.T - K@U.T@I_KH.T)
            self._inject(delta)
            self.P = .5*(P+P.T)
            diag["normalized_innovation_squared"] = float(residual@cho_solve(factor, residual))
            diag["innovation_norm"] = float(np.linalg.norm(residual))
            diag["measurement_rows"] = residual.size
        self.last_diagnostics = diag

    def get_state(self):
        """Return independent copies; covariance uses the documented invariant order."""
        return {"rotation": self.R.copy(), "velocity": self.v.copy(), "position": self.p.copy(),
                "gyro_bias": self.bg.copy(), "accel_bias": self.ba.copy(),
                "contact_positions": {i: self._feet[:, k].copy() for k, i in enumerate(self._ids)},
                "contact_ids": self.contact_ids, "covariance": self.P.copy()}

    def get_diagnostics(self):
        out = {k: (v.copy() if isinstance(v, (dict, list)) else v)
               for k, v in self.last_diagnostics.items()}
        out.update(error_dimension=self.error_dim, active_contacts=list(self._ids),
                   minimum_covariance_eigenvalue=float(np.linalg.eigvalsh(self.P)[0]),
                   rotation_orthogonality_error=float(np.linalg.norm(self.R.T@self.R-np.eye(3))))
        return out


class BaseInEKF:
    """Pinocchio adapter for the uploaded BaseEKF's update/output interface.

    Internal state is always at the IMU origin, unlike BaseEKF's selectable frame.
    ``urdf_file`` can be a Pinocchio Model and ``robot_config`` can be a dict.
    The configured base frame must be fixed to free-flyer joint 1; a fixed offset
    between them is supported. See set_initial_state() for initialization.
    """

    def __init__(self, urdf_file, robot_config, dt=None, *, sigma_encoder=1e-3,
                 encoder_covariance=None, sigma_kinematics=.003, sigma_contact=.01,
                 initial_covariance=None, imu_noise_is_density=True,
                 mean_integration="zoh", covariance_discretization="polynomial",
                 contact_gate_probability=None):
        try:
            import pinocchio as pin
            import yaml
        except ImportError as exc:
            raise ImportError("BaseInEKF needs Pinocchio (package 'pin') and PyYAML") from exc
        self._pin = pin
        if isinstance(robot_config, Mapping):
            config = robot_config
        else:
            with Path(robot_config).open(encoding="utf-8") as file:
                config = yaml.safe_load(file)
        config = config.get("robot", config)
        self._dt = float(config.get("dt", .001) if dt is None else dt)
        if not np.isfinite(self._dt) or self._dt <= 0:
            raise ValueError("dt must be finite and positive")
        self._rmodel = (urdf_file if isinstance(urdf_file, pin.Model) else
                        pin.buildModelFromUrdf(str(urdf_file), pin.JointModelFreeFlyer()))
        self._rdata = self._rmodel.createData()
        if self._rmodel.njoints < 2 or self._rmodel.joints[1].shortname() != "JointModelFreeFlyer":
            raise ValueError("The robot must have a free-flyer as joint 1")
        self._nqj, self._nvj = self._rmodel.nq-7, self._rmodel.nv-6
        self._base_frame_name = config.get("base_link_name", (config.get("link_names") or [None])[0])
        if self._base_frame_name is None or not self._rmodel.existFrame(self._base_frame_name):
            raise ValueError("Configure a valid robot.base_link_name")
        self._base_id = self._rmodel.getFrameId(self._base_frame_name)
        if self._rmodel.frames[self._base_id].parentJoint != 1:
            raise ValueError("The base frame must be fixed to the free-flyer joint")
        self._end_effectors_frame_names = list(config.get("end_effectors_frame_names", []))
        self._nb_ee = len(self._end_effectors_frame_names)
        if not self._nb_ee:
            raise ValueError("Configure at least one end_effectors_frame_names entry")
        if len(set(self._end_effectors_frame_names)) != self._nb_ee:
            raise ValueError("Contact frame names must be unique")
        if any(not self._rmodel.existFrame(i) for i in self._end_effectors_frame_names):
            raise ValueError("Unknown end-effector frame")
        self._foot_ids = [self._rmodel.getFrameId(i) for i in self._end_effectors_frame_names]
        self._R_BI = _rotation(config.get("rot_base_to_imu", np.eye(3)), "rot_base_to_imu").T
        self._r_BI = _vector(config.get("r_base_to_imu", np.zeros(3)), 3, "r_base_to_imu")
        if encoder_covariance is None:
            self._encoder_cov = np.diag(_std(sigma_encoder, self._nvj, "sigma_encoder")**2)
        else:
            self._encoder_cov = _covariance(encoder_covariance, self._nvj, "encoder_covariance")
        self._sigma_fk = float(_std(sigma_kinematics, 1, "sigma_kinematics", positive=True)[0])
        for key in ("imu_gyro_thermal_noise", "imu_accel_thermal_noise",
                    "imu_gyro_bias_noise", "imu_accel_bias_noise"):
            if key not in config:
                raise ValueError(f"Missing robot configuration noise entry {key!r}")
        q0 = _vector(config.get("initial_configuration", pin.neutral(self._rmodel)),
                     self._rmodel.nq, "initial_configuration")
        if np.linalg.norm(q0[3:7]) < 1e-12:
            raise ValueError("Initial free-flyer quaternion must be nonzero")
        q0 = pin.normalize(self._rmodel, q0)
        pin.framesForwardKinematics(self._rmodel, self._rdata, q0)
        base = self._rdata.oMf[self._base_id]
        self.core = ContactAidedInEKF(
            rotation=base.rotation@self._R_BI,
            position=base.translation+base.rotation@self._r_BI,
            covariance=initial_covariance, gravity=self._rmodel.gravity.linear,
            gyro_noise_std=config["imu_gyro_thermal_noise"],
            accel_noise_std=config["imu_accel_thermal_noise"],
            gyro_bias_noise_std=config["imu_gyro_bias_noise"],
            accel_bias_noise_std=config["imu_accel_bias_noise"],
            contact_noise_std=sigma_contact, kinematics_noise_std=sigma_kinematics,
            imu_noise_is_density=imu_noise_is_density,
            mean_integration=mean_integration, covariance_discretization=covariance_discretization,
            contact_gate_probability=contact_gate_probability)
        self._initial_covariance = self.core.P.copy()
        self._last_gyro = np.zeros(3)
        self._last_update_seconds = np.nan

    def set_initial_state(self, base_position, base_velocity, base_orientation, *,
                          gyro_bias=None, accel_bias=None, omega_tilde=None, covariance=None):
        """Reset from a BASE state; clear contacts, then call initialize_contacts().

        base_position: world position. base_velocity: base-frame linear velocity.
        base_orientation: Pinocchio Quaternion, xyzw vector, or R_WB matrix.
        gyro_bias/accel_bias: IMU axes. omega_tilde: raw IMU gyro at this time;
        omitted means zero true angular velocity is assumed for initialization.
        covariance: invariant IMU-error order [R_WI, v_WI, p_WI, bg_I, ba_I],
        NOT the old EKF's covariance. Default reuses the constructor's prior.
        """
        pB = _vector(base_position, 3, "base_position")
        vB = _vector(base_velocity, 3, "base_velocity")
        if isinstance(base_orientation, self._pin.Quaternion):
            q = self._pin.Quaternion(base_orientation.coeffs())
            if np.linalg.norm(q.coeffs()) < 1e-12:
                raise ValueError("Quaternion must be nonzero")
            q.normalize()
            RB = q.matrix()
        else:
            value = np.asarray(base_orientation, dtype=float)
            if value.shape == (3, 3):
                RB = _rotation(value, "base_orientation")
            else:
                qv = _vector(value, 4, "base_orientation_xyzw")
                if np.linalg.norm(qv) < 1e-12:
                    raise ValueError("Quaternion must be nonzero")
                q = self._pin.Quaternion(qv)
                q.normalize()
                RB = q.matrix()
        bg = _vector(np.zeros(3) if gyro_bias is None else gyro_bias, 3, "gyro_bias")
        ba = _vector(np.zeros(3) if accel_bias is None else accel_bias, 3, "accel_bias")
        gyro = bg.copy() if omega_tilde is None else _vector(omega_tilde, 3, "omega_tilde")
        wB = self._R_BI@(gyro-bg)
        self.core.reset(rotation=RB@self._R_BI, position=pB+RB@self._r_BI,
                        velocity=RB@(vB+np.cross(wB, self._r_BI)), gyro_bias=bg, accel_bias=ba,
                        covariance=self._initial_covariance if covariance is None else covariance)
        self._last_gyro = gyro

    def _kinematics(self, contacts_schedule, joint_positions):
        mask = np.asarray(contacts_schedule).reshape(-1)
        if mask.size != self._nb_ee or not np.all(np.isin(mask, [0, 1])):
            raise ValueError("contacts_schedule must contain one Boolean/0/1 per configured foot")
        mask = mask.astype(bool)
        qj = _vector(joint_positions, self._nqj, "joint_positions")
        q = np.r_[np.zeros(3), 0., 0., 0., 1., qj]
        if not self._pin.isNormalized(self._rmodel, q, 1e-6):
            raise ValueError("Joint configuration must be normalized on the model's configuration manifold")
        self._pin.computeJointJacobians(self._rmodel, self._rdata, q)
        self._pin.updateFramePlacements(self._rmodel, self._rdata)
        base = self._rdata.oMf[self._base_id]
        # Root coordinates are fixed for FK. The named base may have a fixed
        # offset from root; it has zero actuated-joint Jacobian by construction.
        root_to_imu = self._R_BI.T@base.rotation.T
        positions, jacobians = {}, []
        for i, on in enumerate(mask):
            if not on:
                continue
            foot = self._rdata.oMf[self._foot_ids[i]]
            pB = base.rotation.T@(foot.translation-base.translation)
            positions[i] = self._R_BI.T@(pB-self._r_BI)
            J = self._pin.getFrameJacobian(self._rmodel, self._rdata, self._foot_ids[i],
                                           self._pin.LOCAL_WORLD_ALIGNED)
            # Pinocchio linear rows are 0:3; use the joint tangent columns 6:.
            jacobians.append(root_to_imu@J[:3, 6:])
        count = np.count_nonzero(mask)
        if count:
            Jall = np.vstack(jacobians)
            C = Jall@self._encoder_cov@Jall.T + self._sigma_fk**2*np.eye(3*count)
        else:
            C = np.empty((0, 0))
        schedule = {i: bool(on) for i, on in enumerate(mask)}
        return positions, schedule, C

    def initialize_contacts(self, contacts_schedule, joint_positions, *, omega_tilde=None):
        """Initialize stance feet at the CURRENT state time without propagating.

        Call once at sample zero, after any set_initial_state(). Subsequent contact
        changes are handled by update_filter(); do not repeatedly submit sample zero.
        This method does not infer the initial linear velocity from a gyro sample.
        """
        if self.core.contact_ids:
            raise RuntimeError("Contacts already initialized; use update_filter for subsequent samples")
        positions, schedule, covariance = self._kinematics(contacts_schedule, joint_positions)
        if omega_tilde is not None:
            self._last_gyro = _vector(omega_tilde, 3, "omega_tilde")
        self.core.correct_contacts(positions, schedule, covariance)

    def update_filter(self, a_tilde, omega_tilde, contacts_schedule, joint_positions,
                      joint_velocities=None, *, dt=None):
        """Advance by dt, then update FK/contact measurements at the new time.

        joint_velocities is accepted for compatibility and is not used. If this
        is your first call without initialize_contacts(), it STILL advances time,
        matching the old BaseEKF; the first feet are then initialized at that time.
        """
        started = time.perf_counter()
        interval = self._dt if dt is None else float(dt)
        acceleration = _vector(a_tilde, 3, "a_tilde")
        gyro = _vector(omega_tilde, 3, "omega_tilde")
        if joint_velocities is not None:
            _vector(joint_velocities, self._nvj, "joint_velocities")
        # Validate kinematics before mutating the filter state.
        positions, schedule, covariance = self._kinematics(contacts_schedule, joint_positions)
        self.core.propagate(acceleration, gyro, interval)
        self.core.correct_contacts(positions, schedule, covariance)
        self._last_gyro = gyro
        self._last_update_seconds = time.perf_counter()-started

    def get_filter_output(self):
        """Same dictionary/frames as BaseEKF, including a Pinocchio Quaternion."""
        RB = self.core.R@self._R_BI.T
        pB = self.core.p-RB@self._r_BI
        wB = self._R_BI@(self._last_gyro-self.core.bg)
        vB = RB.T@self.core.v-np.cross(wB, self._r_BI)
        quaternion = self._pin.Quaternion(RB)
        quaternion.normalize()
        return {"base_position": pB.copy(), "base_velocity": vB.copy(),
                "base_orientation": quaternion}

    def get_is_in_imu_frame(self):
        return True

    @property
    def robot_nq(self):
        return self._rmodel.nq

    @property
    def robot_nv(self):
        return self._rmodel.nv

    @property
    def number_of_end_effectors(self):
        return self._nb_ee

    def get_imu_biases(self):
        return {"imu_bias_acceleration": self.core.ba.copy(),
                "imu_bias_orientation": self.core.bg.copy()}

    def get_contact_positions(self):
        return {self._end_effectors_frame_names[i]: self.core._feet[:, k].copy()
                for k, i in enumerate(self.core.contact_ids)}

    def get_covariance(self):
        """Full invariant covariance [rotation,world v,world p,contacts,bg,ba]."""
        return self.core.P.copy()

    def get_diagnostics(self):
        result = self.core.get_diagnostics()
        result["update_seconds"] = self._last_update_seconds
        return result


# Alternative descriptive name, without replacing the user's original BaseEKF.
ContactAidedBaseInEKF = BaseInEKF

# Example usage from base_state_inekf import BaseInEKF
# ekf = BaseInEKF(
#     urdf_file,
#     robot_config,
#     dt=0.001,
# )

# # Register sample-zero contacts without advancing time.
# ekf.initialize_contacts(
#     contacts0,
#     joint_positions0,
#     omega_tilde=gyro0,
# )

# # Process subsequent samples.
# ekf.update_filter(
#     a_tilde,
#     omega_tilde,
#     contacts_schedule,
#     joint_positions,
#     joint_velocities,
# )

# state = ekf.get_filter_output()
