"""Extended Kalman Filter Class
License BSD-3-Clause
Copyright (c) 2021, New York University and Max Planck Gesellschaft.
Authors: Ahmad Gazar, Shahram Khorshidi
"""

import yaml
import numpy as np
import pinocchio as pin
from pathlib import Path
from numpy.linalg import inv
from pinocchio import Quaternion

# plus and minus operators on SO3
def box_plus(R, theta):
    return R @ pin.exp(theta)

def box_minus(R_plus, R):
    return pin.log(R.T @ R_plus)


class EKF(object):
    """EKF class for estimation of the position, velocity, orientation of the base frame on the robot, and IMU bias_linear_acceleration and bias_angular_rate.
    Position and orientation are expressed in the world, velocity is expressed in the base_frame, and bias terms are expressed in the IMU frame. EKF_frame
    can be defined in the Base or IMU frame.

    Attributes:
        rmodel : obj:'pinocchio.Model'
        rdata : obj:'pinocchio.Data'
        nx : int
            Dimension of the error state vector, (delta_x = [delta_p, delta_v, delta_theta, delta_b_a, delta_b_omega]).
        init_robot_config : ndarray
            Initial configuration of the robot.
        dt : float
            Discretization time.
        g_vector : np.array(3,1)
            Gravity acceleration vector.
        base_frame_name : str
        end_effectors_frame_names : list
        nb_ee : int
            Number of end_effectors
        ekf_in_imu_frame : bool
            False, EKF default frame is in the Base frame. True, EKF_frame is in the IMU frame.
        base_state : dict
            Base states estimated by EKF.
            (x_base = {p:(np.array(3,1)), v:(np.array(3,1)), q:(pinocchio.Quaternion)).
        mu_pre : dict
            A priori estimate of the mean of the state vector,
            (x = {p:(np.array(3,1)), v:(np.array(3,1)), q:(pinocchio.Quaternion), b_a:(np.array(3,1)), b_omega:(np.array(3,1))}).
        mu_post : dict
            A posteriori estimate of the mean of the state vector (x).
        Sigma_pre : np.array(15,15)
            A priori error covariance matrix.
        Sigma_post : np.array(15,15)
            A posteriori error covariance matrix.
        SE3_imu_to_base : pinocchio.SE3
            SE3 transformation from IMU to Base.
        SE3_base_to_imu : pinocchio.SE3
            SE3 transformation from Base to IMU.
        Q_a : np.array(3,3)
            Continuous accelerometer noise covariance.
        Q_omega : np.array(3,3)
            Continuous gyroscope noise covariance.
        Qb_a : np.array(3,3)
            Continuous accelerometer bias noise covariance.
        Qb_omega : np.array(3,3)
            Continuous gyroscope bias noise covariance.
        R : ndarray
            Continuous measurement noise covariance.
    """

    def __init__(self,urdf_file, robot_config, dt=0.001):
        """Initializes the EKF.

        Args:
            robot_config : Configuration file of the robot.
            dt (float): Discretization time.
        """
        # private members
        # Load configuration from YAML file
        with open(robot_config, 'r') as file:
            config = yaml.safe_load(file)
        self._rmodel = pin.buildModelFromUrdf(urdf_file, pin.JointModelFreeFlyer())
        self.__rdata = self._rmodel.createData()
        robot_config = config.get('robot', {})
        self.__nx = 5 * 3
        self.__init_robot_config = np.copy(robot_config.initial_configuration)
        self.__dt = dt
        self.__g_vector = np.array([0, 0, -9.81])
        self.__base_frame_name = robot_config.base_link_name
        self.__end_effectors_frame_names = robot_config.end_effector_names
        self.__nb_ee = len(self.__end_effectors_frame_names)
        self.__ekf_in_imu_frame = False
        self.__base_state = dict.fromkeys(
            [
                "base_position",
                "base_velocity",
                "base_orientation",
            ]
        )
        self.__mu_pre = dict.fromkeys(
            [
                "ekf_frame_position",
                "ekf_frame_velocity",
                "ekf_frame_orientation",
                "imu_bias_acceleration",
                "imu_bias_orientation",
            ]
        )
        self.__mu_post = dict.fromkeys(
            [
                "ekf_frame_position",
                "ekf_frame_velocity",
                "ekf_frame_orientation",
                "imu_bias_acceleration",
                "imu_bias_orientation",
            ]
        )
        self.__Sigma_pre = np.zeros((self.__nx, self.__nx))
        self.__Sigma_post = np.zeros((self.__nx, self.__nx))
        self.__omega_hat = np.zeros(3)
        self.__omega_base_prev = np.zeros(3)
        self.__base_ang_acc = np.zeros(3)
        self.__rot_base_to_imu = np.array(robot_config.get('rot_base_to_imu'))
        self.__r_base_to_imu = np.array(robot_config.get('r_base_to_imu'))
        self.__SE3_imu_to_base = pin.SE3(
            self.__rot_base_to_imu.T, self.__r_base_to_imu
        )
        self.__SE3_base_to_imu = self.__SE3_imu_to_base.inverse()
        self.__Q_a = self.__dt * np.diag((0.0001962**2) * np.ones([3]))
        self.__Q_omega = self.__dt * np.diag((0.0000873**2) * np.ones([3]))
        self.__Qb_a = np.diag((0.0001**2) * np.ones([3]))
        self.__Qb_omega = np.diag((0.000309**2) * np.ones([3]))
        self.__R = np.zeros((3 * self.__nb_ee, 3 * self.__nb_ee), dtype=float)
        np.fill_diagonal(self.__R, np.array([1e-5, 1e-5, 1e-5]))
        # call private methods
        self.__init_filter()

    # private methods
    def __init_filter(self):
        """Sets the initial values for the 'a posteriori estimate'."""
        base_se3 = self.__compute_base_pose_se3(self.__init_robot_config)
        if self.__ekf_in_imu_frame:
            base_motion = pin.Motion(
                np.zeros(3, dtype=float), np.zeros(3, dtype=float)
            )
            imu_se3 = base_se3.act(self.__SE3_imu_to_base)
            imu_motion = self.__SE3_imu_to_base.actInv(base_motion)
            q = Quaternion(imu_se3.rotation)
            q.normalize()
            self.__mu_post["ekf_frame_position"] = imu_se3.translation
            self.__mu_post["ekf_frame_orientation"] = q
            self.__mu_post["ekf_frame_velocity"] = imu_motion.linear
        else:
            q = Quaternion(base_se3.rotation)
            q.normalize()
            self.__mu_post["ekf_frame_position"] = base_se3.translation
            self.__mu_post["ekf_frame_orientation"] = q
            self.__mu_post["ekf_frame_velocity"] = np.zeros(3, dtype=float)
        self.__mu_post["imu_bias_acceleration"] = np.zeros(3, dtype=float)
        self.__mu_post["imu_bias_orientation"] = np.zeros(3, dtype=float)

    def __compute_base_pose_se3(self, robot_configuration):
        """Returns the SE3 transformation from base to world.

        Args:
            robot_configuration (ndarray): Initial configuration of the robot.

        Returns:
            pinocchio.SE3
        """
        pin.forwardKinematics(self.__rmodel, self.__rdata, robot_configuration)
        pin.framesForwardKinematics(
            self.__rmodel, self.__rdata, robot_configuration
        )
        base_link_index = self.__rmodel.getFrameId(self.__base_frame_name)
        return self.__rdata.oMf[base_link_index]

    # public methods
    # accessors
    def get_is_in_imu_frame(self):
        """Returns a boolean value for the ekf frame location.

        Returns:
            bool
        """
        return self.__ekf_in_imu_frame

    # mutators (use only if you know what you are doing)
    def set_mu_pre(self, key, value):
        """Sets a value for a state in the 'a priori estimate of the mean of the state vector'.

        Args:
            key (str): Key of the state in the dictionary.
            value (np.array(3,1)) or (pinocchio.Quaternion): Value for the corresponding state.
        """
        self.__mu_pre[key] = value

    def set_mu_post(self, key, value):
        """Sets a value for a state in the 'a posteriori estimate of the mean of the state vector'.

        Args:
            key (str): Key of the state in the dictionary.
            value (np.array(3,1)) or (pinocchio.Quaternion): Value for the corresponding state.
        """
        self.__mu_post[key] = value

    def set_ekf_in_imu_frame(self, bool_value):
        """Sets the location of EKF frame on the robot, and updates the EKf initialization.

        Args:
            bool_value (bool): False in Base frame, True in IMU frame.
        """
        self.__ekf_in_imu_frame = bool_value
        if bool_value:
            self.__init_filter()

    def set_SE3_imu_to_base(self, rotation, translation):
        """Sets the SE3 transformation from IMU to Base, and updates SE3 from Base to IMU.

        Args:
            rotation (np.array(3,3))
            translation (np.array(3,1))
        """
        self.__SE3_imu_to_base.rotation = rotation
        self.__SE3_imu_to_base.translation = translation
        self.__SE3_base_to_imu = self.__SE3_imu_to_base.inverse()

    def set_meas_noise_cov(self, value):
        """Sets a value for continuous measurement noise covariance.

        Args:
            value (np.array(3,1))
        """
        np.fill_diagonal(self.__R, value)

    def compute_end_effectors_FK_quantities(
        self, joint_positions, joint_velocities
    ):
        """Returns end effectors position and linear velocity w.r.t. base, expressed in base.

        Args:
            joint_positions (ndarray): Generalized joint positions.
            joint_velocities (ndarray): Generalized joint velocities.

        Returns:
            list: Position of all feet in the base frame.
            list: Linear velocity of all feet in the base frame.
        """
        # locking the base frame to the world frame
        base_pose = np.zeros(7)
        base_pose[6] = 1.0
        robot_configuration = np.concatenate([base_pose, joint_positions])
        robot_velocity = np.concatenate([np.zeros(6), joint_velocities])
        end_effectors_positions = []
        end_effectors_velocities = []
        pin.forwardKinematics(
            self.__rmodel, self.__rdata, robot_configuration, robot_velocity
        )
        pin.framesForwardKinematics(
            self.__rmodel, self.__rdata, robot_configuration
        )
        for i in range(self.__nb_ee):
            frame_index = self.__rmodel.getFrameId(
                self.__end_effectors_frame_names[i]
            )
            frame_position = self.__rdata.oMf[frame_index].translation
            frame_velocity = pin.getFrameVelocity(
                self.__rmodel,
                self.__rdata,
                frame_index,
                pin.LOCAL_WORLD_ALIGNED,
            )
            end_effectors_positions.append(frame_position)
            end_effectors_velocities.append(frame_velocity.linear)
        return end_effectors_positions, end_effectors_velocities

    def integrate_model(self, a_tilde, omega_tilde):
        """Calculates the 'a priori estimate of the mean of the state vector' from deterministic transient model by first_order integration.

        Args:
            a_tilde (np.array(3,1)): IMU accelerometer in the IMU frame.
            omega_tilde (np.array(3,1)): IMU gyroscope in the IMU frame.
        """
        dt, g = self.__dt, self.__g_vector
        mu_post = self.__mu_post
        v_post = mu_post["ekf_frame_velocity"]
        q_post = mu_post["ekf_frame_orientation"]
        b_a_post, b_omega_post = (
            self.__mu_post["imu_bias_acceleration"],
            self.__mu_post["imu_bias_orientation"],
        )
        R_post = q_post.matrix()

        if self.__ekf_in_imu_frame:
            # IMU readings in the IMU frame
            a_hat = a_tilde - b_a_post
            omega_hat = omega_tilde - b_omega_post
        else:
            # compute base acceleration numerically
            rot_imu_to_base = self.__SE3_imu_to_base.rotation
            r_base_to_imu_in_base = self.__SE3_imu_to_base.translation
            omega_base = rot_imu_to_base @ (omega_tilde - b_omega_post)
            self.__base_ang_acc = (1.0 / dt) * (
                omega_base - self.__omega_base_prev
            )
            self.__omega_base_prev = omega_base
            a_base_in_base = (
                rot_imu_to_base @ (a_tilde - b_a_post)
                + np.cross(self.__base_ang_acc, -r_base_to_imu_in_base)
                + np.cross(
                    omega_base, np.cross(omega_base, -r_base_to_imu_in_base)
                )
            )
            # IMU readings in the base frame
            a_hat = a_base_in_base
            omega_hat = omega_base

        R_pre = box_plus(R_post, omega_hat * self.__dt)
        q_pre = Quaternion(R_pre)
        q_pre.normalize()
        self.__mu_pre["ekf_frame_position"] = (
            self.__mu_post["ekf_frame_position"] + (R_post @ v_post) * dt
        )
        self.__mu_pre["ekf_frame_velocity"] = (
            self.__mu_post["ekf_frame_velocity"]
            + (-pin.skew(omega_hat) @ v_post + R_post.T @ g + a_hat) * dt
        )
        self.__mu_pre["ekf_frame_orientation"] = q_pre
        self.__mu_pre["imu_bias_acceleration"] = self.__mu_post[
            "imu_bias_acceleration"
        ]
        self.__mu_pre["imu_bias_orientation"] = self.__mu_post[
            "imu_bias_orientation"
        ]
        self.__omega_hat = omega_hat

    def compute_discrete_prediction_jacobian(self):
        """Returns the discrete linearized error dynamic matrix of the prediction step.

        Returns:
            np.array(15,15)
        """
        dt, g = self.__dt, self.__g_vector
        Fc = np.zeros((self.__nx, self.__nx), dtype=float)
        mu_pre = self.__mu_pre
        q_pre = mu_pre["ekf_frame_orientation"]
        v_pre = mu_pre["ekf_frame_velocity"]
        omega_hat = self.__omega_hat
        R_pre = q_pre.matrix()
        # dp/ddelta_x
        Fc[0:3, 3:6] = R_pre
        Fc[0:3, 6:9] = -R_pre @ pin.skew(v_pre)
        # dv/ddelta_x
        Fc[3:6, 3:6] = -pin.skew(omega_hat)
        Fc[3:6, 6:9] = pin.skew(R_pre.T @ g)
        Fc[3:6, 9:12] = -np.eye(3)
        Fc[3:6, 12:15] = -pin.skew(v_pre)
        # dtheta/ddelta_x
        Fc[6:9, 6:9] = -pin.skew(omega_hat)
        Fc[6:9, 12:15] = -np.eye(3)
        Fk = np.eye(self.__nx) + Fc * dt
        return Fk

    def compute_noise_jacobian(self):
        """Returns the continuous noise jacobian of the prediction step.

        Returns:
            np.array(15,12)
        """
        v_pre = self.__mu_pre["ekf_frame_velocity"]
        Lc = np.zeros((self.__nx, self.__nx - 3), dtype=float)
        Lc[3:6, 0:3] = -np.eye(3)
        Lc[3:6, 3:6] = -pin.skew(v_pre)
        Lc[6:9, 3:6] = -np.eye(3)
        Lc[9:12, 6:9] = np.eye(3)
        Lc[12:15, 9:12] = np.eye(3)
        return Lc

    def construct_continuous_noise_covariance(self):
        """Returns the continuous noise covariance corresponding to the process noise vector.

        Returns:
            np.array(12,12)
        """
        Qc = np.zeros((self.__nx - 3, self.__nx - 3), dtype=float)
        Qc[0:3, 0:3] = self.__Q_a
        Qc[3:6, 3:6] = self.__Q_omega
        Qc[6:9, 6:9] = self.__Qb_a
        Qc[9:12, 9:12] = self.__Qb_omega
        return Qc

    def construct_discrete_noise_covariance(self, Fk, Lc, Qc):
        """Returns the discrete noise covariance by using zero-order hold and truncating higher-order terms.

        Args:
            Fk (np.array(15,15)): Discrete linearized error dynamic matrix.
            Lc (np.array(15,12)): Continuous noise jacobian.
            Qc (np.array(12,12)): Continuous process noise covariance.

        Returns:
            np.array(15,15)
        """
        Qk_left = Fk @ Lc
        Qk_right = Lc.T @ Fk.T
        return (Qk_left @ Qc @ Qk_right) * self.__dt

    def construct_discrete_measurement_noise_covariance(self):
        """Returns the discrete measurement noise covariance.

        Returns:
            ndarray
        """
        return (1 / self.__dt) * self.__R

    def prediction_step(self):
        """Calculates the 'a priori error covariance matrix' in the prediction step."""
        Fk = self.compute_discrete_prediction_jacobian()
        Lc = self.compute_noise_jacobian()
        Qc = self.construct_continuous_noise_covariance()
        Qk = self.construct_discrete_noise_covariance(Fk, Lc, Qc)
        self.__Sigma_pre = (Fk @ self.__Sigma_post @ Fk.T) + Qk

    # assuming contact logic variables coming from the contact schedule for now
    # TODO contact logic prediction using a probabilistic model
    def measurement_model(
        self, contacts_schedule, joint_positions, joint_velocities
    ):
        """Returns the discrete measurement jacobian matrix and the measurement residual.

        Args:
            contacts_schedule (list): Contact schedule of the feet.
            joint_positions (ndarray): Generalized joint positions.
            joint_velocities (ndarray): Generalized joint velocities.

        Returns:
            ndarray: Discrete measurement jacobian matrix.
            ndarray: Measurement residual.
        """
        Hk = np.zeros((3 * self.__nb_ee, self.__nx))
        predicted_frame_velocity = np.zeros(3 * self.__nb_ee)
        measured_frame_velocity = np.zeros(3 * self.__nb_ee)
        # end effectors frame positions and velocities expressed in the base frame
        ee_positions, ee_velocities = self.compute_end_effectors_FK_quantities(
            joint_positions, joint_velocities
        )
        i = 0
        for index in range(self.__nb_ee):
            # check if foot is in contact based on contact schedule
            if contacts_schedule[index]:
                # compute measurement jacobian
                Hk[i : i + 3, 3:6] = np.eye(3)
                Hk[i : i + 3, 12:15] = pin.skew(ee_positions[index])
                # get the predicted frame velocity
                predicted_frame_velocity[i : i + 3] = self.__mu_pre[
                    "ekf_frame_velocity"
                ]
                if self.__ekf_in_imu_frame:
                    base_motion = pin.Motion(
                        -ee_velocities[index]
                        - pin.skew(self.__omega_hat) @ ee_positions[index],
                        self.__SE3_imu_to_base.rotation @ self.__omega_hat,
                    )
                    measured_frame_velocity[
                        i : i + 3
                    ] = self.__SE3_base_to_imu.act(base_motion).linear
                else:
                    measured_frame_velocity[i : i + 3] = (
                        -ee_velocities[index]
                        - pin.skew(self.__omega_hat) @ ee_positions[index]
                    )
            else:
                predicted_frame_velocity[i : i + 3] = np.zeros(3)
                measured_frame_velocity[i : i + 3] = np.zeros(3)
            i += 3
        error = measured_frame_velocity - predicted_frame_velocity
        return Hk, error

    def compute_innovation_covariance(self, Hk, Rk):
        """Returns the innovation covariance matrix.

        Args:
            Hk (ndarray): Discrete measurement jacobian matrix.
            Rk (ndarray): Discrete measurement noise covariance.

        Returns:
            ndarray
        """
        return (Hk @ self.__Sigma_pre @ Hk.T) + Rk

    def update_step(
        self, contacts_schedule, joint_positions, joint_velocities
    ):
        """Calculates the 'a posteriori error covariance matrix' and the 'a posteriori estimate of the mean of the state vector',
            based on new kinematic measurements.

        Args:
            contacts_schedule (list): Contact schedule of the feet.
            joint_positions (ndarray): Generalized joint positions.
            joint_velocities (ndarray): Generalized joint velocities.
        """
        q_pre = self.__mu_pre["ekf_frame_orientation"]
        R_pre = q_pre.matrix()
        Hk, measurement_error = self.measurement_model(
            contacts_schedule, joint_positions, joint_velocities
        )
        Rk = self.construct_discrete_measurement_noise_covariance()
        Sk = self.compute_innovation_covariance(Hk, Rk)
        K = (self.__Sigma_pre @ Hk.T) @ inv(Sk)  # kalman gain
        delta_x = K @ measurement_error
        self.__Sigma_post = (np.eye(self.__nx) - (K @ Hk)) @ self.__Sigma_pre
        self.__mu_post["ekf_frame_position"] = (
            self.__mu_pre["ekf_frame_position"] + delta_x[0:3]
        )
        self.__mu_post["ekf_frame_velocity"] = (
            self.__mu_pre["ekf_frame_velocity"] + delta_x[3:6]
        )
        q_post = Quaternion(box_plus(R_pre, delta_x[6:9]))
        q_post.normalize()
        self.__mu_post["ekf_frame_orientation"] = q_post
        self.__mu_post["imu_bias_acceleration"] = (
            self.__mu_pre["imu_bias_acceleration"] + delta_x[9:12]
        )
        self.__mu_post["imu_bias_orientation"] = (
            self.__mu_pre["imu_bias_orientation"] + delta_x[12:15]
        )
    
    def no_update(self):
        self.__mu_post["ekf_frame_position"] = self.__mu_pre["ekf_frame_position"]
        self.__mu_post["ekf_frame_velocity"] = self.__mu_pre["ekf_frame_velocity"]
        self.__mu_post["ekf_frame_orientation"] = self.__mu_pre["ekf_frame_orientation"]
        self.__mu_post["imu_bias_acceleration"] = self.__mu_pre["imu_bias_acceleration"]
        self.__mu_post["imu_bias_orientation"]  = self.__mu_pre["imu_bias_orientation"] 

    def update_filter(
        self,
        a_tilde,
        omega_tilde,
        contacts_schedule,
        joint_positions,
        joint_velocities,
    ):
        """Updates the filter.

        Args:
            a_tilde (np.array(3,1)): IMU accelerometer in the IMU frame.
            omega_tilde (np.array(3,1)): IMU gyroscope in the IMU frame.
            contacts_schedule (list): Contact schedule of the feet.
            joint_positions (ndarray): Generalized joint positions.
            joint_velocities (ndarray): Generalized joint velocities.
        """
        self.integrate_model(a_tilde, omega_tilde)
        self.prediction_step()
        self.update_step(contacts_schedule, joint_positions, joint_velocities)

    def get_filter_output(self):
        """Returns the base states, estimated by the EKF.

        Returns:
            dict
        """
        # mu post is expressed in the imu frame
        if self.__ekf_in_imu_frame:
            imu_se3 = pin.SE3(
                self.__mu_post["ekf_frame_orientation"].matrix(),
                self.__mu_post["ekf_frame_position"],
            )
            imu_motion = pin.Motion(
                self.__mu_post["ekf_frame_velocity"], self.__omega_hat
            )
            base_se3 = imu_se3.act(self.__SE3_base_to_imu)
            base_motion = self.__SE3_imu_to_base.act(imu_motion)
            q = Quaternion(base_se3.rotation)
            q.normalize()
            self.__base_state["base_position"] = base_se3.translation
            self.__base_state["base_velocity"] = base_motion.linear
            self.__base_state["base_orientation"] = q
        # mu post is expressed in the base frame.
        else:
            self.__base_state["base_position"] = self.__mu_post[
                "ekf_frame_position"
            ]
            self.__base_state["base_velocity"] = self.__mu_post[
                "ekf_frame_velocity"
            ]
            self.__base_state["base_orientation"] = self.__mu_post[
                "ekf_frame_orientation"
            ]
        return self.__base_state


if __name__ == "__main__":
    cur_dir = Path.cwd()
    robot_urdf = cur_dir/"files"/"go1.urdf"
    robot_config = cur_dir/"files"/"go1_config.yaml"
    robot_base_ekf = EKF(str(robot_urdf), robot_config)

    # solo_EKF.set_meas_noise_cov(np.array([1e-4, 1e-4, 1e-4]))
    # f_tilde = random.rand(3)
    # w_tilde = random.rand(3)
    # # contacts schedule for solo12 end_effoctors: ['FL_FOOT', 'FR_FOOT', 'HL_FOOT', 'HR_FOOT']
    # contacts_schedule = [True, True, True, True]
    # joint_positions = random.rand(12)
    # joint_velocities = random.rand(12)
    # solo_EKF.update_filter(
    #     f_tilde,
    #     w_tilde,
    #     contacts_schedule,
    #     joint_positions,
    #     joint_velocities,
    # )
    # base_state = solo_EKF.get_filter_output()
