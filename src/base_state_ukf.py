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


class UKF(object):
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
        self.__end_effectors_frame_names = robot_config.get('end_effectors_frame_names')
        self.__rot_base_to_imu = np.array(robot_config.get('rot_base_to_imu'))
        self.__r_base_to_imu = np.array(robot_config.get('r_base_to_imu'))
        self.__SE3_imu_to_base = pin.SE3(
            self.__rot_base_to_imu.T, self.__r_base_to_imu
        )


if __name__ == "__main__":
    cur_dir = Path.cwd()
    robot_urdf = cur_dir/"files"/"go1.urdf"
    robot_config = cur_dir/"files"/"go1_config.yaml"
    solo_cent_ekf = UKF(str(robot_urdf), robot_config)