# legged_robot_state_estimation
This repository provides python code templates of different state estimators for legged robots.

### Base state estimation
This estimator fuses IMU data with leg odometry in EKF and UKF estimation framework. 

### Centoridal state estimation
This estimator uses contact wrencehes and joint torque measurement to estimate the centroidal states.

### Dependencies
numpy, pinocchio, matplotlib.
```
pip install numpy
pip install pin
pip install -U matplotlib
```
