# Legged Robot State Estimation

**Proprioceptive Base and Centroidal State Estimation for Legged Robots.**

This repository provides Python implementations of state estimators for legged
robots. The estimators combine inertial measurements, joint encoders, contact
information, contact forces, or joint torques with a Pinocchio robot model.

The codebase is divided into two estimation groups.

## Base-state estimation

The base-state estimators compute the floating-base position, linear velocity,
and orientation. They also account for IMU accelerometer and gyroscope biases.

- **(1) Extended Kalman Filter (EKF) [1]**

  A classical error-state EKF that propagates the base state with IMU
  measurements and corrects it using contact-constrained leg kinematics.

- **(2) Unscented Kalman Filter on SO(3) (UKF) [2, 3]**

  A scaled unscented Kalman filter adapted to the legged-robot state. The
  orientation sigma points, mean, and residuals are handled directly on
  `SO(3)` rather than by treating rotation as an unconstrained Euclidean vector.

- **(3) Contact-Aided Invariant Extended Kalman Filter (InEKF) [4]**

  A right-invariant EKF that represents rotation, velocity, position, and
  active contact points on the corresponding matrix Lie group. It supports
  switching contacts and IMU-bias estimation.

## Centroidal-state estimation

The centroidal estimators compute center-of-mass position, linear momentum, and
angular momentum.

- **(1) Contact-force-based centroidal EKF [5]**

  Uses measured contact forces and contact-point locations in the centroidal
  momentum dynamics. Euler and RK4 integration are available in the demo.

- **(2) Joint-torque-based centroidal EKF [6]**

  Projects the whole-body dynamics into the null space of the contact
  constraints. This relates measured joint torques to centroidal dynamics
  without requiring measured contact forces.

---

## Key features

- Three base-state estimators: EKF, manifold UKF, and contact-aided InEKF
- Two centroidal-state estimators: contact-force and projected-torque EKFs
- Rotation estimation on `SO(3)` for the UKF and Lie-group states for the InEKF
- Fusion of IMU, joint encoder, contact, force, and torque measurements
- Switching-contact support for legged locomotion
- Robot model loading and rigid-body computations through Pinocchio
- Command-line selection of robot and estimator
- Example Go1 and Solo12 descriptions and datasets
- RMSE reporting, interactive plotting, and headless figure export

---

## Repository structure (high-level)

- `src/base_estimation/`: EKF, UKF, and contact-aided InEKF implementations
- `src/centroidal_estimation/`: force- and torque-based centroidal EKFs
- `demos/`: runnable base-state and centroidal-state demos
- `tests/`: numerical and interface tests for the base-state filters
- `data/`: example trajectories, sensor measurements, and contact schedules
- `files/go1_description/`: Unitree Go1 URDF and estimator configuration
- `files/solo12_description/`: Solo12 URDF and estimator configuration
- `environment.yml`: Conda environment specification

---

## Installation

### (1) Clone

```bash
git clone https://github.com/ShahramKhorshidi/legged_robot_state_estimation.git
cd legged_robot_state_estimation
```

### (2) Create the Conda environment

Using the provided environment file:

```bash
conda env create -f environment.yml
conda activate state_estimation
```

The environment installs Python, NumPy, SciPy, Pinocchio, PyYAML, and
Matplotlib.

---

## Run the estimators

### (1) Base-state estimation demo

Select the robot with `--robot` and the filter with `--estimator`:

```bash
python demos/demo_base_estimation.py --robot solo12 --estimator ekf
python demos/demo_base_estimation.py --robot solo12 --estimator ukf
python demos/demo_base_estimation.py --robot solo12 --estimator inekf
```

The same estimators can be run on another registered robot, for example:

```bash
python demos/demo_base_estimation.py --robot go1 --estimator ekf
```

The demo reports position, velocity, and orientation RMSE and plots each
estimated state against its reference.

### (2) Centroidal-state estimation demo

Use `force` when measured contact forces are available, or `torque` to use the
projected-dynamics estimator with joint torque measurement:

```bash
python demos/demo_centroidal_estimation.py --robot solo12 --estimator force
python demos/demo_centroidal_estimation.py --robot solo12 --estimator torque
```

The Go1 example uses the recorded `cent_state.dat` reference. For datasets
without a separate centroidal-state file, such as Solo12, the demo computes the
reference from the URDF and recorded `q/dq` trajectory.

The force estimator supports both integration methods:

```bash
python demos/demo_centroidal_estimation.py --robot solo12 --estimator force \
    --integration_method rk4
```

### (3) Common demo options

Both demos accept `--samples`, `--data_dir`, `--save`, and `--no_show`. For
example, the following command processes 1,000 samples and saves the figure
without opening a graphical window:

```bash
python demos/demo_base_estimation.py --robot solo12 --estimator ukf \
    --samples 1000 --save results/base_ukf.png --no_show
```

List all estimator-specific options with:

```bash
python demos/demo_base_estimation.py --help
python demos/demo_centroidal_estimation.py --help
```

### (4) Adding a new robot

The framework can be extended to another legged robot by providing its model,
configuration, and recorded data.

#### (i) Robot description

Place the URDF and related assets under:

```text
files/<robot_name>_description/
```

The URDF must define the correct floating-base convention, joint ordering,
inertial parameters, base frame, and contact frames.

#### (ii) Robot configuration

Add a YAML configuration beside the URDF. Use `solo12_config.yaml` as a template. The configuration includes:

- Robot mass and base-link name
- End-effector/contact frame names
- Initial floating-base configuration
- IMU-to-base rotation and translation
- IMU measurement and bias noise parameters
- Optional sampling period `dt` (the default is `0.001` s)

Frame names in the configuration must exactly match names defined in the URDF.

#### (iii) Trajectory and sensor data

Base-state estimation requires generalized position `q`, generalized velocity
`dq`, contact schedule, and IMU measurements. Force-based centroidal estimation
additionally requires one 3D contact force per end effector. Torque-based
centroidal estimation requires the measured joint torques instead.

Store each signal as a tab-separated matrix with one sample per column. The
provided Go1 and Solo12 files in `data/` show the expected layout.

#### (iv) Registering the robot

Add the robot URDF, configuration, data prefix, and estimator input filenames to
`get_robot_paths()` in the corresponding demo script:

- `demos/demo_base_estimation.py`
- `demos/demo_centroidal_estimation.py`

Then run:

```bash
python demos/demo_base_estimation.py \
    --robot <robot_name> --estimator <ekf|ukf|inekf>

python demos/demo_centroidal_estimation.py \
    --robot <robot_name> --estimator <force|torque>
```

---

## Tests

Run the numerical and interface tests with:

```bash
python -m unittest discover -s tests -v
```

---

## Citation

If you use the torque-based centroidal state estimator in an academic work, please cite the following paper:
```
@inproceedings{khorshidi23icra,
  title={On the Use of Torque Measurement in Centroidal State Estimation}, 
  author={Khorshidi, Shahram and Gazar, Ahmad and Rotella, Nicholas and Naveau, Maximilien and Righetti, Ludovic and Bennewitz, Maren and Khadiv, Majid},
  booktitle={Proc. of the IEEE International Conference on Robotics and Automation (ICRA)}, 
  year={2023}
}
```

---

## References

[1] M. Bloesch, M. Hutter, M. A. Hoepflinger, S. Leutenegger, C. Gehring,
C. D. Remy, and R. Siegwart, “State Estimation for Legged Robots: Consistent
Fusion of Leg Kinematics and IMU,” *Robotics: Science and Systems VIII*, 2012.
[paper](https://www.roboticsproceedings.org/rss08/p03.html),
[doi](https://doi.org/10.15607/RSS.2012.VIII.003).

[2] E. A. Wan and R. van der Merwe, “The Unscented Kalman Filter for Nonlinear
Estimation,” *Proceedings of the IEEE 2000 Adaptive Systems for Signal
Processing, Communications, and Control Symposium*, pp. 153–158, 2000.
[doi](https://doi.org/10.1109/ASSPCC.2000.882463).

[3] M. Brossard, A. Barrau, and S. Bonnabel, “A Code for Unscented Kalman
Filtering on Manifolds (UKF-M),” *2020 IEEE International Conference on
Robotics and Automation (ICRA)*, pp. 5701–5708, 2020.
[paper](https://arxiv.org/abs/2002.00878),
[doi](https://doi.org/10.1109/ICRA40945.2020.9197489).

[4] R. Hartley, M. Ghaffari Jadidi, J. W. Grizzle, and R. M. Eustice,
“Contact-Aided Invariant Extended Kalman Filtering for Legged Robot State
Estimation,” *Robotics: Science and Systems XIV*, 2018.
[paper](https://www.roboticsproceedings.org/rss14/p50.html),
[doi](https://doi.org/10.15607/RSS.2018.XIV.050).

[5] N. Rotella, A. Herzog, S. Schaal, and L. Righetti, “Humanoid Momentum
Estimation Using Sensed Contact Wrenches,” *2015 IEEE-RAS 15th International
Conference on Humanoid Robots (Humanoids)*, pp. 556–563, 2015.
[paper](https://arxiv.org/abs/1507.04401),
[doi](https://doi.org/10.1109/HUMANOIDS.2015.7363417).

[6] S. Khorshidi, A. Gazar, N. Rotella, M. Naveau, L. Righetti, M. Bennewitz,
and M. Khadiv, “On the Use of Torque Measurement in Centroidal State
Estimation,” *2023 IEEE International Conference on Robotics and Automation
(ICRA)*, 2023.
[paper](https://arxiv.org/abs/2202.12574),
[doi](https://doi.org/10.1109/ICRA48891.2023.10160823).
