import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin


# Allow this file to be run directly from any working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from base_estimation.base_est_ekf import BaseEKF
from base_estimation.base_est_ukf import BaseUKF
from base_estimation.base_state_inekf import BaseInEKF


def load_data(path, data_prefix):
    """Load the robot trajectory, contact schedule, and IMU measurements."""
    path = Path(path)
    q = np.loadtxt(path / f"{data_prefix}_q.dat", delimiter="\t")
    dq = np.loadtxt(path / f"{data_prefix}_dq.dat", delimiter="\t")
    contacts = np.loadtxt(path / f"{data_prefix}_cnt.dat", delimiter="\t")
    imu = np.loadtxt(path / f"{data_prefix}_imu.dat", delimiter="\t")

    arrays = {"q": q, "dq": dq, "contacts": contacts, "imu": imu}
    for name, value in arrays.items():
        if value.ndim != 2:
            raise ValueError(f"{name} data must be a two-dimensional array")

    number_of_samples = q.shape[1]
    if any(value.shape[1] != number_of_samples for value in arrays.values()):
        raise ValueError("All input files must contain the same number of samples")
    if q.shape[0] < 8 or dq.shape[0] != q.shape[0] - 1:
        raise ValueError("q and dq do not have compatible floating-base dimensions")
    if imu.shape[0] != 6:
        raise ValueError("IMU data must contain 3 accelerometer and 3 gyro rows")

    return q, dq, contacts, imu[:3, :], imu[3:, :]


def repo_root_from_this_file():
    # demos/demo_base_estimation.py -> repository root is one level above demos/
    return Path(__file__).resolve().parents[1]


def get_robot_paths(root, robot):
    """Central mapping from a robot name to its files and data prefix."""
    robots = {
        "go1": {
            "urdf": root / "files" / "go1_description" / "go1.urdf",
            "config": root / "files" / "go1_description" / "go1_config.yaml",
            "data_prefix": "robot",
        },

        "solo12": {
                    "urdf": root / "files" / "solo12_description" / "solo12.urdf",
                    "config": root / "files" / "solo12_description" / "solo12_config.yaml",
                    "data_prefix": "solo_robot",
                },
        # Add more robots here as you extend the framework.
        # "anymal": {
        #     "urdf": root / "files" / "anymal_description" / "anymal.urdf",
        #     "config": root / "files" / "anymal_description" / "anymal_config.yaml",
        #     "data_prefix": "anymal_robot",
        # },
    }
    if robot not in robots:
        available = ", ".join(robots)
        raise ValueError(f"Unknown robot '{robot}'. Available robots: {available}")
    return robots[robot]


def create_estimator(args, robot_paths):
    """Create the estimator selected on the command line."""
    urdf = str(robot_paths["urdf"])
    config = robot_paths["config"]

    if args.estimator == "ekf":
        estimator = BaseEKF(urdf, config, dt=args.dt)
        estimator.set_meas_noise_cov(
            np.full(3, args.measurement_noise, dtype=float)
        )
    elif args.estimator == "ukf":
        estimator = BaseUKF(urdf, config, dt=args.dt)
        estimator.set_meas_noise_cov(
            np.full(3, args.measurement_noise, dtype=float)
        )
    elif args.estimator == "inekf":
        estimator = BaseInEKF(
            urdf,
            config,
            dt=args.dt,
            sigma_encoder=args.sigma_encoder,
            sigma_kinematics=args.sigma_kinematics,
            sigma_contact=args.sigma_contact,
            contact_gate_probability=args.contact_gate_probability,
        )
    else:
        raise ValueError(f"Unknown estimator '{args.estimator}'")

    return estimator


def initialize_estimator(
    estimator_name,
    estimator,
    q,
    dq,
    contacts,
    angular_velocity,
):
    """Initialize all filters from the first sample of the trajectory."""
    orientation = pin.Quaternion(q[3:7, 0])
    orientation.normalize()

    if estimator_name == "inekf":
        estimator.set_initial_state(
            q[:3, 0],
            dq[:3, 0],
            orientation,
            omega_tilde=angular_velocity[:, 0],
        )
        estimator.initialize_contacts(
            contacts[:, 0],
            q[7:, 0],
            omega_tilde=angular_velocity[:, 0],
        )
    else:
        estimator.set_mu_post("ekf_frame_position", q[:3, 0].copy())
        estimator.set_mu_post("ekf_frame_velocity", dq[:3, 0].copy())
        estimator.set_mu_post("ekf_frame_orientation", orientation)


def run_estimation(
    estimator_name,
    estimator,
    q,
    dq,
    contacts,
    linear_acceleration,
    angular_velocity,
    number_of_samples,
):
    """Run one selected base-state estimator over the loaded trajectory."""
    initialize_estimator(
        estimator_name,
        estimator,
        q,
        dq,
        contacts,
        angular_velocity,
    )

    estimated_position = np.zeros((number_of_samples, 3))
    estimated_velocity = np.zeros((number_of_samples, 3))
    estimated_rpy = np.zeros((number_of_samples, 3))
    reference_rpy = np.zeros((number_of_samples, 3))
    rotation_error = np.zeros((number_of_samples, 3))

    for i in range(number_of_samples):
        # Sample zero defines the initial condition. Each later call advances
        # the estimator by one dt and then applies the contact correction.
        if i > 0:
            estimator.update_filter(
                linear_acceleration[:, i],
                angular_velocity[:, i],
                contacts[:, i],
                q[7:, i],
                dq[6:, i],
            )

        state = estimator.get_filter_output()
        estimated_position[i, :] = state["base_position"]
        estimated_velocity[i, :] = state["base_velocity"]

        reference_orientation = pin.Quaternion(q[3:7, i])
        reference_orientation.normalize()
        estimated_orientation = state["base_orientation"]
        reference_rotation = reference_orientation.matrix()
        estimated_rotation = estimated_orientation.matrix()

        reference_rpy[i, :] = pin.utils.matrixToRpy(reference_rotation)
        estimated_rpy[i, :] = pin.utils.matrixToRpy(estimated_rotation)
        rotation_error[i, :] = pin.log3(
            reference_rotation.T @ estimated_rotation
        )

    return {
        "reference_position": q[:3, :number_of_samples].T,
        "reference_velocity": dq[:3, :number_of_samples].T,
        "reference_rpy": reference_rpy,
        "estimated_position": estimated_position,
        "estimated_velocity": estimated_velocity,
        "estimated_rpy": estimated_rpy,
        "rotation_error": rotation_error,
    }


def print_errors(estimator_name, results):
    position_error = (
        results["estimated_position"] - results["reference_position"]
    )
    velocity_error = (
        results["estimated_velocity"] - results["reference_velocity"]
    )

    position_rmse = np.sqrt(np.mean(position_error**2))
    velocity_rmse = np.sqrt(np.mean(velocity_error**2))
    orientation_rmse = np.sqrt(np.mean(results["rotation_error"] ** 2))

    print(f"{estimator_name.upper()} results over {position_error.shape[0]} samples")
    print(f"Position RMSE:    {position_rmse:.8f} m")
    print(f"Velocity RMSE:    {velocity_rmse:.8f} m/s")
    print(f"Orientation RMSE: {orientation_rmse:.8f} rad")


def plot_results(time, estimator_name, results, save_path=None, show=True):
    """Plot reference and estimated position, velocity, and orientation."""
    quantities = (
        ("reference_position", "estimated_position", "Position (m)"),
        ("reference_velocity", "estimated_velocity", "Velocity (m/s)"),
        ("reference_rpy", "estimated_rpy", "Orientation RPY (rad)"),
    )
    axes_names = ("x", "y", "z")
    figure, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)

    for row, (reference_key, estimate_key, label) in enumerate(quantities):
        for column, axis_name in enumerate(axes_names):
            axis = axes[row, column]
            axis.plot(
                time,
                results[reference_key][:, column],
                "b",
                linewidth=0.9,
                label="Reference",
            )
            axis.plot(
                time,
                results[estimate_key][:, column],
                "r--",
                linewidth=0.9,
                label=estimator_name.upper(),
            )
            axis.set_title(f"{label}: {axis_name}")
            axis.grid(True)

    axes[0, 0].legend(loc="best")
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    figure.suptitle(f"Base-state estimation with {estimator_name.upper()}")
    figure.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a legged-robot base-state estimator"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="go1",
        help="Robot name registered in get_robot_paths (default: go1)",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="ekf",
        choices=["ekf", "ukf", "inekf"],
        help="Base-state estimator (default: ekf)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Data directory (default: <repository>/data)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.001,
        help="Sampling period in seconds (default: 0.001)",
    )
    parser.add_argument(
        "--measurement_noise",
        type=float,
        default=0.1,
        help="EKF/UKF leg-odometry noise covariance diagonal (default: 0.1)",
    )
    parser.add_argument(
        "--sigma_encoder",
        type=float,
        default=1e-3,
        help="InEKF encoder noise standard deviation (default: 1e-3)",
    )
    parser.add_argument(
        "--sigma_kinematics",
        type=float,
        default=0.1,
        help="InEKF forward-kinematics noise standard deviation (default: 0.1)",
    )
    parser.add_argument(
        "--sigma_contact",
        type=float,
        default=0.1,
        help="InEKF contact random-walk standard deviation (default: 0.1)",
    )
    parser.add_argument(
        "--contact_gate_probability",
        type=float,
        default=None,
        help="Optional InEKF chi-square contact gate probability",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path for saving the result figure",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not open an interactive plot window",
    )
    return parser.parse_args()


def main():
    # Step 1: Parse arguments and set up paths.
    args = parse_args()
    root = repo_root_from_this_file()
    data_dir = Path(args.data_dir) if args.data_dir else root / "data"
    robot_paths = get_robot_paths(root, args.robot)

    # Step 2: Load the robot trajectory and sensor data.
    q, dq, contacts, linear_acceleration, angular_velocity = load_data(
        data_dir, robot_paths["data_prefix"]
    )
    available_samples = q.shape[1]
    number_of_samples = (
        available_samples if args.samples is None else args.samples
    )
    if number_of_samples < 1 or number_of_samples > available_samples:
        raise ValueError(
            f"--samples must be between 1 and {available_samples}"
        )

    # Step 3: Build the selected estimator.
    estimator = create_estimator(args, robot_paths)

    # Step 4: Estimate the base state, report errors, and plot the result.
    results = run_estimation(
        args.estimator,
        estimator,
        q,
        dq,
        contacts,
        linear_acceleration,
        angular_velocity,
        number_of_samples,
    )
    print_errors(args.estimator, results)
    time = np.arange(number_of_samples) * args.dt
    plot_results(
        time,
        args.estimator,
        results,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
