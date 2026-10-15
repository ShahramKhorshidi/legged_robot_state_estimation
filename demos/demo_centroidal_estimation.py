import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import yaml


# Allow this file to be run directly from any working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from centroidal_estimation.centroidal_state_ekf_force import ForceCentroidalEKF
from centroidal_estimation.centroidal_state_ekf_torque import TorqueCentroidalEKF


def load_data(path, robot_paths, estimator_name):
    """Load the trajectory and the input required by the selected estimator."""
    path = Path(path)
    data_prefix = robot_paths["data_prefix"]

    q = np.loadtxt(path / f"{data_prefix}_q.dat", delimiter="\t")
    dq = np.loadtxt(path / f"{data_prefix}_dq.dat", delimiter="\t")
    contacts = np.loadtxt(path / f"{data_prefix}_cnt.dat", delimiter="\t")

    if estimator_name == "force":
        input_file = robot_paths["force_file"]
    elif estimator_name == "torque":
        input_file = robot_paths["torque_file"]
    else:
        raise ValueError(f"Unknown estimator '{estimator_name}'")

    estimator_input = np.loadtxt(path / input_file, delimiter="\t")
    arrays = {
        "q": q,
        "dq": dq,
        "contacts": contacts,
        "estimator input": estimator_input,
    }
    for name, value in arrays.items():
        if value.ndim != 2:
            raise ValueError(f"{name} data must be a two-dimensional array")

    number_of_samples = q.shape[1]
    if any(value.shape[1] != number_of_samples for value in arrays.values()):
        raise ValueError("All input files must contain the same number of samples")
    if q.shape[0] < 8 or dq.shape[0] != q.shape[0] - 1:
        raise ValueError("q and dq do not have compatible floating-base dimensions")

    if estimator_name == "force":
        expected_rows = 3 * contacts.shape[0]
        if estimator_input.shape[0] != expected_rows:
            raise ValueError(
                f"Contact-force data must contain {expected_rows} rows"
            )
    elif estimator_input.shape[0] != dq.shape[0] - 6:
        raise ValueError(
            f"Joint-torque data must contain {dq.shape[0] - 6} rows"
        )

    return q, dq, contacts, estimator_input


def repo_root_from_this_file():
    # demos/demo_centroidal_estimation.py -> repository root is above demos/
    return Path(__file__).resolve().parents[1]


def get_robot_paths(root, robot):
    """Central mapping from a robot name to its model and trajectory files."""
    robots = {
        "go1": {
            "urdf": root / "files" / "go1_description" / "go1.urdf",
            "config": root / "files" / "go1_description" / "go1_config.yaml",
            "data_prefix": "robot",
            "force_file": "ee_force.dat",
            "torque_file": "robot_tau.dat",
            "reference_file": "cent_state.dat",
        },
        "solo12": {
            "urdf": root / "files" / "solo12_description" / "solo12.urdf",
            "config": root
            / "files"
            / "solo12_description"
            / "solo12_config.yaml",
            "data_prefix": "solo_robot",
            "force_file": "solo_robot_ee_force.dat",
            "torque_file": "solo_robot_tau.dat",
            "reference_file": None,
        },
        # Add more robots here as you extend the framework.
    }
    if robot not in robots:
        available = ", ".join(robots)
        raise ValueError(f"Unknown robot '{robot}'. Available robots: {available}")
    return robots[robot]


def get_sampling_period(config_file):
    """Read the sampling period used internally by the centroidal filters."""
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    dt = float(config.get("robot", {}).get("dt", 0.001))
    if dt <= 0.0:
        raise ValueError("Robot sampling period must be positive")
    return dt


def create_estimator(estimator_name, robot_paths):
    """Create and tune the selected centroidal-state estimator."""
    urdf = str(robot_paths["urdf"])
    config = robot_paths["config"]

    if estimator_name == "force":
        estimator = ForceCentroidalEKF(urdf, config)
        estimator.set_process_noise_cov(1e-1, 1e-1, 1e-1)
        measurement_noise = np.diag(
            [1e-2, 1e-2, 1e-2]
            + [1e-2, 1e-2, 1e-2]
            + [1e-4, 1e-4, 1e-4]
        )
        estimator.set_measurement_noise_cov(measurement_noise)
    elif estimator_name == "torque":
        estimator = TorqueCentroidalEKF(urdf, config)
        estimator.set_process_noise_cov(1e-7, 1e-5, 1e-4)
        estimator.set_measurement_noise_cov(1e-5, 1e-5, 1e-6)
    else:
        raise ValueError(f"Unknown estimator '{estimator_name}'")

    return estimator


def compute_reference_states(urdf_file, q, dq, number_of_samples):
    """Compute model-based CoM position and centroidal momentum references."""
    model = pin.buildModelFromUrdf(str(urdf_file), pin.JointModelFreeFlyer())
    data = model.createData()
    if q.shape[0] != model.nq or dq.shape[0] != model.nv:
        raise ValueError(
            "Trajectory dimensions do not match the selected robot model: "
            f"expected q/dq rows {model.nq}/{model.nv}, "
            f"received {q.shape[0]}/{dq.shape[0]}"
        )

    reference = np.zeros((number_of_samples, 9))
    for i in range(number_of_samples):
        pin.computeCentroidalMomentum(model, data, q[:, i], dq[:, i])
        reference[i, 0:3] = np.asarray(data.com[0]).reshape(3)
        reference[i, 3:6] = np.asarray(data.hg.linear).reshape(3)
        reference[i, 6:9] = np.asarray(data.hg.angular).reshape(3)

    return reference


def get_reference_states(
    data_dir,
    robot_paths,
    q,
    dq,
    number_of_samples,
):
    """Load a recorded reference, or compute one when none is provided."""
    reference_file = robot_paths.get("reference_file")
    if reference_file is None:
        return compute_reference_states(
            robot_paths["urdf"], q, dq, number_of_samples
        )

    reference = np.loadtxt(Path(data_dir) / reference_file, delimiter="\t")
    if reference.ndim != 2 or reference.shape[0] != 9:
        raise ValueError("Centroidal reference data must contain 9 rows")
    if reference.shape[1] < number_of_samples:
        raise ValueError(
            "Centroidal reference data contains fewer samples than the trajectory"
        )
    return reference[:, :number_of_samples].T.copy()


def run_estimation(
    estimator_name,
    estimator,
    q,
    dq,
    contacts,
    estimator_input,
    reference,
    integration_method,
):
    """Run one centroidal-state estimator over the loaded trajectory."""
    number_of_samples = reference.shape[0]
    estimated = np.zeros((number_of_samples, 9))

    estimator.set_mu_post(reference[0, :].copy())
    estimated[0, :] = np.concatenate(estimator.get_filter_output())

    for i in range(1, number_of_samples):
        if estimator_name == "force":
            estimator.update_filter(
                q[:, i],
                dq[:, i],
                contacts[:, i],
                estimator_input[:, i],
                integration_method=integration_method,
            )
        elif estimator_name == "torque":
            estimator.update_filter(
                q[:, i],
                dq[:, i],
                contacts[:, i],
                estimator_input[:, i],
            )
        else:
            raise ValueError(f"Unknown estimator '{estimator_name}'")

        estimated[i, :] = np.concatenate(estimator.get_filter_output())

    return estimated


def print_errors(estimator_name, reference, estimated):
    error = estimated - reference
    com_rmse = np.sqrt(np.mean(error[:, 0:3] ** 2))
    linear_momentum_rmse = np.sqrt(np.mean(error[:, 3:6] ** 2))
    angular_momentum_rmse = np.sqrt(np.mean(error[:, 6:9] ** 2))

    print(f"{estimator_name.upper()} results over {reference.shape[0]} samples")
    print(f"CoM position RMSE:      {com_rmse:.8f} m")
    print(f"Linear momentum RMSE:   {linear_momentum_rmse:.8f} kg m/s")
    print(f"Angular momentum RMSE:  {angular_momentum_rmse:.8f} kg m^2/s")


def plot_results(
    time,
    estimator_name,
    reference,
    estimated,
    save_path=None,
    show=True,
):
    """Plot the reference and estimated centroidal states."""
    quantities = (
        (slice(0, 3), "CoM position (m)"),
        (slice(3, 6), "Linear momentum (kg m/s)"),
        (slice(6, 9), "Angular momentum (kg m^2/s)"),
    )
    axes_names = ("x", "y", "z")
    figure, axes = plt.subplots(3, 3, figsize=(14, 9), sharex=True)

    for row, (state_slice, label) in enumerate(quantities):
        reference_state = reference[:, state_slice]
        estimated_state = estimated[:, state_slice]
        for column, axis_name in enumerate(axes_names):
            axis = axes[row, column]
            axis.plot(
                time,
                reference_state[:, column],
                "b",
                linewidth=0.9,
                label="Reference",
            )
            axis.plot(
                time,
                estimated_state[:, column],
                "r--",
                linewidth=0.9,
                label=estimator_name.upper(),
            )
            axis.set_title(f"{label}: {axis_name}")
            axis.grid(True)

    axes[0, 0].legend(loc="best")
    for axis in axes[-1, :]:
        axis.set_xlabel("Time (s)")
    figure.suptitle(
        f"Centroidal-state estimation with {estimator_name.upper()} inputs"
    )
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
        description="Run a legged-robot centroidal-state estimator"
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="solo12",
        help="Robot name registered in get_robot_paths (default: solo12)",
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="force",
        choices=["force", "torque"],
        help="Centroidal estimator input type (default: force)",
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
        "--integration_method",
        type=str,
        default="euler",
        choices=["euler", "rk4"],
        help="Force-estimator integration method (default: euler)",
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

    # Step 2: Load the trajectory and the selected estimator input.
    q, dq, contacts, estimator_input = load_data(
        data_dir, robot_paths, args.estimator
    )
    available_samples = q.shape[1]
    number_of_samples = (
        available_samples if args.samples is None else args.samples
    )
    if number_of_samples < 1 or number_of_samples > available_samples:
        raise ValueError(
            f"--samples must be between 1 and {available_samples}"
        )

    # Step 3: Build the estimator and prepare the reference trajectory.
    estimator = create_estimator(args.estimator, robot_paths)
    reference = get_reference_states(
        data_dir,
        robot_paths,
        q,
        dq,
        number_of_samples,
    )

    # Step 4: Estimate the centroidal states, report errors, and plot results.
    estimated = run_estimation(
        args.estimator,
        estimator,
        q,
        dq,
        contacts,
        estimator_input,
        reference,
        args.integration_method,
    )
    print_errors(args.estimator, reference, estimated)

    dt = get_sampling_period(robot_paths["config"])
    time = np.arange(number_of_samples) * dt
    plot_results(
        time,
        args.estimator,
        reference,
        estimated,
        save_path=args.save,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
