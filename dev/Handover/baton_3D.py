# Import the library.
import intprim
import numpy as np
import trajPlotting
from pathlib import Path
import matplotlib.pyplot as plt
import re
import copy
import sklearn.metrics



def discover_handover_pairs(data_dir):
    """Discover paired baton/taker CSVs.

    Expects filenames like:
        baton_1.csv, taker_1.csv
        baton_001.csv, taker_001.csv

    Args:
        data_dir: Directory containing the CSV files.

    Returns:
        List of (pair_id, baton_path, taker_path), sorted by numeric ID when possible.
    """
    data_dir = Path(data_dir)
    baton_re = re.compile(r"^baton_(?P<id>\d+)\.csv$")
    taker_re = re.compile(r"^taker_(?P<id>\d+)\.csv$")

    baton_by_id = {}
    taker_by_id = {}
    for path in data_dir.glob("*.csv"):
        m = baton_re.match(path.name)
        if m:
            baton_by_id[m.group("id")] = path
            continue
        m = taker_re.match(path.name)
        if m:
            taker_by_id[m.group("id")] = path

    common_ids = sorted(set(baton_by_id) & set(taker_by_id), key=lambda s: int(s))
    return [(pair_id, baton_by_id[pair_id], taker_by_id[pair_id]) for pair_id in common_ids]


def load_handover_pair(baton_path, taker_path, delimiter=",", dtype=None, skiprows=1):
    """Load a baton/taker trajectory pair from CSV."""
    if dtype is None:
        dtype = intprim.constants.DTYPE
    baton = np.loadtxt(str(baton_path), delimiter=delimiter, dtype=dtype, skiprows=skiprows)
    taker = np.loadtxt(str(taker_path), delimiter=delimiter, dtype=dtype, skiprows=skiprows)
    return baton, taker

def normalize_to_giver_start(trajectory):
    """Normalize all DOFs relative to the giver's (baton/robot) starting position.

    Role mapping:
      DOFs 0-2 : baton / giver  = ROBOT (controlled agent, predicted by BIP)
      DOFs 3-5 : taker          = HUMAN (observed agent, drives EnKF)

    All 6 DOFs are shifted so the giver starts at the origin. This makes the
    BIP invariant to where in the workspace the handover occurs, while
    preserving the critical spatial relationship between the two agents
    (i.e. where the human stands relative to the robot at the start).

    Normalizing only the taker independently would destroy this relationship;
    normalizing only the giver is insufficient when the robot starts at a
    different position than the training humans.

    Returns:
        normalized: all 6 DOFs shifted so baton DOFs start at zero
        anchor:     shape (3, 1) — the original giver start in world coordinates,
                    needed to recover world-frame coordinates after inference.
    """
    anchor = trajectory[0:3, 0:1].copy()  # giver (baton) start, shape (3, 1)
    normalized = trajectory.copy()
    normalized[0:3, :] -= anchor
    normalized[3:6, :] -= anchor
    return normalized, anchor


def denormalize_interaction(trajectory, anchor):
    """Add the giver anchor back to all 6 DOFs to recover world coordinates."""
    result = trajectory.copy()
    result[0:3, :] += anchor
    result[3:6, :] += anchor
    return result


def apply_endpoint_offset(trajectory, taker_offset):
    """Apply a phase-scaled spatial offset to the taker's (DOFs 3-5) trajectory.

    The offset ramps linearly from 0 at the start to `taker_offset` at the end,
    so the taker's starting position is unchanged but their handover location
    is displaced.  This tests whether the BIP can predict a different meeting
    point purely from observing the taker's deviated approach.

    Args:
        trajectory:   shape (6, T) trajectory, already normalized to giver start.
        taker_offset: array-like of length 3, (dx, dy, dz) offset applied at the
                      trajectory endpoint (phase=1).

    Returns:
        Modified trajectory with taker DOFs (3-5) offset.
    """
    T = trajectory.shape[1]
    scale = np.linspace(0.0, 1.0, T)  # 0 at start, 1 at end
    offset = np.array(taker_offset, dtype=trajectory.dtype).reshape(3, 1)
    result = trajectory.copy()
    result[3:6, :] += offset * scale
    return result


def stretch_trajectory(trajectory, factor):
    """Resample a trajectory to be `factor` times longer using linear interpolation.

    The motion profile shape is preserved — the interaction simply plays out
    more slowly.  This is used to test temporal robustness: a BIP trained on
    normal-speed demonstrations should still track a slower execution.

    Args:
        trajectory: (D, T) array.
        factor:     Stretch factor > 1.  2.0 = twice as long.

    Returns:
        (D, round(T * factor)) array.
    """
    D, T = trajectory.shape
    T_new = round(T * factor)
    x_old = np.linspace(0.0, 1.0, T)
    x_new = np.linspace(0.0, 1.0, T_new)
    return np.vstack([np.interp(x_new, x_old, trajectory[d]) for d in range(D)])


def process_approach(baton, taker, cutoff):
    start = np.average(baton[:cutoff, 2])
    for i in range(cutoff, baton.shape[0]):
        if baton[i, 2] > start + 0.01:
            # print("Cutoff at index %d, z: %f" % (i, baton[i, 2]))
            return baton[i:,:], taker[i:,:]
    print("No cutoff found, returning full trajectories")
    return baton, taker
    

def evaluate(primitive, filter, test_trajectories, observation_noise, giver_anchors=None, delay_prob=0.0, delay_ratio=0.0, time_step=1, pause_secs=0.04):
    """Run BIP inference over test trajectories.

    Args:
        giver_anchors: list of (3, 1) arrays, one per test trajectory, containing
                       the giver's (robot's) world-frame start position.  Used to
                       recover real-world coordinates for MSE and plotting.
        time_step:     Number of new observations consumed per inference call.
                       Use 1 for a smooth per-index live view, or higher (e.g. 20)
                       for faster batch evaluation.
        pause_secs:    Seconds to pause after each viewer update.  Increase to
                       slow the animation; decrease for faster playback.
    """
    if giver_anchors is None:
        giver_anchors = [None] * len(test_trajectories)

    for test_trajectory, giver_anchor in zip(test_trajectories, giver_anchors):
        test_trajectory_partial = np.array(test_trajectory, copy=True)
        test_trajectory_partial[0:3, :] = 0.0  # Zero out giver/robot DOFs (0-2) — unobserved at inference.
                                                # Only human taker DOFs (3-5) drive the EnKF.
        new_filter = copy.deepcopy(filter)

        primitive.set_filter(new_filter)

        mean_trajectory = primitive.get_mean_trajectory()
        mean_world = denormalize_interaction(mean_trajectory, giver_anchor) if giver_anchor is not None else mean_trajectory

        # Single persistent viewer — updated each timestep, not recreated.
        viewer = trajPlotting.LiveInteractionViewer(
            mean_interaction=mean_world,
            observed_agent="taker",
            title="BIP live inference",
            pause_secs=pause_secs,
        )

        mean_mse = 0.0
        phase_mae = 0.0
        mse_count = 0
        prev_observed_index = 0
        for observed_index in range(time_step, test_trajectory.shape[1], time_step):
            gen_trajectory, phase, mean, var = primitive.generate_probable_trajectory_recursive(
                test_trajectory_partial[:, prev_observed_index:observed_index],
                observation_noise,
                np.array([3, 4, 5], dtype=np.int32),
                num_samples = test_trajectory_partial.shape[1] - observed_index,
            )

            # Recover world coordinates for MSE and plotting.
            gen_trajectory_world = denormalize_interaction(gen_trajectory, giver_anchor) if giver_anchor is not None else gen_trajectory
            test_future_world = denormalize_interaction(test_trajectory[:, observed_index:], giver_anchor) if giver_anchor is not None else test_trajectory[:, observed_index:]

            mse = sklearn.metrics.mean_squared_error(test_future_world, gen_trajectory_world)
            mean_mse += mse
            mse_count += 1

            phase_mae += np.abs((float(observed_index) / test_trajectory.shape[1]) - phase)

            if(delay_prob > 0.0 and np.random.binomial(1, delay_prob) == 1):
                length = int(delay_ratio * test_trajectory.shape[1])
                # Repeat the last observation for delay_ratio times.
                delay_trajectory = np.tile(test_trajectory[:, observed_index - 1], (length, 1)).T

                gen_trajectory, phase, mean, var = primitive.generate_probable_trajectory_recursive(
                    delay_trajectory,
                    observation_noise,
                    np.array([3, 4, 5], dtype=np.int32),
                    num_samples = test_trajectory_partial.shape[1] - observed_index,
                )
                gen_trajectory_world = denormalize_interaction(gen_trajectory, giver_anchor) if giver_anchor is not None else gen_trajectory
                mse = sklearn.metrics.mean_squared_error(test_future_world, gen_trajectory_world)
                mean_mse += mse
                mse_count += 1
                phase_mae += np.abs((float(observed_index) / test_trajectory.shape[1]) - phase)

                # Plot the phase/phase velocity PDF for each time step? Want to show it for temporal non-linearity.

            plot_obs = denormalize_interaction(test_trajectory_partial[:, :observed_index], giver_anchor) if giver_anchor is not None else test_trajectory_partial[:, :observed_index]
            viewer.update(gen_trajectory_world, plot_obs)

            prev_observed_index = observed_index

        print("Mean DoF MSE: " + str(mean_mse / mse_count) + ". Phase MAE: " + str(phase_mae / mse_count))
        viewer.keep_open()

if __name__ == "__main__":
    # Set a seed for reproducibility
    np.random.seed(213413414)

    # Use a stable path so you can run from any working directory.
    data_dir = Path(__file__).resolve().parent / "trainingData"
    testdata_dir = Path(__file__).resolve().parent / "testingData"

    pairs = discover_handover_pairs(data_dir)
    if not pairs:
        raise IOError(
            "No paired CSVs found in {}. Expected baton_#.csv and taker_#.csv.".format(data_dir)
        )


    training_trajectories = []
    for i in range(len(pairs)):
        pair_id, baton_path, taker_path = pairs[i]
        baton, taker = load_handover_pair(baton_path, taker_path)
        baton_approach, taker_approach = process_approach(baton[:400, :3], taker[:400, :3], 50)
        raw_traj = np.concatenate((baton_approach.T, taker_approach.T), axis=0)
        # Normalize all 6 DOFs relative to the giver's (baton) start position.
        # This makes the BIP invariant to workspace location while preserving
        # the spatial relationship between giver and taker.
        norm_traj, _ = normalize_to_giver_start(raw_traj)
        training_trajectories.append(norm_traj)
        
        # Plotting individual paired trajectories - Don't do with big dataset, useful for sanity check
        if i < 2:
            fig, ax, ani = trajPlotting.plot_animate(
            baton_approach,
            taker_approach,
            interval=20,
            show=True,
            labels=("baton", "taker"),
            )

    
    # Define DOFs: Controlled (giver - baton) and Observed (taker)
    dof_names = np.array(["X (Controlled)", "Y (Controlled)", "Z (Controlled)", "X (Observed)", "Y (Observed)", "Z (Observed)"])

    # Basis space selection
    selection = intprim.basis.Selection(dof_names)

    for trajectory in training_trajectories:
        selection.add_demonstration(trajectory)

    # aic, bic = selection.get_information_criteria(np.array([0, 1], dtype = np.int32))
    # selection.get_best_model(aic, bic)

    basis_model_gaussian = intprim.basis.GaussianModel(5, 0.19, dof_names)

    # basis_model_gaussian.plot()


    # Initialize a BIP instance.
    primitive = intprim.BayesianInteractionPrimitive(basis_model_gaussian)

    # Train the model.
    for trajectory in training_trajectories:
        primitive.add_demonstration(trajectory)

    # Plot the distribution of the trained model.
    mean, upper_bound, lower_bound = primitive.get_probability_distribution()
    # intprim.util.visualization.plot_distribution(dof_names, mean, upper_bound, lower_bound)
    # plt.show()  # Block until the user closes the plot window(s)

    observation_noise = np.diag(selection.get_model_mse(basis_model_gaussian, np.array([0, 1, 2, 3, 4, 5])))

    # print(observation_noise)

    # Compute the phase mean and phase velocities from the demonstrations.
    phase_velocity_mean, phase_velocity_var = intprim.examples.get_phase_stats(training_trajectories)
    # print("Phase velocity mean: %f, variance: %f" % (phase_velocity_mean, phase_velocity_var))

    # Define a filter to use. Here we use an ensemble Kalman filter
    filter = intprim.filter.spatiotemporal.EnsembleKalmanFilter(
    basis_model = basis_model_gaussian,
    initial_phase_mean = [0.0, phase_velocity_mean],
    initial_phase_var = [1e-4, phase_velocity_var],
    proc_var = 1e-8,
    initial_ensemble = primitive.basis_weights)




    # Create testing data
    dtype = intprim.constants.DTYPE

    test_baton_path = testdata_dir / "baton_058.csv"
    test_taker_path = testdata_dir / "taker_058.csv"
    test_baton = np.loadtxt(str(test_baton_path), delimiter=",", dtype=dtype, skiprows=1)
    test_taker = np.loadtxt(str(test_taker_path), delimiter=",", dtype=dtype, skiprows=1)
    # test_taker[:,2] += 0.10  # Shift the taker X position by +5 cm to test spatial robustness.
    test_baton_approach, test_taker_approach = process_approach(test_baton[:400, :3], test_taker[:400, :3], 50)

    raw_test_trajectory = np.concatenate((test_baton_approach.T, test_taker_approach.T), axis=0)
    # Normalize all 6 DOFs relative to the giver's (baton) start position.
    # At robot runtime, anchor = robot EEF world position at t=0.
    test_trajectory, giver_anchor = normalize_to_giver_start(raw_test_trajectory)
    print("Giver anchor (world-frame start): ", giver_anchor.T)

    # Apply a phase-scaled offset to the taker (DOFs 3-5) to simulate the human
    # reaching for a different handover location.  The offset is zero at t=0
    # (start position unchanged) and reaches full value at the end of the
    # trajectory.  Adjust dx/dy/dz to explore spatial robustness.
    # taker_endpoint_offset = [0.0, 0.0, -0.10]  # metres: [dx, dy, dz]
    # test_trajectory = apply_endpoint_offset(test_trajectory, taker_endpoint_offset)
    # print("Taker endpoint offset (giver-relative frame): ", taker_endpoint_offset)

    # Temporal robustness test: stretch the trajectory to 2x duration while
    # keeping the same motion profile.  The BIP was trained on normal-speed
    # demos; this checks whether the phase estimator can track a slower execution.
    RECORDING_HZ = 120
    STRETCH_FACTOR = 1.0
    test_trajectory = stretch_trajectory(test_trajectory, STRETCH_FACTOR)
    print(f"Trajectory stretched {STRETCH_FACTOR}x: {test_trajectory.shape[1]} timesteps "
          f"({test_trajectory.shape[1] / RECORDING_HZ:.2f} s at {RECORDING_HZ} Hz)")



    # fig, ax, ani = trajPlotting.plot_animate(
    #     test_baton_approach,
    #     test_taker_approach,
    #     interval=20,
    #     show=True,
    #     labels=("baton", "taker"),
    # )

    # trajPlotting.plot_animate(test_baton_approach, test_taker_approach, interval=20, show=True, labels=("baton", "taker"))

    # Evaluate the trajectories: step=1 gives a per-index live view.
    # Data was recorded at 120 Hz, so each index = 1/120 s ≈ 0.00833 s.
    # evaluate(primitive, filter, [test_trajectory], observation_noise,
            #  giver_anchors=[giver_anchor], time_step=1, pause_secs=1.0 / RECORDING_HZ)

    # viewer.keep_open() is called inside evaluate() after each trajectory.

    
    
    

    # Decompose the handwriting trajectories to a basis space with 8 uniformly distributed Gaussian functions and a variance of 0.1.
    # basis_model = intprim.basis.GaussianModel(8, 0.1, dof_names)

    # Initialize a BIP instance.
    # primitive = intprim.BayesianInteractionPrimitive(basis_model)


