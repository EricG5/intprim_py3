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

def process_approach(baton, taker, cutoff):
    start = np.average(baton[:cutoff, 2])
    for i in range(cutoff, baton.shape[0]):
        if baton[i, 2] > start + 0.01:
            # print("Cutoff at index %d, z: %f" % (i, baton[i, 2]))
            return baton[i:,:], taker[i:,:]
    print("No cutoff found, returning full trajectories")
    return baton, taker
    

def evaluate(primitive, filter, test_trajectories, observation_noise, delay_prob = 0.0, delay_ratio = 0.0):
    for test_trajectory in test_trajectories:
        test_trajectory_partial = np.array(test_trajectory, copy = True)
        test_trajectory_partial[0:3, :] = 0.0 # Only observe the taker (observed) DOFs
        new_filter = copy.deepcopy(filter)

        primitive.set_filter(new_filter)

        # all_gen_trajectories = []
        # all_test_trajectories = []
        mean_trajectory = primitive.get_mean_trajectory()

        mean_mse = 0.0
        phase_mae = 0.0
        mse_count = 0
        prev_observed_index = 0
        time_step = 20
        for observed_index in range(time_step, test_trajectory.shape[1], time_step):
            gen_trajectory, phase, mean, var = primitive.generate_probable_trajectory_recursive(
                test_trajectory_partial[:, prev_observed_index:observed_index],
                observation_noise,
                np.array([3, 4, 5], dtype=np.int32),
                num_samples = test_trajectory_partial.shape[1] - observed_index,
            )

            mse = sklearn.metrics.mean_squared_error(test_trajectory[:, observed_index:], gen_trajectory)
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
                mse = sklearn.metrics.mean_squared_error(test_trajectory[:, observed_index:], gen_trajectory)
                mean_mse += mse
                mse_count += 1
                phase_mae += np.abs((float(observed_index) / test_trajectory.shape[1]) - phase)

                # Plot the phase/phase velocity PDF for each time step? Want to show it for temporal non-linearity.

            trajPlotting.plot_partial_trajectory_3d(
                gen_trajectory,
                test_trajectory_partial[:, :observed_index],
                mean_interaction=mean_trajectory,
                show=True,
                observed_agent="taker",
                show_mean=True,
            )
            # all_gen_trajectories.append(gen_trajectory)
            # all_test_trajectories.append(test_trajectory[:, :observed_index])

            prev_observed_index = observed_index

        print("Mean DoF MSE: " + str(mean_mse / mse_count) + ". Phase MAE: " + str(phase_mae / mse_count))

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
        training_trajectories.append(
            np.concatenate((baton_approach.T, taker_approach.T), axis=0)
        )
        
        # Plotting individual paired trajectories - Don't do with big dataset, useful for sanity check
        # if i < 1:
        #     fig, ax, ani = trajPlotting.plot_animate(
        #     baton_approach,
        #     taker_approach,
        #     interval=20,
        #     show=True,
        #     labels=("baton", "taker"),
        #     )

    
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
    test_baton_approach, test_taker_approach = process_approach(test_baton[:400, :3], test_taker[:400, :3], 50)
    # for i in range(test_baton_approach.shape[0]):
    #     test_taker_approach[i,2] -= 0.05 # Add a small offset to test spatial handling
    #     test_taker_approach[i,0] += 0.05
    #     test_taker_approach[i,1] += 0.05

    test_trajectory = np.concatenate((test_baton_approach.T, test_taker_approach.T), axis=0)

    # fig, ax, ani = trajPlotting.plot_animate(
    #     test_baton_approach,
    #     test_taker_approach,
    #     interval=20,
    #     show=True,
    #     labels=("baton", "taker"),
    # )

    # trajPlotting.plot_animate(test_baton_approach, test_taker_approach, interval=20, show=True, labels=("baton", "taker"))

    # Evaluate the trajectories. ***TEST REQUIRED***
    evaluate(primitive, filter, [test_trajectory], observation_noise)

    # Hold for plots
    plt.show(block=True)

    
    
    

    # Decompose the handwriting trajectories to a basis space with 8 uniformly distributed Gaussian functions and a variance of 0.1.
    # basis_model = intprim.basis.GaussianModel(8, 0.1, dof_names)

    # Initialize a BIP instance.
    # primitive = intprim.BayesianInteractionPrimitive(basis_model)


