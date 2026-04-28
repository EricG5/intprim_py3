import intprim
import numpy as np
import trajPlotting
from pathlib import Path
import matplotlib.pyplot as plt
import re
import copy
import sklearn.metrics
from scipy.spatial.transform import Rotation
import baton_3D  
import baton_6D
from pathlib import Path




if __name__ == "__main__":
    # Set a seed for reproducibility
    np.random.seed(213413414)

    degree = 5
    scale = 0.16
    observation_noise = np.diag([0.000004, 0.000002, 0.000007, 0.002026, 0.001604, 0.000505, 0.000029, 0.000009, 0.000016, 0.002554, 0.001071, 0.000610])
    phase_vel_mean = 0.006534
    phase_vel_var = 0.000002
    dof_names = np.array(["X (Controlled)", "Y (Controlled)", "Z (Controlled)", "RX (Controlled)", "RY (Controlled)", "RZ (Controlled)", "X (Observed)", "Y (Observed)", "Z (Observed)", "RX (Observed)", "RY (Observed)", "RZ (Observed)"])
    basis_model_gaussian = intprim.basis.GaussianModel(degree, scale, dof_names)
    primitive = intprim.BayesianInteractionPrimitive(basis_model_gaussian)

    model_name = "baton_6d_model_world.bip"
    model_path = Path(__file__).parent / "models" / model_name
    primitive.import_data(model_path)

    mean, upper_bound, lower_bound = primitive.get_probability_distribution()
    filter = intprim.filter.spatiotemporal.EnsembleKalmanFilter(
        basis_model = basis_model_gaussian,
        initial_phase_mean = [0.0, phase_vel_mean],
        initial_phase_var = [1e-4, phase_vel_var],
        proc_var = 1e-8,
        initial_ensemble = primitive.basis_weights)
    
    primitive.set_filter(copy.deepcopy(filter))
    observed_dof_indices = np.array([6, 7, 8, 9, 10, 11], dtype=np.int32)  # Indices of observed DOFs in the state vector

    test_trajectory_path = Path(__file__).parent / "test_trajectory.csv"
    test_trajectory = np.loadtxt(test_trajectory_path, delimiter=",", skiprows=1).T

    RECORDING_HZ = 120

    baton_6D.evaluate_6d(primitive, filter, [test_trajectory], observation_noise,
                # TEMP: world-frame -> no denormalization offsets
                taker_anchors=[None], time_step=1, pause_secs=1.0 / RECORDING_HZ,
                observe_controlled_start=False,
                training_handover_stats=None)

