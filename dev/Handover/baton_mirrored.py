# Import the library.
import intprim
import numpy as np
import trajPlotting
from pathlib import Path
import json
import datetime
import matplotlib.pyplot as plt
import re
import copy
import sklearn.metrics
from scipy.spatial.transform import Rotation
import baton_3D  
import baton_6D


if __name__ == "__main__":
    
    # Set to True to run training and evaluation; False to just visualize trajectories
    train_flag = True

    # Left-handed simulation: mirror across the x–z plane (invert world-space Y).
    # If your tracking-space origin is not centered, keep MIRROR_Y0=None to
    # mirror about the per-trajectory midline between both agents at t=0.
    MIRROR_Y0 = None

    # Use a stable path so you can run from any working directory.
    data_dir = Path(__file__).resolve().parent / "trainingData"
    testdata_dir = Path(__file__).resolve().parent / "testingData"
    model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)



    pairs = baton_3D.discover_handover_pairs(data_dir)
    if not pairs:
        raise IOError(
            "No paired CSVs found in {}. Expected baton_#.csv and taker_#.csv.".format(data_dir)
        )

    training_trajectories = []
    mean_baton_start = np.zeros(3)
    full_traj_start = 0
    delay = 0
    for i in range(len(pairs)):
        pair_id, baton_path, taker_path = pairs[i]
        baton, taker = baton_3D.load_handover_pair(baton_path, taker_path)
        baton_approach, taker_approach = baton_3D.process_approach(baton[:400, :], taker[:400, :], 50, delay)
        # Skip pre-process of approach to include full behaviour before approach
        # baton_approach, taker_approach = baton[full_traj_start:400, :], taker[full_traj_start:400, :]

        # Convert quaternions (cols 3:7, [x,y,z,w]) to rotation vectors (3D axis-angle).
        # This avoids double-cover and unit-norm issues that raw quaternions cause in BIP.
        baton_pos = baton_approach[:, :3]                                      # (T, 3)
        baton_rot = Rotation.from_quat(baton_6D._continuous_quats(baton_approach[:, 3:7])).as_rotvec()     # (T, 3)
        taker_pos = taker_approach[:, :3]
        taker_rot = Rotation.from_quat(baton_6D._continuous_quats(taker_approach[:, 3:7])).as_rotvec()

        # Combine into (T, 6) per agent: [x, y, z, rx, ry, rz]
        baton_6d = np.hstack((baton_pos, baton_rot))  # (T, 6)
        taker_6d = np.hstack((taker_pos, taker_rot))  # (T, 6)

        raw_traj = np.concatenate((baton_6d.T, taker_6d.T), axis=0)  # (12, T)
        raw_traj = baton_6D.mirror_6d_across_xz_plane(raw_traj, y0=MIRROR_Y0)

        mean_baton_start[0] += raw_traj[0, 0]
        mean_baton_start[1] += raw_traj[1, 0]
        mean_baton_start[2] += raw_traj[2, 0]

        # --- TEMP: world-frame training (no anchors/offsets) -----------------
        training_trajectories.append(raw_traj)

        # --- Previous behavior (taker-relative normalization) ----------------
        # Normalize position DOFs relative to taker start; leave rotation vectors unchanged.
        # anchor = raw_traj[6:9, 0:1].copy()     # taker position start (3, 1)
        # norm_traj = raw_traj.copy()
        # norm_traj[0:3, :] -= anchor            # giver position
        # norm_traj[6:9, :] -= anchor            # taker position
        # # rotation vector DOFs (3:6, 9:12) are already origin-independent
        # training_trajectories.append(norm_traj)
        
        # Plotting individual paired trajectories 
        if (not train_flag and i <2):
            fig, ax, ani = trajPlotting.plot_animate(
            raw_traj[0:6, :].T,
            raw_traj[6:12, :].T,
            interval=20,
            show=True,
            labels=("baton", "taker"),
            )

    mean_baton_start /= len(pairs)
    # print("Mean baton start position across demos (world-frame): ", mean_baton_start)
    # -------------------------------------------------------------------------
    # Final-pose offset analysis across training demonstrations.
    #
    # At the end of each demo the baton and taker meet at the handover point.
    # The difference (controlled - observed) at the final timestep tells us
    # the typical spatial and rotational offset that constitutes a good handover.
    # Mean and std across demos give a quantitative target for the controlled
    # agent's goal pose relative to the observed agent.
    # -------------------------------------------------------------------------
    final_pos_diffs  = []   # baton_pos[-1] - taker_pos[-1]  (3,)
    final_rot_diffs  = []   # baton_rotvec[-1] - taker_rotvec[-1]  (3,)
    final_ctrl_pos   = []   # baton_pos[-1]  — absolute controlled endpoint
    final_ctrl_rot   = []   # baton_rotvec[-1]
    for traj in training_trajectories:
        final_pos_diffs.append(traj[0:3, -1] - traj[6:9,  -1])
        final_rot_diffs.append(traj[3:6, -1] - traj[9:12, -1])
        final_ctrl_pos.append(traj[0:3, -1])
        final_ctrl_rot.append(traj[3:6, -1])

    final_pos_diffs = np.array(final_pos_diffs)   # (N, 3)
    final_rot_diffs = np.array(final_rot_diffs)   # (N, 3)
    final_ctrl_pos  = np.array(final_ctrl_pos)    # (N, 3)
    final_ctrl_rot  = np.array(final_ctrl_rot)    # (N, 3)

    pos_mean = final_pos_diffs.mean(axis=0)
    pos_std  = final_pos_diffs.std(axis=0)
    rot_mean = final_rot_diffs.mean(axis=0)
    rot_std  = final_rot_diffs.std(axis=0)

    # print("\n--- Training demo final-pose offset (controlled - observed) ---")
    # print(f"  Position  [x, y, z]  mean: {pos_mean}  std: {pos_std}  (metres)")
    # print(f"  Rotation [rx,ry,rz]  mean: {rot_mean}  std: {rot_std}  (rad, rotvec)")
    # print("---------------------------------------------------------------\n")

    handover_stats = dict(
        pos_mean=pos_mean, pos_std=pos_std,
        rot_mean=rot_mean, rot_std=rot_std,
        ctrl_pos_mean=final_ctrl_pos.mean(axis=0),
        ctrl_pos_std=final_ctrl_pos.std(axis=0),
        ctrl_rot_mean=final_ctrl_rot.mean(axis=0),
        ctrl_rot_std=final_ctrl_rot.std(axis=0),
    )

    # Analysis output directory (used for optional JSON summary + XML stats).
    analysis_dir = Path(__file__).resolve().parent / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Set to True if you want a one-shot JSON dump of training-set summary stats.
    # This is independent of the per-timestep XML stat collection.
    EXPORT_TRAINING_SUMMARY_JSON = False

    def _jsonify(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value

    if EXPORT_TRAINING_SUMMARY_JSON:
        training_summary = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "num_pairs": int(len(pairs)),
            "mean_baton_start_world": _jsonify(mean_baton_start),
            "handover_stats": {k: _jsonify(v) for k, v in handover_stats.items()},
        }
        summary_path = analysis_dir / "baton_6d_training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(training_summary, f, indent=2, sort_keys=True)

    # Define DOFs: 12 total — 6 per agent (pos + rotvec)
    dof_names = np.array([
        "X (Controlled)", "Y (Controlled)", "Z (Controlled)",
        "RX (Controlled)", "RY (Controlled)", "RZ (Controlled)",
        "X (Observed)", "Y (Observed)", "Z (Observed)",
        "RX (Observed)", "RY (Observed)", "RZ (Observed)",
    ])

    if (train_flag):
        # Basis space selection
        selection = intprim.basis.Selection(dof_names)

        for trajectory in training_trajectories:
            selection.add_demonstration(trajectory)

        # aic, bic = selection.get_information_criteria(np.array([0, 1], dtype = np.int32))
        # selection.get_best_model(aic, bic)

        basis_model_gaussian = intprim.basis.GaussianModel(5, 0.13, dof_names)
        # basis_model_sigmoidal = intprim.basis.SigmoidalModel(5, 0.19, dof_names)

        # basis_model_gaussian.plot()

        # Initialize a BIP instance.
        primitive = intprim.BayesianInteractionPrimitive(basis_model_gaussian)
        # primitive = intprim.BayesianInteractionPrimitive(basis_model_sigmoidal)

        # Train the model.
        for trajectory in training_trajectories:
            primitive.add_demonstration(trajectory)

        # Plot the distribution of the trained model.
        mean, upper_bound, lower_bound = primitive.get_probability_distribution()
        # print("Lower bound (DOF 0): ", lower_bound[0])
        # intprim.util.visualization.plot_distribution(dof_names, mean, upper_bound, lower_bound)
        # plt.show()  

        # Export trained model
        # model_path = model_dir / "mirrored_baton_6d_model_world.bip"
        # primitive.export_data(str(model_path))

        observation_noise = np.diag(selection.get_model_mse(basis_model_gaussian, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])))
        # observation_noise = np.diag(selection.get_model_mse(basis_model_sigmoidal, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])))
        
        # for i in range(len(observation_noise)):
        #     print("Observation noise for DOF %d (%s): %f" % (i, dof_names[i], observation_noise[i,i]))
        # print(observation_noise)

        # Compute the phase mean and phase velocities from the demonstrations.
        phase_velocity_mean, phase_velocity_var = intprim.examples.get_phase_stats(training_trajectories)
        # print("Phase velocity mean: %f, variance: %f" % (phase_velocity_mean, phase_velocity_var))

        #Define a filter to use. Here we use an ensemble Kalman filter
        filter = intprim.filter.spatiotemporal.EnsembleKalmanFilter(
        basis_model = basis_model_gaussian,
        initial_phase_mean = [0.0, phase_velocity_mean],
        initial_phase_var = [1e-4, phase_velocity_var],
        proc_var = 1e-8,
        initial_ensemble = primitive.basis_weights)

        # Create testing data
        dtype = intprim.constants.DTYPE

        test_baton_path = testdata_dir / "baton_060.csv"
        test_taker_path = testdata_dir / "taker_060.csv"
        test_baton = np.loadtxt(str(test_baton_path), delimiter=",", dtype=dtype, skiprows=1)
        test_taker = np.loadtxt(str(test_taker_path), delimiter=",", dtype=dtype, skiprows=1)
        # test_taker[:,0] += 0.10  # Shift the taker X position by +5 cm to test spatial robustness.
        test_baton_approach, test_taker_approach = baton_3D.process_approach(test_baton[:400, :], test_taker[:400, :], 50, delay)
        # test_baton_approach, test_taker_approach = test_baton[full_traj_start:400, :], test_taker[full_traj_start:400, :]

        # Convert quaternions to rotation vectors (same as training data).
        test_baton_pos = test_baton_approach[:, :3]
        test_baton_rot = Rotation.from_quat(baton_6D._continuous_quats(test_baton_approach[:, 3:7])).as_rotvec()
        test_taker_pos = test_taker_approach[:, :3]
        test_taker_rot = Rotation.from_quat(baton_6D._continuous_quats(test_taker_approach[:, 3:7])).as_rotvec()

        test_baton_6d = np.hstack((test_baton_pos, test_baton_rot))  # (T, 6)
        test_taker_6d = np.hstack((test_taker_pos, test_taker_rot))  # (T, 6)

        raw_test_trajectory = np.concatenate((test_baton_6d.T, test_taker_6d.T), axis=0)  # (12, T)
        raw_test_trajectory = baton_6D.mirror_6d_across_xz_plane(raw_test_trajectory, y0=MIRROR_Y0)

        # --- TEMP: world-frame testing (no anchors/offsets) ----------------------
        test_trajectory = raw_test_trajectory.copy()

        # --- Previous behavior (taker-relative normalization) --------------------
        # Normalize position DOFs relative to taker start; leave rotvec unchanged.
        # taker_anchor = raw_test_trajectory[6:9, 0:1].copy()  # (3, 1)
        # test_trajectory = raw_test_trajectory.copy()
        # test_trajectory[0:3, :] -= taker_anchor   # giver position
        # test_trajectory[6:9, :] -= taker_anchor   # taker position
        # print("Taker anchor (world-frame start): ", taker_anchor.T)

        
        # Test to push observed outside known demo set (simulate different interpersonal spacing)
        # test_trajectory[6,:] += 0.20
        # test_trajectory[7,:] += 0.10

        # Apply a phase-scaled offset to the taker (DOFs 3-5) to simulate the human
        # reaching for a different handover location.  The offset is zero at t=0
        # (start position unchanged) and reaches full value at the end of the
        # trajectory.  Adjust dx/dy/dz to explore spatial robustness.
        # taker_endpoint_offset = [0.0, 0.0, -0.10]  # metres: [dx, dy, dz]
        # test_trajectory = apply_endpoint_offset(test_trajectory, taker_endpoint_offset)
        # print("Taker endpoint offset (taker-relative frame): ", taker_endpoint_offset)

        # Temporal robustness test: stretch the trajectory to 2x duration while
        # keeping the same motion profile.  The BIP was trained on normal-speed
        # demos; this checks whether the phase estimator can track a slower execution.
        RECORDING_HZ = 120
        STRETCH_FACTOR = 1.0
        test_trajectory = baton_3D.stretch_trajectory(test_trajectory, STRETCH_FACTOR)
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

        # -------------------------------------------------------------------------
        # Baton start-position offset (normalized / taker-relative coords).
        #
        # Simulates a deployment scenario where the robot (controlled agent) starts
        # at a different position than in the training demonstrations.  In training,
        # the taker starts at [0,0,0] (normalized).  Here we shift the ground-truth
        # baton trajectory by an arbitrary offset to test how well the BIP copes.
        #
        # Because the robot's starting pose is KNOWN at deployment time, we set
        # observe_controlled_start=True so the filter observes the baton position
        # DOFs [0,1,2] at t=0 only, anchoring it to the actual start before
        # switching to taker-only observations for the rest of the interaction.
        #
        # Set to None to reproduce normal (in-distribution) behaviour.
        # -------------------------------------------------------------------------
        baton_start_offset = None  # e.g. np.array([0.05, 0.0, -0.02]) metres
        # baton_start_offset = np.array([0.00, 0.1, 0.0])
        
        if baton_start_offset is not None:
            baton_start_offset = np.asarray(baton_start_offset, dtype=test_trajectory.dtype)
            T = test_trajectory.shape[1]
            # Linear blend: full offset at t=0, zero offset at t=T-1.
            # The baton starts displaced but converges back to the original
            # (training-consistent) handover location by the end of the motion.
            blend = np.linspace(1.0, 0.0, T, dtype=test_trajectory.dtype)  # (T,)
            test_trajectory[0:3, :] += baton_start_offset[:, np.newaxis] * blend[np.newaxis, :]
            print("Baton ground-truth start offset (normalised): ", baton_start_offset,
                "— blended to zero by end of trajectory")
            
        # np.savetxt("mirrored_test_trajectory.csv", raw_test_trajectory.T, delimiter=",", header="X_ctrl,Y_ctrl,Z_ctrl,RX_ctrl,RY_ctrl,RZ_ctrl,X_obs,Y_obs,Z_obs,RX_obs,RY_obs,RZ_obs", comments="")
        
        # Delayed start test: add stationary data points to start of trajectory to simulate a late start in the observed agent's motion.  This checks whether the filter can handle an initial period of static observation before the interaction begins.
        # delayed_start_test_trajectory = np.zeros((12, test_trajectory.shape[1] + 30), dtype=test_trajectory.dtype)
        # delayed_start_test_trajectory[:, 30:test_trajectory.shape[1]+30] = test_trajectory
        # delayed_start_test_trajectory[:, :30] = test_trajectory[:, 0:1]  # Prepend 30 frames of the initial pose to simulate a delayed start with static initial observation.

        # Evaluate the trajectories: step=1 gives a per-index live view.
        # Data was recorded at 120 Hz, so each index = 1/120 s ≈ 0.00833 s.
        baton_6D.evaluate_6d(primitive, filter, [test_trajectory], observation_noise,
                    # TEMP: world-frame -> no denormalization offsets
                    taker_anchors=[None], time_step=1, pause_secs=1.0 / RECORDING_HZ,
                    observe_controlled_start=(baton_start_offset is not None),
                    training_handover_stats=handover_stats,
                    stats_export_dir=analysis_dir,
                    stats_debug_name=f"{test_baton_path.name}|{test_taker_path.name}",
                    save_stats=False)
    


    plt.show()