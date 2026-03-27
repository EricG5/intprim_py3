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
# Quaternions have a sign ambiguity: q and -q represent the same rotation.
# Many data sources will arbitrarily flip signs over time. Converting such a
# sequence to axis-angle / rotvec can create apparent "jitter" (discontinuities)
# even if the underlying orientation is smooth. We fix this by normalizing and
# forcing consecutive quaternions into the same hemisphere.
def _continuous_quats(quats, eps=1e-12):
    """Return a copy of quats normalized and made sign-continuous over time.

    Args:
        quats: (T, 4) array of quaternions in [x, y, z, w] order.
    """
    q = np.array(quats, copy=True)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    q = q / norms

    for i in range(1, q.shape[0]):
        if float(np.dot(q[i - 1], q[i])) < 0.0:
            q[i] *= -1.0
    return q



# -----------------------------------------------------------------------------
# TEMPORARY NOTE (world-frame training/testing)
#
# If you want to train + evaluate in the world frame (no taker-relative anchors),
# keep trajectories in absolute coordinates.
#
# This file previously normalized positions by subtracting the taker start
# position, then later added it back via `denormalize_6d(..., taker_anchor)`.
# For world-frame work we comment out the normalization steps and pass
# `taker_anchors=[None]` into `evaluate_6d`.
# -----------------------------------------------------------------------------



def denormalize_6d(trajectory, anchor):
    """Add the position anchor back to position DOFs (0:3, 6:9).

    Rotation vector DOFs (3:6, 9:12) are left unchanged.

    Args:
        trajectory: (12, T) array.
        anchor:     (3, 1) shared world-frame anchor position.
    """
    result = trajectory.copy()
    result[0:3, :] += anchor
    result[6:9, :] += anchor
    return result


def evaluate_6d(primitive, filter, test_trajectories, observation_noise,
                taker_anchors=None, delay_prob=0.0, delay_ratio=0.0,
                time_step=1, pause_secs=0.04, observe_controlled_start=False,
                training_handover_stats=None,
                drop_redundant_stationary_obs=True,
                stationary_pos_eps=1e-3,
                stationary_rot_eps=5e-3,
                stats_export_dir=None,
                stats_debug_name="baton_6D"):
    """Run 6D BIP inference over test trajectories with live 3D + orientation viewer.

    Like baton_3D.evaluate but operates on 12-DOF trajectories
    (pos + rotvec per agent) and uses Live6DInteractionViewer for
    orientation triads.

    Args:
        primitive:     Trained BayesianInteractionPrimitive.
        filter:        EnKF or similar filter template (will be deep-copied).
        test_trajectories: list of (12, T) arrays.
        observation_noise:  (12, 12) diagonal observation noise.
        taker_anchors: list of (3, 1) arrays — taker world-frame start position.
        time_step:     Observations per inference call.
        pause_secs:    Pause between viewer updates.
        observe_controlled_start: If True, the controlled agent's position DOFs
            [0,1,2] are included as observations at the FIRST inference step
            only, anchoring the filter to the actual starting position.  For
            all subsequent steps only the observed agent (taker) is used.
            This mirrors a real deployment where the robot's starting pose is
            known but its future motion must be inferred from the partner.
        training_handover_stats: Optional dict with keys 'pos_mean', 'pos_std',
            'rot_mean', 'rot_std' (each a (3,) array in normalised coords)
            computed from training demos.  When provided, the final predicted
            endpoint difference (controlled − observed) is compared against
            the training distribution and printed as a handover quality check.
    """
    if taker_anchors is None:
        taker_anchors = [None] * len(test_trajectories)

    # Observed DOF indices: taker position (6,7,8) + taker rotation (9,10,11)
    observed_dof_indices = np.array([6, 7, 8, 9, 10, 11], dtype=np.int32)
    # Extended set that also includes controlled-agent position — used only at
    # the first step when observe_controlled_start=True.
    observed_dof_indices_with_ctrl_pos = np.array([0, 1, 2, 6, 7, 8, 9, 10, 11], dtype=np.int32)

    def _is_stationary_step(traj, prev_col, curr_col, active_dofs):
        """Return True if active DoFs changed less than eps between columns."""
        if prev_col < 0 or curr_col < 0:
            return False

        # Position indices among active DoFs.
        pos_mask = np.isin(active_dofs, np.array([0, 1, 2, 6, 7, 8], dtype=np.int32))
        rot_mask = np.isin(active_dofs, np.array([3, 4, 5, 9, 10, 11], dtype=np.int32))

        diffs = np.abs(traj[active_dofs, curr_col] - traj[active_dofs, prev_col])
        if np.any(pos_mask) and np.max(diffs[pos_mask]) > stationary_pos_eps:
            return False
        if np.any(rot_mask) and np.max(diffs[rot_mask]) > stationary_rot_eps:
            return False
        return True

    for test_trajectory, taker_anchor in zip(test_trajectories, taker_anchors):
        test_trajectory_partial = np.array(test_trajectory, copy=True)
        # Zero out controlled/robot DOFs (0:6) — unobserved at inference.
        test_trajectory_partial[0:6, :] = 0.0

        # If requested, restore the true starting position in the first column
        # so it can be observed at t=0 to anchor the filter.
        if observe_controlled_start:
            test_trajectory_partial[0:3, 0] = test_trajectory[0:3, 0]

        new_filter = copy.deepcopy(filter)
        primitive.set_filter(new_filter)

        mean_trajectory = primitive.get_mean_trajectory()
        mean_world = denormalize_6d(mean_trajectory, taker_anchor) if taker_anchor is not None else mean_trajectory

        viewer = trajPlotting.Live6DInteractionViewer(
            mean_interaction=mean_world,
            observed_agent="taker",
            title="6D BIP live inference",
            pause_secs=pause_secs,
            triad_every_n=15,
            arrow_length=0.04,
        )

        last_gen_world = mean_world

        stat_collector = None
        if stats_export_dir is not None:
            stats_export_dir = Path(stats_export_dir)
            stats_export_dir.mkdir(parents=True, exist_ok=True)

            # Collect PDFs and ensemble projections for the generated (controlled)
            # DoFs while treating the observed (taker) DoFs as observations.
            generated_dof_indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
            stat_collector = intprim.util.stat_collector.StatCollector(
                primitive,
                generated_indices=generated_dof_indices,
                observed_indices=observed_dof_indices,
            )

            # Seed with an initial timestep (clock=None is expected by exporter).
            stat_collector.collect(
                primitive,
                observed_trajectory=test_trajectory_partial[:, :1].T,
                generated_trajectory=mean_trajectory[:, :1].T,
                timestamp=None,
            )

        mean_mse = 0.0
        phase_mae = 0.0
        mse_count = 0
        prev_observed_index = 0

        for observed_index in range(time_step, test_trajectory.shape[1], time_step):
            is_first_step = (prev_observed_index == 0)
            active_dofs = (
                observed_dof_indices_with_ctrl_pos
                if (is_first_step and observe_controlled_start)
                else observed_dof_indices
            )

            # If the observation hasn't changed (within eps), don't feed repeated
            # stationary frames into the phase filter. Repeated identical updates
            # can collapse phase-velocity variance and make inference "stick" at
            # the beginning when long stationary prefixes are present.
            if drop_redundant_stationary_obs and not is_first_step:
                prev_col = (prev_observed_index - 1) if prev_observed_index > 0 else 0
                curr_col = observed_index - 1
                if _is_stationary_step(test_trajectory, prev_col, curr_col, active_dofs):
                    plot_obs = (
                        denormalize_6d(test_trajectory_partial[:, :observed_index], taker_anchor)
                        if taker_anchor is not None
                        else test_trajectory_partial[:, :observed_index]
                    )
                    viewer.update(last_gen_world, plot_obs)
                    prev_observed_index = observed_index
                    continue

            gen_trajectory, phase, mean, var = primitive.generate_probable_trajectory_recursive(
                test_trajectory_partial[:, prev_observed_index:observed_index],
                observation_noise,
                active_dofs,
                num_samples=test_trajectory_partial.shape[1] - observed_index,
            )

            if stat_collector is not None:
                stat_collector.collect(
                    primitive,
                    observed_trajectory=test_trajectory[:, prev_observed_index:observed_index].T,
                    generated_trajectory=gen_trajectory.T,
                    timestamp=float(observed_index),
                )

            gen_world = denormalize_6d(gen_trajectory, taker_anchor) if taker_anchor is not None else gen_trajectory
            last_gen_world = gen_world
            test_future_world = denormalize_6d(test_trajectory[:, observed_index:], taker_anchor) if taker_anchor is not None else test_trajectory[:, observed_index:]

            mse = sklearn.metrics.mean_squared_error(test_future_world, gen_world)
            mean_mse += mse
            mse_count += 1
            phase_mae += np.abs((float(observed_index) / test_trajectory.shape[1]) - phase)

            if delay_prob > 0.0 and np.random.binomial(1, delay_prob) == 1:
                length = int(delay_ratio * test_trajectory.shape[1])
                delay_trajectory = np.tile(test_trajectory[:, observed_index - 1], (length, 1)).T
                gen_trajectory, phase, mean, var = primitive.generate_probable_trajectory_recursive(
                    delay_trajectory,
                    observation_noise,
                    observed_dof_indices,
                    num_samples=test_trajectory_partial.shape[1] - observed_index,
                )
                gen_world = denormalize_6d(gen_trajectory, taker_anchor) if taker_anchor is not None else gen_trajectory
                mse = sklearn.metrics.mean_squared_error(test_future_world, gen_world)
                mean_mse += mse
                mse_count += 1
                phase_mae += np.abs((float(observed_index) / test_trajectory.shape[1]) - phase)

            plot_obs = denormalize_6d(test_trajectory_partial[:, :observed_index], taker_anchor) if taker_anchor is not None else test_trajectory_partial[:, :observed_index]
            viewer.update(gen_world, plot_obs)

            prev_observed_index = observed_index

        print("Mean DoF MSE: " + str(mean_mse / mse_count) + ". Phase MAE: " + str(phase_mae / mse_count))

        # ---- Final-pose handover quality check --------------------------------
        # gen_trajectory[:, -1] is the predicted endpoint of the controlled agent
        # in normalised space.  test_trajectory[:, -1] is the actual final pose
        # of the observed agent.  Their difference should match the training
        # handover offset distribution if the interaction was successful.
        if training_handover_stats is not None:
            # Compare the predicted controlled-agent endpoint directly against
            # the training distribution of controlled-agent final positions.
            # This avoids contamination from artificial offsets applied to the
            # test observed-agent trajectory.
            pred_ctrl_pos = gen_trajectory[0:3, -1]
            pred_ctrl_rot = gen_trajectory[3:6, -1]

            ctrl_pos_mean = training_handover_stats["ctrl_pos_mean"]
            ctrl_pos_std  = training_handover_stats["ctrl_pos_std"]
            ctrl_rot_mean = training_handover_stats["ctrl_rot_mean"]
            ctrl_rot_std  = training_handover_stats["ctrl_rot_std"]

            # Z-score to determine the number of standard deviations away from the mean the resulting difference is
            pos_dev = (pred_ctrl_pos - ctrl_pos_mean) / (ctrl_pos_std + 1e-9)
            rot_dev = (pred_ctrl_rot - ctrl_rot_mean) / (ctrl_rot_std + 1e-9)

            print("\n--- Final-pose handover quality check (trajectory coords) ---")
            print(f"  Predicted  ctrl final pos [x,y,z]  : {pred_ctrl_pos}")
            print(f"  Training   mean ± std              : {ctrl_pos_mean} ± {ctrl_pos_std}")
            print(f"  Deviation  (σ)                     : {pos_dev}")
            print(f"  Predicted  ctrl final rot [rx,ry,rz]: {pred_ctrl_rot}")
            print(f"  Training   mean ± std              : {ctrl_rot_mean} ± {ctrl_rot_std}")
            print(f"  Deviation  (σ)                     : {rot_dev}")
            within = np.all(np.abs(pos_dev) < 2.0)
            print(f"  Position within 2σ of training?    : {'YES' if within else 'NO — likely poor handover'}")
            print("-------------------------------------------------------------\n")

        viewer.keep_open()

        if stat_collector is not None:
            # The XML exporter assumes >=3 timesteps (it extrapolates an
            # initial timestamp from the first two real ones).
            if len(stat_collector.timestep_clock) >= 3:
                stat_collector.export(
                    primitive,
                    str(stats_export_dir),
                    debug_bag_file=str(stats_debug_name),
                    response_length=int(test_trajectory.shape[1]),
                    use_spt=False,
                    spt_phase="current",
                )
            else:
                print("[stats] Skipping XML export: not enough timesteps collected.")


if __name__ == "__main__":
    # Set a seed for reproducibility
    np.random.seed(213413414)

    # Use a stable path so you can run from any working directory.
    data_dir = Path(__file__).resolve().parent / "trainingData"
    testdata_dir = Path(__file__).resolve().parent / "testingData"



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
        baton_rot = Rotation.from_quat(_continuous_quats(baton_approach[:, 3:7])).as_rotvec()     # (T, 3)
        taker_pos = taker_approach[:, :3]
        taker_rot = Rotation.from_quat(_continuous_quats(taker_approach[:, 3:7])).as_rotvec()

        mean_baton_start[0] += baton_pos[0,0]
        mean_baton_start[1] += baton_pos[0,1]
        mean_baton_start[2] += baton_pos[0,2]

        # Combine into (T, 6) per agent: [x, y, z, rx, ry, rz]
        baton_6d = np.hstack((baton_pos, baton_rot))  # (T, 6)
        taker_6d = np.hstack((taker_pos, taker_rot))  # (T, 6)

        raw_traj = np.concatenate((baton_6d.T, taker_6d.T), axis=0)  # (12, T)

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
        
        # Plotting individual paired trajectories - Don't do with big dataset, useful for sanity check
        # if i < 2:
        #     fig, ax, ani = trajPlotting.plot_animate(
        #     baton_approach,
        #     taker_approach,
        #     interval=20,
        #     show=True,
        #     labels=("baton", "taker"),
        #     )

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

    # Set to True to run training and evaluation; False to just visualize trajectories
    train_flag = True
    if (train_flag):
        # Basis space selection
        selection = intprim.basis.Selection(dof_names)

        for trajectory in training_trajectories:
            selection.add_demonstration(trajectory)

        # aic, bic = selection.get_information_criteria(np.array([0, 1], dtype = np.int32))
        # selection.get_best_model(aic, bic)

        basis_model_gaussian = intprim.basis.GaussianModel(5, 0.16, dof_names)
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
        primitive.export_data("full_baton_6d_model_world.bip")

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

        test_baton_path = testdata_dir / "baton_058.csv"
        test_taker_path = testdata_dir / "taker_058.csv"
        test_baton = np.loadtxt(str(test_baton_path), delimiter=",", dtype=dtype, skiprows=1)
        test_taker = np.loadtxt(str(test_taker_path), delimiter=",", dtype=dtype, skiprows=1)
        # test_taker[:,0] += 0.10  # Shift the taker X position by +5 cm to test spatial robustness.
        test_baton_approach, test_taker_approach = baton_3D.process_approach(test_baton[:400, :], test_taker[:400, :], 50, delay)
        # test_baton_approach, test_taker_approach = test_baton[full_traj_start:400, :], test_taker[full_traj_start:400, :]

        # Convert quaternions to rotation vectors (same as training data).
        test_baton_pos = test_baton_approach[:, :3]
        test_baton_rot = Rotation.from_quat(_continuous_quats(test_baton_approach[:, 3:7])).as_rotvec()
        test_taker_pos = test_taker_approach[:, :3]
        test_taker_rot = Rotation.from_quat(_continuous_quats(test_taker_approach[:, 3:7])).as_rotvec()

        test_baton_6d = np.hstack((test_baton_pos, test_baton_rot))  # (T, 6)
        test_taker_6d = np.hstack((test_taker_pos, test_taker_rot))  # (T, 6)

        raw_test_trajectory = np.concatenate((test_baton_6d.T, test_taker_6d.T), axis=0)  # (12, T)

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
            
        # np.savetxt("test_trajectory.csv", raw_test_trajectory.T, delimiter=",", header="X_ctrl,Y_ctrl,Z_ctrl,RX_ctrl,RY_ctrl,RZ_ctrl,X_obs,Y_obs,Z_obs,RX_obs,RY_obs,RZ_obs", comments="")
        
        # Delayed start test: add stationary data points to start of trajectory to simulate a late start in the observed agent's motion.  This checks whether the filter can handle an initial period of static observation before the interaction begins.
        # delayed_start_test_trajectory = np.zeros((12, test_trajectory.shape[1] + 30), dtype=test_trajectory.dtype)
        # delayed_start_test_trajectory[:, 30:test_trajectory.shape[1]+30] = test_trajectory
        # delayed_start_test_trajectory[:, :30] = test_trajectory[:, 0:1]  # Prepend 30 frames of the initial pose to simulate a delayed start with static initial observation.

        # Evaluate the trajectories: step=1 gives a per-index live view.
        # Data was recorded at 120 Hz, so each index = 1/120 s ≈ 0.00833 s.
        evaluate_6d(primitive, filter, [test_trajectory], observation_noise,
                    # TEMP: world-frame -> no denormalization offsets
                    taker_anchors=[None], time_step=1, pause_secs=1.0 / RECORDING_HZ,
                    observe_controlled_start=(baton_start_offset is not None),
                    training_handover_stats=handover_stats,
                    stats_export_dir=analysis_dir,
                    stats_debug_name=f"{test_baton_path.name}|{test_taker_path.name}")
    


    plt.show()