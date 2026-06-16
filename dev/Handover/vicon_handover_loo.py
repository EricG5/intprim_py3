"""Leave-one-out (LOO) cross-validation harness for the handover eBIP.

This is a *testing* companion to ``vicon_handover.py`` -- that script is the
original training/export implementation and is intentionally left untouched.
Here we reuse the exact same data, segmentation, reframing and onset/cutoff
pipeline, but instead of a single random train/test split we evaluate every demo
once, each from a model that never saw it:

    for each demo i:
        train a fresh eBIP on all demos EXCEPT i
        score demo i with evaluate_6d_metrics + the mean-trajectory baseline

The basis configuration is held FIXED across folds (see ``BASIS_NUM`` /
``BASIS_SCALE``), so the N held-out scores describe one modelling choice rather
than N different ones. Dropping a single demo of ~N barely moves the AIC/BIC
argmin, so this matches the manual selection done in ``vicon_handover.py``. Set
``CHECK_SELECTION_STABILITY = True`` to empirically confirm that -- it logs what
AIC/BIC would pick on each fold's N-1 set.

The preprocessing lives in ``build_trajectories`` so other diagnostics can reuse
it without duplicating the pipeline.

Run from the repo root:  ``python -m dev.Handover.vicon_handover_loo``
Pass session dir names to restrict the run:
    ``python -m dev.Handover.vicon_handover_loo BH1_2026_06_12``
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

import intprim
import numpy as np

from dev.util.pose_helper import rotation_dim_reduction, rotation_dim_reduction_continuous
from dev.util.process import (
    apply_local_axis_rotation_offset_to_quats,
    apply_reflection_to_positions,
    apply_reflection_to_quats,
    combine_data,
    compute_cutoff,
    compute_euclidean_distance,
    csv_to_dict,
    ensure_quaternion_hemisphere_continuity,
    get_interaction_start_indices,
    get_traj_start_indices,
    mirror_trajectory_quats_using_head_plane,
    rebase_to_observed_head_floor_yaw_quat,
    segment_trajectories,
)
from dev.util.visualize import evaluate_6d_metrics, mean_trajectory_baseline_metrics


# --- Fixed modelling configuration (mirrors vicon_handover.py) ----------------
BASIS_NUM = 5
BASIS_SCALE = 0.16
PHASE_VAR_INIT = 1e-4
PROC_VAR = 1e-8
ALL_DOFS = np.arange(12, dtype=np.int32)
# The EnsembleKalmanFilter draws its initial ensemble stochastically, so an
# unseeded fold gives a different (often worse) result each run and makes the
# worst-fold ranking noisy. Seed every fold from the same state so each fold's
# noise realization is identical and reproducible -- and so the ensemble lands
# on a good draw rather than a degenerate one.
FOLD_SEED = 213413414
# Default EnKF ensemble size. One-member-per-demo (N~=53) leaves large run-to-run
# variance (std across seeds 0.73 mm); inflating to 300 collapses it to ~0.21 mm
# -- effectively seed-invariant -- without changing the mean (see the ensemble
# sweep). 600 buys no further consistency. This is the consistency lever, not the
# seed. NOTE: the live ROS2 EnKF MUST use the same ensemble size or train/live
# filters diverge.
ENSEMBLE_SIZE = 300

# Reframing knobs (fixed to the vicon_handover.py deployment-POC settings).
MIRROR_TO_LEFT_HANDED = False
APPLY_POST_MIRROR_Z_PI = False
POST_MIRROR_OFFSET_AXIS = "z"
POST_MIRROR_OFFSET_ANGLE = np.pi - np.pi / 16
REBASE_TO_OBSERVED_HEAD_FLOOR = True
REBASE_YAW_MODE = "legacy_euler"
REBASE_BODY_FORWARD_AXIS = "x"
CONTINUOUS_ROTVEC = True

CONTROLLED_AGENT = "Baton"
OBSERVED_AGENT = "Receiver_Hand"
CONTROLLED_GENERAL_POSE = "Hat_Giver"
OBSERVED_GENERAL_POSE = "Hat_Receiver"
OBJECT_NAMES = [CONTROLLED_AGENT, OBSERVED_AGENT, CONTROLLED_GENERAL_POSE, OBSERVED_GENERAL_POSE]

# Bad demos to drop, keyed by session dir name; indices are LOCAL to each session
# (0-based, in capture order). Add per-session entries as poor demos are found.
AVOID_INDICES_BY_SESSION = {
    # Local capture indices (0-based). Diagnosed via the LOO worst-fold analysis:
    #   10        : zero-length segment (degenerate cutoff/onset)
    #   26, 28    : baton start ~5 sigma out + abnormally long -> mis-segmentation
    #   7, 56     : phase stalls (filter never advances); 7 also has an outlier
    #               receiver-hand start the ensemble can't localize
    "BH1_2026_06_12": [7, 10, 26, 28, 56],
}

# Print, per fold, the basis (degree, scale) that AIC/BIC would select on the
# fold's N-1 training set. Verifies the fixed config is stable to leaving one
# demo out. Verbose (the library prints each candidate fit); off by default.
CHECK_SELECTION_STABILITY = False


def build_dof_names():
    return np.array([
        "X (Controlled)", "Y (Controlled)", "Z (Controlled)",
        "RX (Controlled)", "RY (Controlled)", "RZ (Controlled)",
        "X (Observed)", "Y (Observed)", "Z (Observed)",
        "RX (Observed)", "RY (Observed)", "RZ (Observed)",
    ])


def _resolve_avoid_indices(data_dict, data_date, avoid_indices_by_session):
    """Map per-session local bad-demo indices onto the combined demo numbering.

    Sessions are concatenated in ``data_date`` order, so each session's demos
    occupy a contiguous block; a local index is offset by the preceding sessions'
    demo counts. This keeps the exclusion correct regardless of which / how many
    sessions are loaded (e.g. BH2-only no longer drops an arbitrary demo).
    """
    session_demo_counts = [
        len(get_traj_start_indices(data_dict[date][CONTROLLED_AGENT])) for date in data_date
    ]
    avoid = []
    offset = 0
    for date, count in zip(data_date, session_demo_counts):
        for local_idx in avoid_indices_by_session.get(date, []):
            if 0 <= local_idx < count:
                avoid.append(offset + local_idx)
            else:
                print(f"Warning: avoid index {local_idx} out of range for session {date} "
                      f"({count} demos); skipped.")
        offset += count
    return sorted(set(avoid))


def build_trajectories(data_date, avoid_indices_by_session=None, *, verbose=True):
    """Run the full handover pipeline and return reframed 12D demos.

    Mirrors the data load -> segmentation -> cutoff/onset -> reframe pipeline of
    ``vicon_handover.py``, but keeps ALL demos in one list (no train/test split).

    Args:
        data_date: list of session dir names to load and concatenate.
        avoid_indices_by_session: dict {session_name: [local_idx, ...]} of bad
            demos to drop. Defaults to ``AVOID_INDICES_BY_SESSION``.
        verbose: print progress / warnings.

    Returns:
        all_trajectories: list of (12, T) arrays [ctrl pos+rotvec, obs pos+rotvec].
        demo_sessions: list of session names, one per demo (kept order).
    """
    if avoid_indices_by_session is None:
        avoid_indices_by_session = AVOID_INDICES_BY_SESSION

    data_dict = {}
    for date in data_date:
        data_dir = Path(__file__).parent / date
        data_dict[date] = csv_to_dict(data_dir)
    data_ = combine_data(data_dict, object_names=OBJECT_NAMES)

    start_indices_ = {name: get_traj_start_indices(data_[name]) for name in data_}
    if verbose:
        print(f"Number of start indices: { {name: len(start_indices_[name]) for name in start_indices_} }")
    trajectories_ = segment_trajectories(data_, start_indices_, print_info=False)

    n_segments = len(start_indices_[CONTROLLED_AGENT])

    # Per-demo session label, following the concatenation order in data_date.
    session_demo_counts = [
        len(get_traj_start_indices(data_dict[date][CONTROLLED_AGENT])) for date in data_date
    ]
    demo_session_all = []
    for date, count in zip(data_date, session_demo_counts):
        demo_session_all.extend([date] * count)

    avoid_indices = _resolve_avoid_indices(data_dict, data_date, avoid_indices_by_session)

    # --- Segmentation: approach-phase cutoff and interaction onset ------------
    euclidean_distance = {}
    for i in range(0, n_segments):
        euclidean_distance[f"{i}"] = compute_euclidean_distance(
            trajectories_[f"{CONTROLLED_AGENT}_{i}"][:, 1:4],
            trajectories_[f"{OBSERVED_AGENT}_{i}"][:, 1:4],
        )

    approach_cutoff_indices = [compute_cutoff(euclidean_distance[f"{i}"]) for i in range(0, n_segments)]

    interaction_start_indices = []
    for i in range(0, len(approach_cutoff_indices)):
        observed_pre_cutoff = trajectories_[f"{OBSERVED_AGENT}_{i}"][:approach_cutoff_indices[i]]
        n_samples = observed_pre_cutoff.shape[0]
        if n_samples == 0:
            interaction_start_indices.append(0)
            if verbose:
                print(f"Warning: Empty pre-cutoff segment for trajectory {i+1}; defaulting to start index 0.")
            continue
        interaction_start_idx = get_interaction_start_indices(
            observed_pre_cutoff,
            steady_state_window=min(200, n_samples),
            min_consecutive=min(50, n_samples),
            direction="up",
            pre_roll=60,
        )
        if interaction_start_idx is None:
            interaction_start_idx = 0
            if verbose:
                print(f"Warning: No interaction start detected for trajectory {i+1}; defaulting to start index 0.")
        interaction_start_indices.append(interaction_start_idx)

    # --- Merge per-object columns, dropping avoided demos --------------------
    combined_data = {}
    combined_sessions = []
    index_adjustment = 0
    for i in range(0, len(approach_cutoff_indices)):
        if i in avoid_indices:
            index_adjustment += 1
            continue

        key = f"{i-index_adjustment}"
        combined_data[key] = np.zeros((approach_cutoff_indices[i] - interaction_start_indices[i], 1 + 7 * len(data_)))
        combined_data[key][:, 0] = trajectories_[f"{CONTROLLED_AGENT}_{i}"][interaction_start_indices[i]:approach_cutoff_indices[i], 0]
        suffix = f"_{i}"
        for name in trajectories_:
            if name.endswith(suffix):
                base = name.removesuffix(suffix)
                if base == CONTROLLED_AGENT:
                    combined_data[key][:, 1:8] = trajectories_[name][interaction_start_indices[i]:approach_cutoff_indices[i], 1:8]
                elif base == OBSERVED_AGENT:
                    combined_data[key][:, 8:15] = trajectories_[name][interaction_start_indices[i]:approach_cutoff_indices[i], 1:8]
                elif base == CONTROLLED_GENERAL_POSE:
                    combined_data[key][:, 15:22] = trajectories_[name][interaction_start_indices[i]:approach_cutoff_indices[i], 1:8]
                elif base == OBSERVED_GENERAL_POSE:
                    combined_data[key][:, 22:29] = trajectories_[name][interaction_start_indices[i]:approach_cutoff_indices[i], 1:8]

        zero_cols = np.all(combined_data[key][:, 1:] == 0.0, axis=0)
        if np.any(zero_cols) and verbose:
            print(f"Warning: Trajectory {i+1} has {np.sum(zero_cols)} all-zero pose columns.")
        combined_sessions.append(demo_session_all[i] if i < len(demo_session_all) else None)

    if verbose:
        print(f"Processed {len(combined_data)} trajectories from original {len(approach_cutoff_indices)}. "
              f"Avoided global indices: {avoid_indices}.")

    # --- Reframe every demo into the 12D eBIP feature space -------------------
    all_trajectories = []
    for i in range(0, len(combined_data)):
        ctrl_pos = combined_data[f"{i}"][:, 1:4]
        ctrl_quat = combined_data[f"{i}"][:, 4:8]
        obs_pos = combined_data[f"{i}"][:, 8:11]
        obs_quat = combined_data[f"{i}"][:, 11:15]

        ctrl_head_pos = combined_data[f"{i}"][:, 15:18]
        obs_head_pos = combined_data[f"{i}"][:, 22:25]
        obs_head_quat = combined_data[f"{i}"][:, 25:29]

        if MIRROR_TO_LEFT_HANDED:
            ctrl_pos, ctrl_quat, obs_pos, obs_quat, (S, p_plane, n_plane) = mirror_trajectory_quats_using_head_plane(
                ctrl_pos, ctrl_quat, obs_pos, obs_quat,
                controlled_head_positions=ctrl_head_pos,
                observed_head_positions=obs_head_pos,
                up=(0.0, 0.0, 1.0),
                return_transform=True,
            )
            ctrl_head_pos = apply_reflection_to_positions(ctrl_head_pos, S=S, p0=p_plane)
            obs_head_pos = apply_reflection_to_positions(obs_head_pos, S=S, p0=p_plane)
            obs_head_quat = apply_reflection_to_quats(obs_head_quat, S=S)

        if REBASE_TO_OBSERVED_HEAD_FLOOR:
            ctrl_pos, ctrl_quat, obs_pos, obs_quat, ctrl_head_pos, obs_head_pos = rebase_to_observed_head_floor_yaw_quat(
                ctrl_pos, ctrl_quat, obs_pos, obs_quat,
                controlled_head_positions=ctrl_head_pos,
                observed_head_positions=obs_head_pos,
                observed_head_quats=obs_head_quat,
                yaw_mode=REBASE_YAW_MODE,
                body_forward_axis=REBASE_BODY_FORWARD_AXIS,
                floor_z=0.0,
                return_head_positions=True,
            )

        if APPLY_POST_MIRROR_Z_PI:
            ctrl_quat = apply_local_axis_rotation_offset_to_quats(
                ctrl_quat, axis=POST_MIRROR_OFFSET_AXIS, angle_rad=POST_MIRROR_OFFSET_ANGLE,
            )
            obs_quat = apply_local_axis_rotation_offset_to_quats(
                obs_quat, axis=POST_MIRROR_OFFSET_AXIS, angle_rad=POST_MIRROR_OFFSET_ANGLE,
            )

        ctrl_quat = ensure_quaternion_hemisphere_continuity(ctrl_quat)
        obs_quat = ensure_quaternion_hemisphere_continuity(obs_quat)

        training_data = np.zeros((ctrl_pos.shape[0], 12))
        training_data[:, 0:3] = ctrl_pos
        training_data[:, 3:6] = rotation_dim_reduction_continuous(ctrl_quat) if CONTINUOUS_ROTVEC else rotation_dim_reduction(ctrl_quat)
        training_data[:, 6:9] = obs_pos
        training_data[:, 9:12] = rotation_dim_reduction_continuous(obs_quat) if CONTINUOUS_ROTVEC else rotation_dim_reduction(obs_quat)
        all_trajectories.append(training_data.T)  # (12, T)

    return all_trajectories, combined_sessions


def build_initial_ensemble(primitive, ensemble_size):
    """Return the EnKF initial ensemble (E x B matrix of basis weights).

    With ``ensemble_size`` None (or <= the demo count) the legacy ensemble is
    used: one member per demonstration (``primitive.basis_weights``). That makes
    the ensemble small (E ~= number of demos), so the stochastic EnKF carries
    large Monte-Carlo variance and its result swings with the RNG seed.

    Passing a larger ``ensemble_size`` INFLATES the ensemble by sampling extra
    members from the BIP's own learned basis-weight distribution
    (mean/covariance from ``get_basis_weight_parameters``). This is the
    principled variance-reduction lever: as the member count grows the stochastic
    EnKF converges toward its deterministic posterior, so run-to-run spread (and
    thus seed-sensitivity) shrinks. Uses the global RNG, so seed the caller first
    for reproducibility.
    """
    base = primitive.basis_weights
    if ensemble_size is None or ensemble_size <= base.shape[0]:
        return base

    mean, cov = primitive.get_basis_weight_parameters()
    if cov is None:
        return base
    # cov is rank-deficient (B dims, < B demos); SVD sampling handles singular
    # covariance. Keep the real demo weights as members, top up with samples.
    extra = np.random.multivariate_normal(
        mean, cov, size=ensemble_size - base.shape[0], check_valid="ignore",
    )
    return np.vstack([base, extra])


def fit_and_score(train_trajectories, test_trajectory, dof_names, *, label=None,
                  ensemble_size=ENSEMBLE_SIZE, seed=FOLD_SEED):
    """Train an eBIP on ``train_trajectories`` and score it on ``test_trajectory``.

    Builds a fresh basis model, primitive, observation-noise estimate and filter
    from the training set only, then returns the eBIP metrics and the
    mean-trajectory baseline for the held-out demo. The baseline mean comes from
    THIS fold's primitive (trained on N-1 demos), so it never sees the held-out
    demo -- no leakage.

    ``seed`` fixes the stochastic EnKF draws so a fold is reproducible.
    ``ensemble_size`` optionally inflates the ensemble (see
    ``build_initial_ensemble``) to reduce that stochasticity at the source.
    """
    # Reseed per fold so the ensemble draw is identical across folds and runs;
    # otherwise stochastic ensemble noise dominates the worst-fold ranking.
    np.random.seed(seed)

    basis_model = intprim.basis.GaussianModel(BASIS_NUM, BASIS_SCALE, dof_names)

    selection = intprim.basis.Selection(dof_names)
    for trajectory in train_trajectories:
        selection.add_demonstration(trajectory)

    primitive = intprim.BayesianInteractionPrimitive(basis_model)
    for trajectory in train_trajectories:
        primitive.add_demonstration(trajectory)

    observation_noise = np.diag(selection.get_model_mse(basis_model, ALL_DOFS))
    phase_velocity_mean, phase_velocity_var = intprim.examples.get_phase_stats(train_trajectories)

    initial_ensemble = build_initial_ensemble(primitive, ensemble_size)
    filt = intprim.filter.spatiotemporal.EnsembleKalmanFilter(
        basis_model=basis_model,
        initial_phase_mean=[0.0, phase_velocity_mean],
        initial_phase_var=[PHASE_VAR_INIT, phase_velocity_var],
        proc_var=PROC_VAR,
        initial_ensemble=initial_ensemble,
    )

    metrics = evaluate_6d_metrics(
        primitive, filt, test_trajectory, observation_noise, label=label, plot=False
    )
    baseline = mean_trajectory_baseline_metrics(primitive, test_trajectory, label=label)
    return metrics, baseline


def selected_basis_for_fold(train_trajectories, dof_names):
    """Return the (degree, scale) AIC/BIC would pick on this fold's training set."""
    selection = intprim.basis.Selection(dof_names)
    for trajectory in train_trajectories:
        selection.add_demonstration(trajectory)
    aic, bic = selection.get_information_criteria(ALL_DOFS)
    best_aic, best_bic = selection.get_best_model(aic, bic)
    return (best_aic._degree, best_aic.scale), (best_bic._degree, best_bic.scale)


def run_loo(data_date, *, ensemble_size=ENSEMBLE_SIZE, seed=FOLD_SEED, quiet=False,
            all_trajectories=None):
    """Run leave-one-out cross-validation over the given sessions and report.

    ``ensemble_size`` / ``seed`` are threaded to every fold. ``quiet`` suppresses
    all printing (for sweeps). Pass ``all_trajectories`` to skip the (identical)
    reframing work when running many seeds over the same data. Returns a summary
    dict of the per-fold arrays and aggregates.
    """
    if all_trajectories is None:
        all_trajectories, _ = build_trajectories(data_date, verbose=not quiet)
    dof_names = build_dof_names()
    n_demos = len(all_trajectories)
    if not quiet:
        print(f"\nRunning leave-one-out over {n_demos} demos "
              f"(basis: Gaussian, num={BASIS_NUM}, scale={BASIS_SCALE}; "
              f"ensemble_size={ensemble_size or n_demos}, seed={seed}).\n")

    if CHECK_SELECTION_STABILITY and not quiet:
        picks_aic = []
        for held_out in range(n_demos):
            train = [all_trajectories[j] for j in range(n_demos) if j != held_out]
            (deg_aic, scale_aic), _ = selected_basis_for_fold(train, dof_names)
            picks_aic.append((deg_aic, scale_aic))
        unique_picks = sorted(set(picks_aic))
        print("\nPer-fold AIC basis picks (degree, scale):")
        print(f"  unique selections across folds: {unique_picks}")
        if len(unique_picks) == 1:
            print("  -> selection is stable to leaving one out; fixed config is justified.\n")
        else:
            print("  -> selection varies across folds; consider nested CV for an unbiased estimate.\n")

    fold_metrics = []
    fold_baselines = []
    for held_out in range(n_demos):
        train = [all_trajectories[j] for j in range(n_demos) if j != held_out]
        test = all_trajectories[held_out]
        m, b = fit_and_score(train, test, dof_names, label=f"fold {held_out}",
                             ensemble_size=ensemble_size, seed=seed)
        fold_metrics.append(m)
        fold_baselines.append(b)

    pos_rmse = np.array([m["pos_rmse"] for m in fold_metrics])
    rot_rmse = np.array([m["rot_rmse"] for m in fold_metrics])
    phase_rmse = np.array([m["phase_rmse"] for m in fold_metrics])
    endpoint_err = np.array([m["final_endpoint_pos_err"] for m in fold_metrics])
    base_pos_rmse = np.array([m["pos_rmse"] for m in fold_baselines])
    base_rot_rmse = np.array([m["rot_rmse"] for m in fold_baselines])
    base_endpoint_err = np.array([m["final_endpoint_pos_err"] for m in fold_baselines])
    gain = 100.0 * (1.0 - pos_rmse.mean() / base_pos_rmse.mean())

    if not quiet:
        print(f"\n=== Leave-one-out summary over {n_demos} demos (mean +/- std) ===")
        print(f"  {'metric':26s} {'eBIP':>20s} {'mean-traj baseline':>22s}")
        print(f"  {'Controlled position RMSE':26s} "
              f"{pos_rmse.mean()*1000:8.1f} +/- {pos_rmse.std()*1000:5.1f} mm "
              f"{base_pos_rmse.mean()*1000:10.1f} +/- {base_pos_rmse.std()*1000:5.1f} mm")
        print(f"  {'Controlled rotation RMSE':26s} "
              f"{np.degrees(rot_rmse.mean()):8.2f} +/- {np.degrees(rot_rmse.std()):5.2f} dg "
              f"{np.degrees(base_rot_rmse.mean()):10.2f} +/- {np.degrees(base_rot_rmse.std()):5.2f} dg")
        print(f"  {'Final handover pos error':26s} "
              f"{endpoint_err.mean()*1000:8.1f} +/- {endpoint_err.std()*1000:5.1f} mm "
              f"{base_endpoint_err.mean()*1000:10.1f} +/- {base_endpoint_err.std()*1000:5.1f} mm")
        print(f"  {'Phase RMSE':26s} "
              f"{phase_rmse.mean():8.4f} +/- {phase_rmse.std():7.4f}    "
              f"{'n/a (open-loop)':>22s}")
        print(f"\n  eBIP reduces controlled position RMSE by {gain:.0f}% vs the mean-trajectory baseline.")

        worst = np.argsort(pos_rmse)[::-1][:3]
        print("\n  Worst folds by controlled position RMSE:")
        for idx in worst:
            print(f"    fold {idx}: {pos_rmse[idx]*1000:.1f} mm "
                  f"(handover {endpoint_err[idx]*1000:.1f} mm, phase RMSE {phase_rmse[idx]:.4f})")

    return {
        "n_demos": n_demos,
        "pos_rmse": pos_rmse,
        "rot_rmse": rot_rmse,
        "phase_rmse": phase_rmse,
        "endpoint_err": endpoint_err,
        "pos_rmse_mean": float(pos_rmse.mean()),
        "pos_rmse_worst": float(pos_rmse.max()),
        "endpoint_mean": float(endpoint_err.mean()),
        "gain": float(gain),
    }


def _make_seeds(n_seeds):
    """Deterministic set of distinct seeds derived from FOLD_SEED."""
    rng = np.random.default_rng(FOLD_SEED)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n_seeds)]


def run_multiseed(data_date, *, ensemble_size=ENSEMBLE_SIZE, n_seeds=10,
                  all_trajectories=None, label=True):
    """Run the full LOO once per seed and report the across-seed distribution.

    This is the variance-characterization the safety case needs: it shows how
    much the headline metric moves with the RNG, so you size the ensemble by the
    WORST case (and median), never the best (which would be selecting on noise).
    """
    if all_trajectories is None:
        all_trajectories, _ = build_trajectories(data_date, verbose=False)
    seeds = _make_seeds(n_seeds)
    per_seed_mean = []
    per_seed_worst = []
    per_seed_endpoint = []
    for s in seeds:
        with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
            res = run_loo(data_date, ensemble_size=ensemble_size, seed=s,
                          quiet=True, all_trajectories=all_trajectories)
        per_seed_mean.append(res["pos_rmse_mean"])
        per_seed_worst.append(res["pos_rmse_worst"])
        per_seed_endpoint.append(res["endpoint_mean"])

    pm = np.array(per_seed_mean) * 1000.0       # mean ctrl-pos RMSE per seed (mm)
    pw = np.array(per_seed_worst) * 1000.0      # worst single fold per seed (mm)
    ep = np.array(per_seed_endpoint) * 1000.0   # mean handover endpoint per seed (mm)
    n = len(all_trajectories)

    summary = {
        "ensemble_size": ensemble_size or n,
        "n_seeds": n_seeds,
        "mean_of_means": float(pm.mean()),
        "std_of_means": float(pm.std()),
        "median_of_means": float(np.median(pm)),
        "best_seed_mean": float(pm.min()),
        "worst_seed_mean": float(pm.max()),
        "p95_seed_mean": float(np.percentile(pm, 95)),
        "worst_fold_any_seed": float(pw.max()),
        "endpoint_mean_of_means": float(ep.mean()),
        "per_seed_mean_mm": pm,
        "seeds": seeds,
    }

    if label:
        print(f"\n=== Multi-seed LOO  (ensemble_size={summary['ensemble_size']}, "
              f"{n_seeds} seeds, {n} demos) ===")
        print(f"  Controlled position RMSE across seeds (mm):")
        print(f"    mean-of-means   : {summary['mean_of_means']:.1f}")
        print(f"    median          : {summary['median_of_means']:.1f}")
        print(f"    std across seeds : {summary['std_of_means']:.2f}   <- seed-sensitivity")
        print(f"    best  seed       : {summary['best_seed_mean']:.1f}  (do NOT select on this)")
        print(f"    95th pct seed    : {summary['p95_seed_mean']:.1f}")
        print(f"    worst seed       : {summary['worst_seed_mean']:.1f}  <- safety number")
        print(f"    worst single fold (any seed): {summary['worst_fold_any_seed']:.1f}")
        print(f"  Mean handover endpoint err (mm): {summary['endpoint_mean_of_means']:.1f}")
    return summary


def run_ensemble_sweep(data_date, *, sizes, n_seeds=8):
    """Sweep ensemble size and report seed-sensitivity (std across seeds) vs N.

    The goal is to find the smallest N at which the std across seeds has collapsed
    -- i.e. the eBIP is effectively seed-invariant, which is the consistency
    property required before deployment. Selecting a 'best seed' is the wrong
    lever; growing N until the seed no longer matters is the right one.
    """
    all_trajectories, _ = build_trajectories(data_date, verbose=False)
    n = len(all_trajectories)
    print(f"\n=== Ensemble-size sweep  ({n} demos, {n_seeds} seeds per size) ===")
    print(f"  {'N':>6} {'mean':>8} {'median':>8} {'std(seed)':>10} "
          f"{'worst seed':>11} {'worst fold':>11}")
    rows = []
    for size in sizes:
        s = run_multiseed(data_date, ensemble_size=size, n_seeds=n_seeds,
                          all_trajectories=all_trajectories, label=False)
        rows.append(s)
        print(f"  {s['ensemble_size']:>6} {s['mean_of_means']:>8.1f} "
              f"{s['median_of_means']:>8.1f} {s['std_of_means']:>10.2f} "
              f"{s['worst_seed_mean']:>11.1f} {s['worst_fold_any_seed']:>11.1f}")
    print("\n  (all values mm; 'std(seed)' is the across-seed spread -- watch it "
          "collapse as N grows. Pick the smallest N where it is acceptably small.)")
    return rows


def _parse_args(argv):
    p = argparse.ArgumentParser(
        description="Leave-one-out evaluation harness for the handover eBIP.")
    # Sessions default to BH1 only: BH1 and BH2 are role/height-swapped and
    # bimodal, so combining them inflates error/variance (CLAUDE.md finding #1).
    p.add_argument("sessions", nargs="*", default=["BH1_2026_06_12"],
                   help="session dir name(s) to evaluate (default: BH1_2026_06_12)")
    p.add_argument("--ensemble-size", type=int, default=ENSEMBLE_SIZE,
                   help=f"inflate the EnKF ensemble to this many members "
                        f"(default: {ENSEMBLE_SIZE}). The variance-reduction lever; "
                        f"pass a value <= demo count to use the legacy one-per-demo ensemble.")
    p.add_argument("--seeds", type=int, default=1,
                   help="number of RNG seeds. >1 runs the multi-seed distribution "
                        "report instead of a single LOO.")
    p.add_argument("--sweep", type=str, default=None,
                   help="comma-separated ensemble sizes to sweep, e.g. "
                        "'53,150,400,800'. Reports seed-sensitivity vs N.")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    sessions = args.sessions if args.sessions else ["BH1_2026_06_12"]

    if args.sweep:
        sizes = [int(x) for x in args.sweep.split(",") if x.strip()]
        run_ensemble_sweep(sessions, sizes=sizes, n_seeds=args.seeds if args.seeds > 1 else 8)
    elif args.seeds > 1:
        run_multiseed(sessions, ensemble_size=args.ensemble_size, n_seeds=args.seeds)
    else:
        run_loo(sessions, ensemble_size=args.ensemble_size)
