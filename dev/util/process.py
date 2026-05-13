import numpy as np


def _normalize_vector(vec, eps=1e-12):
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return None
    return vec / norm


def _pick_nonparallel_axis(vec):
    """Pick a basis axis that's not (nearly) parallel to vec."""
    vec = np.asarray(vec, dtype=float)
    candidates = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    best_axis = candidates[0]
    best_score = -1.0
    for axis in candidates:
        score = float(np.linalg.norm(np.cross(vec, axis)))
        if score > best_score:
            best_score = score
            best_axis = axis
    return best_axis


def compute_mirror_transform_from_heads(
    controlled_head_positions,
    observed_head_positions,
    up=(0.0, 0.0, 1.0),
):
    """Compute a reflection transform (S, p0) from two head trajectories.

    The mirror plane is chosen as the vertical plane spanned by the head-to-head
    vector and the global up vector.

    Args:
        controlled_head_positions: (T, 3) array of controlled agent head positions.
        observed_head_positions: (T, 3) array of observed agent head positions.
        up: Global up direction (3,). Defaults to +Z.

    Returns:
        S: (3, 3) reflection matrix, S = I - 2 n n^T (det(S) = -1)
        p0: (3,) point on the plane (midpoint between mean head positions)
        n: (3,) unit normal of the plane
    """
    controlled_head_positions = np.asarray(controlled_head_positions, dtype=float)
    observed_head_positions = np.asarray(observed_head_positions, dtype=float)

    if controlled_head_positions.ndim != 2 or controlled_head_positions.shape[1] != 3:
        raise ValueError(f"Expected controlled_head_positions shape (T,3); got {controlled_head_positions.shape}")
    if observed_head_positions.ndim != 2 or observed_head_positions.shape[1] != 3:
        raise ValueError(f"Expected observed_head_positions shape (T,3); got {observed_head_positions.shape}")

    p_ctrl = np.mean(controlled_head_positions, axis=0)
    p_obs = np.mean(observed_head_positions, axis=0)

    v = p_obs - p_ctrl
    up = np.asarray(up, dtype=float)
    up_unit = _normalize_vector(up)
    if up_unit is None:
        raise ValueError("Up vector has near-zero norm; cannot define mirror plane")

    n = np.cross(v, up_unit)
    n_unit = _normalize_vector(n)

    if n_unit is None:
        axis = _pick_nonparallel_axis(v)
        n_unit = _normalize_vector(np.cross(v, axis))
        if n_unit is None:
            raise ValueError("Head positions are degenerate; cannot compute a stable mirror plane")

    p0 = 0.5 * (p_ctrl + p_obs)
    S = np.eye(3) - 2.0 * np.outer(n_unit, n_unit)
    return S, p0, n_unit


def _axis_letter_to_index(axis):
    if isinstance(axis, str):
        axis_l = axis.lower()
        if axis_l == "x":
            return 0
        if axis_l == "y":
            return 1
        if axis_l == "z":
            return 2
    raise ValueError(f"axis must be one of 'x','y','z'; got {axis!r}")


def _extract_global_yaw_from_rotation_matrix(
    R_body_to_world,
    *,
    mode="legacy_euler",
    body_forward_axis="x",
):
    """Extract a global +Z yaw angle from a rotation matrix.

    Args:
        R_body_to_world: (3,3) rotation matrix.
        mode:
            - 'legacy_euler': yaw = as_euler('zyx')[0]
            - 'projected_forward': yaw from the chosen body forward axis projected onto world XY
        body_forward_axis: 'x','y', or 'z' (used only for 'projected_forward')

    Returns:
        yaw angle (radians)
    """
    try:
        from scipy.spatial.transform import Rotation
    except Exception as exc:
        raise ImportError("SciPy is required to extract yaw") from exc

    R_body_to_world = np.asarray(R_body_to_world, dtype=float)
    if R_body_to_world.shape != (3, 3):
        raise ValueError(f"Expected R_body_to_world shape (3,3); got {R_body_to_world.shape}")

    mode_l = str(mode).lower()
    if mode_l in ("legacy", "legacy_euler", "euler"):
        return float(Rotation.from_matrix(R_body_to_world).as_euler("zyx", degrees=False)[0])

    if mode_l in ("projected_forward", "forward", "heading"):
        axis_idx = _axis_letter_to_index(body_forward_axis)
        forward_world = R_body_to_world[:, axis_idx]
        norm_xy = float(np.hypot(forward_world[0], forward_world[1]))
        if norm_xy < 1e-12:
            return float(Rotation.from_matrix(R_body_to_world).as_euler("zyx", degrees=False)[0])
        return float(np.arctan2(forward_world[1], forward_world[0]))

    raise ValueError(f"Unknown yaw extraction mode: {mode!r}")


def _rotation_from_quat_scalar_first(quats, Rotation):
    """Create a SciPy Rotation from scalar-first quaternions.

    Args:
        quats: shape (4,) or (T,4) in (w,x,y,z) order.
        Rotation: scipy.spatial.transform.Rotation class.
    """
    quats = np.asarray(quats, dtype=float)
    try:
        return Rotation.from_quat(quats, scalar_first=True)
    except TypeError:
        if quats.ndim == 1:
            quat_xyzw = np.array([quats[1], quats[2], quats[3], quats[0]], dtype=float)
        else:
            quat_xyzw = np.column_stack((quats[:, 1], quats[:, 2], quats[:, 3], quats[:, 0]))
        return Rotation.from_quat(quat_xyzw)


def _quat_scalar_first_from_rotation(rot, Rotation):
    """Convert a SciPy Rotation to scalar-first quaternions."""
    try:
        return rot.as_quat(scalar_first=True)
    except TypeError:
        quat_xyzw = rot.as_quat()
        quat_xyzw = np.asarray(quat_xyzw, dtype=float)
        if quat_xyzw.ndim == 1:
            return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)
        return np.column_stack((quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]))


def ensure_quaternion_hemisphere_continuity(quats_scalar_first):
    """Flip quaternion signs over time to avoid discontinuities.

    Ensures consecutive quaternions satisfy dot(q[t], q[t-1]) >= 0.

    Args:
        quats_scalar_first: (T,4) in (w,x,y,z)

    Returns:
        (T,4) quaternions with sign continuity.
    """
    quats = np.asarray(quats_scalar_first, dtype=float)
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(f"Expected quats shape (T,4); got {quats.shape}")
    if quats.shape[0] <= 1:
        return quats

    fixed = np.array(quats, copy=True)
    for i in range(1, fixed.shape[0]):
        if float(np.dot(fixed[i], fixed[i - 1])) < 0.0:
            fixed[i] *= -1.0
    return fixed


def mirror_trajectory_quats_using_head_plane(
    controlled_positions,
    controlled_quats,
    observed_positions,
    observed_quats,
    *,
    controlled_head_positions,
    observed_head_positions,
    up=(0.0, 0.0, 1.0),
    return_transform=False,
):
    """Mirror two pose streams (pos + quaternion) using the head-derived plane.

    Quaternion convention is scalar-first (w,x,y,z).

    Returns mirrored (controlled_positions, controlled_quats, observed_positions, observed_quats)
    with the same shapes as inputs.
    """
    ctrl_pos = np.asarray(controlled_positions, dtype=float)
    obs_pos = np.asarray(observed_positions, dtype=float)
    ctrl_q = np.asarray(controlled_quats, dtype=float)
    obs_q = np.asarray(observed_quats, dtype=float)

    if ctrl_pos.shape != obs_pos.shape or ctrl_pos.ndim != 2 or ctrl_pos.shape[1] != 3:
        raise ValueError("controlled_positions and observed_positions must both have shape (T,3)")
    if ctrl_q.shape != obs_q.shape or ctrl_q.ndim != 2 or ctrl_q.shape[1] != 4:
        raise ValueError("controlled_quats and observed_quats must both have shape (T,4)")
    if ctrl_pos.shape[0] != ctrl_q.shape[0]:
        raise ValueError("positions and quats must have the same length")

    S, p0, n = compute_mirror_transform_from_heads(
        controlled_head_positions=controlled_head_positions,
        observed_head_positions=observed_head_positions,
        up=up,
    )

    ctrl_pos_m = (ctrl_pos - p0) @ S.T + p0
    obs_pos_m = (obs_pos - p0) @ S.T + p0

    try:
        from scipy.spatial.transform import Rotation
    except Exception as exc:
        raise ImportError("SciPy is required to mirror quaternions") from exc

    ctrl_rot = _rotation_from_quat_scalar_first(ctrl_q, Rotation)
    obs_rot = _rotation_from_quat_scalar_first(obs_q, Rotation)

    ctrl_mats = ctrl_rot.as_matrix()
    obs_mats = obs_rot.as_matrix()
    ctrl_mats_m = np.einsum("ij,tjk,kl->til", S, ctrl_mats, S)
    obs_mats_m = np.einsum("ij,tjk,kl->til", S, obs_mats, S)

    ctrl_q_m = _quat_scalar_first_from_rotation(Rotation.from_matrix(ctrl_mats_m), Rotation)
    obs_q_m = _quat_scalar_first_from_rotation(Rotation.from_matrix(obs_mats_m), Rotation)

    if return_transform:
        return ctrl_pos_m, ctrl_q_m, obs_pos_m, obs_q_m, (S, p0, n)
    return ctrl_pos_m, ctrl_q_m, obs_pos_m, obs_q_m


def apply_reflection_to_positions(positions, *, S, p0):
    """Apply a reflection transform about a plane/anchor point.

    Uses the same row-vector convention as `mirror_trajectory_quats_using_head_plane`:
        p' = (p - p0) @ S.T + p0

    Args:
        positions: (T,3) array
        S: (3,3) reflection matrix
        p0: (3,) point on the reflection plane

    Returns:
        (T,3) reflected positions
    """
    pos = np.asarray(positions, dtype=float)
    S = np.asarray(S, dtype=float)
    p0 = np.asarray(p0, dtype=float)

    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Expected positions shape (T,3); got {pos.shape}")
    if S.shape != (3, 3):
        raise ValueError(f"Expected S shape (3,3); got {S.shape}")
    if p0.shape != (3,):
        raise ValueError(f"Expected p0 shape (3,); got {p0.shape}")

    return (pos - p0) @ S.T + p0


def apply_reflection_to_quats(quats_scalar_first, *, S):
    """Apply a reflection transform to scalar-first quaternions.

    Mirrors a rotation matrix via: R' = S R S

    Args:
        quats_scalar_first: (T,4) array in (w,x,y,z)
        S: (3,3) reflection matrix

    Returns:
        (T,4) mirrored quaternions in (w,x,y,z)
    """
    q = np.asarray(quats_scalar_first, dtype=float)
    S = np.asarray(S, dtype=float)

    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected quats shape (T,4); got {q.shape}")
    if S.shape != (3, 3):
        raise ValueError(f"Expected S shape (3,3); got {S.shape}")
    if q.shape[0] == 0:
        return q

    try:
        from scipy.spatial.transform import Rotation
    except Exception as exc:
        raise ImportError("SciPy is required to mirror quaternions") from exc

    mats = _rotation_from_quat_scalar_first(q, Rotation).as_matrix()
    mats_m = np.einsum("ij,tjk,kl->til", S, mats, S)
    return _quat_scalar_first_from_rotation(Rotation.from_matrix(mats_m), Rotation)


def apply_local_axis_rotation_offset_to_quats(
    quats_scalar_first,
    *,
    axis="Z",
    angle_rad=np.pi,
):
    """Apply a constant local/body-fixed rotation about a local axis to quaternions.

    Applies a local/body-fixed rotation offset: R_new = R @ R_offset.

    Args:
        quats_scalar_first: (T,4) array in (w,x,y,z)
        axis: 'x', 'y', or 'z' local axis.
        angle_rad: rotation angle in radians.

    Returns:
        (T,4) adjusted quaternions in (w,x,y,z)
    """
    q = np.asarray(quats_scalar_first, dtype=float)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"Expected quats shape (T,4); got {q.shape}")

    axis_l = str(axis).lower()
    if axis_l not in ("x", "y", "z", "X", "Y", "Z"):
        raise ValueError(f"axis must be one of 'x','y','z','X','Y','Z'; got {axis!r}")

    try:
        from scipy.spatial.transform import Rotation
    except Exception as exc:
        raise ImportError("SciPy is required to apply quaternion rotation offsets") from exc

    rot = _rotation_from_quat_scalar_first(q, Rotation)
    mats = rot.as_matrix()
    R_offset = Rotation.from_euler(axis_l, float(angle_rad), degrees=False).as_matrix()
    mats_new = np.einsum("tij,jk->tik", mats, R_offset)
    return _quat_scalar_first_from_rotation(Rotation.from_matrix(mats_new), Rotation)


def rebase_to_head_midpoint_floor_yaw_quat(
    controlled_positions,
    controlled_quats,
    observed_positions,
    observed_quats,
    *,
    controlled_head_positions,
    observed_head_positions,
    observed_head_quats,
    yaw_mode="legacy_euler",
    body_forward_axis="x",
    midpoint_time="mean",
    floor_z=0.0,
    return_anchor=False,
):
    """Rebase pos+quat streams to a head-midpoint floor origin and observed-head yaw.

    - Origin: floor-projected midpoint between controlled/observed head positions.
    - Orientation: yaw-only about global +Z extracted from the observed head quaternion
      at the start of the segment.

    Quaternion convention is scalar-first (w,x,y,z).
    """
    ctrl_pos = np.asarray(controlled_positions, dtype=float)
    obs_pos = np.asarray(observed_positions, dtype=float)
    ctrl_q = np.asarray(controlled_quats, dtype=float)
    obs_q = np.asarray(observed_quats, dtype=float)

    ctrl_head_pos = np.asarray(controlled_head_positions, dtype=float)
    obs_head_pos = np.asarray(observed_head_positions, dtype=float)
    obs_head_q = np.asarray(observed_head_quats, dtype=float)

    if ctrl_pos.shape != obs_pos.shape or ctrl_pos.ndim != 2 or ctrl_pos.shape[1] != 3:
        raise ValueError("controlled_positions and observed_positions must both have shape (T,3)")
    if ctrl_q.shape != obs_q.shape or ctrl_q.ndim != 2 or ctrl_q.shape[1] != 4:
        raise ValueError("controlled_quats and observed_quats must both have shape (T,4)")
    if ctrl_pos.shape[0] != ctrl_q.shape[0]:
        raise ValueError("positions and quats must have the same length")
    if ctrl_head_pos.ndim != 2 or ctrl_head_pos.shape[1] != 3:
        raise ValueError("controlled_head_positions must have shape (T,3)")
    if obs_head_pos.ndim != 2 or obs_head_pos.shape[1] != 3:
        raise ValueError("observed_head_positions must have shape (T,3)")
    if obs_head_q.ndim != 2 or obs_head_q.shape[1] != 4:
        raise ValueError("observed_head_quats must have shape (T,4)")

    if ctrl_pos.shape[0] == 0:
        if return_anchor:
            return ctrl_pos, ctrl_q, obs_pos, obs_q, (np.zeros(3), np.eye(3), 0.0)
        return ctrl_pos, ctrl_q, obs_pos, obs_q

    midpoint_time_l = str(midpoint_time).lower()
    if midpoint_time_l not in ("mean",):
        raise ValueError(f"Unsupported midpoint_time: {midpoint_time!r} (expected 'mean')")

    p_ctrl = np.mean(ctrl_head_pos, axis=0) if ctrl_head_pos.shape[0] else np.zeros(3)
    p_obs = np.mean(obs_head_pos, axis=0) if obs_head_pos.shape[0] else np.zeros(3)
    p0 = 0.5 * (p_ctrl + p_obs)
    p0 = np.asarray(p0, dtype=float)
    p0[2] = float(floor_z)

    try:
        from scipy.spatial.transform import Rotation
    except Exception as exc:
        raise ImportError("SciPy is required to rebase quaternions") from exc

    if obs_head_q.shape[0] == 0:
        yaw0 = 0.0
        R_base = np.eye(3)
    else:
        R_head0 = _rotation_from_quat_scalar_first(obs_head_q[0], Rotation).as_matrix()
        yaw0 = _extract_global_yaw_from_rotation_matrix(
            R_head0,
            mode=yaw_mode,
            body_forward_axis=body_forward_axis,
        )
        R_base = Rotation.from_euler("z", float(yaw0), degrees=False).as_matrix()

    R_inv = R_base.T

    ctrl_pos_r = (ctrl_pos - p0) @ R_inv.T
    obs_pos_r = (obs_pos - p0) @ R_inv.T

    ctrl_mats = _rotation_from_quat_scalar_first(ctrl_q, Rotation).as_matrix()
    obs_mats = _rotation_from_quat_scalar_first(obs_q, Rotation).as_matrix()
    ctrl_mats_r = np.einsum("ij,tjk->tik", R_inv, ctrl_mats)
    obs_mats_r = np.einsum("ij,tjk->tik", R_inv, obs_mats)
    ctrl_q_r = _quat_scalar_first_from_rotation(Rotation.from_matrix(ctrl_mats_r), Rotation)
    obs_q_r = _quat_scalar_first_from_rotation(Rotation.from_matrix(obs_mats_r), Rotation)

    if return_anchor:
        return ctrl_pos_r, ctrl_q_r, obs_pos_r, obs_q_r, (p0, R_base, float(yaw0))
    return ctrl_pos_r, ctrl_q_r, obs_pos_r, obs_q_r


def csv_to_dict(file_path):
    """Import csv files from data directory and return a dictionary of numpy arrays."""
    traj_data = {}
    for file in file_path.glob("*.csv"):
        # print(f"Processing {file.name}...")
        prefix = "traj_vicon_"
        name = file.stem.removeprefix(prefix)
        print(f"Importing {name} data...")
        traj_data[name] = np.loadtxt(file, delimiter=",", skiprows=1)

    return traj_data


def get_traj_start_indices(traj_data, time_threshold=0.5):
    """Get the starting indices of trajectories based on time gaps."""
    start_indices = [0]  # Start with the first index
    time_prev = traj_data[0, 0]

    for i in range(1, len(traj_data)):
        if traj_data[i, 0] - time_prev > time_threshold:
            start_indices.append(i)
        time_prev = traj_data[i, 0]

    return start_indices


def segment_trajectories(traj_data, start_indices, print_info=True):
    """Segment trajectory data into a list of trajectories based on start indices."""
    trajectories_ = {}
    for name in traj_data:
        for i in range(0,len(start_indices[name])):
            trajectories_[f"{name}_{i}"] = traj_data[name][start_indices[name][i]:(start_indices[name][i+1]-1) if i+1 < len(start_indices[name]) else None]
            if (print_info):
                print(f"Trajectory {name}_{i} has {len(trajectories_[f'{name}_{i}'])} points.")

    return trajectories_


def compute_euclidean_distance(traj1, traj2):
    """Compute the Euclidean distance between two trajectories."""
    if len(traj1) != len(traj2):
        raise ValueError("Trajectories must have the same length.")
    
    return np.sqrt(np.sum((traj1 - traj2) ** 2, axis=1))


def compute_cutoff(
    euclidean_distance,
    steady_state_window=50,
    sigma=4.5,
    min_consecutive=5,
):
    """Compute the handover location cutoff for the approach-to-handover phase."""
    steady_state = euclidean_distance[-steady_state_window:]
    cutoff = np.mean(steady_state)  # Steady state after the receiver grasps the object.
    cutoff_margin = sigma * np.std(steady_state)

    consecutive = 0
    for i in range(len(euclidean_distance)):
        if euclidean_distance[i] < cutoff + cutoff_margin:
            consecutive += 1
            if consecutive >= min_consecutive:
                return i - min_consecutive + 1
        else:
            consecutive = 0

    return None


def get_interaction_start_indices(
    trajectory,
    steady_state_window=20,
    z_sigma=4.0,
    min_consecutive=5,
    direction="up",
):
    """Get the starting indices of interaction phases based on when the receiver's hand begins to move."""
    z_values = trajectory[:, 3]  # Assuming z-axis is vertical and trajectory is in the order [time, x, y, z]
    starting_point = np.mean(z_values[:steady_state_window])
    cutoff_margin = z_sigma * np.std(z_values[:steady_state_window])

    consecutive = 0
    for i in range(steady_state_window, len(z_values)):
        delta = z_values[i] - starting_point
        if direction == "up":
            triggered = delta > cutoff_margin
        elif direction == "down":
            triggered = delta < -cutoff_margin
        else:
            triggered = np.abs(delta) > cutoff_margin

        if triggered:
            consecutive += 1
            if consecutive >= min_consecutive:
                return i - min_consecutive + 1
        else:
            consecutive = 0

    return None