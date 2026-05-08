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


def mirror_12d_trajectory_using_head_plane(
    trajectory_12d,
    controlled_head_positions,
    observed_head_positions,
    up=(0.0, 0.0, 1.0),
    pos_slices=(slice(0, 3), slice(6, 9)),
    rotvec_slices=(slice(3, 6), slice(9, 12)),
    return_transform=False,
):
    """Mirror a 12D (pos+rotvec per agent) trajectory using a head-derived plane.

    Mirrors ONLY the controlled+observed agents (the 12D trajectory) and uses the
    head trajectories only to define the mirror plane.

    Args:
        trajectory_12d: (T, 12) or (12, T) array with DOFs:
            ctrl pos [0:3], ctrl rotvec [3:6], obs pos [6:9], obs rotvec [9:12].
        controlled_head_positions: (T, 3) head positions for the controlled agent.
        observed_head_positions: (T, 3) head positions for the observed agent.
        up: Global up direction.
        pos_slices: Two slices for the position columns.
        rotvec_slices: Two slices for the rotation-vector columns.
        return_transform: If True, also returns (S, p0, n).

    Returns:
        mirrored_traj: same shape as input trajectory_12d
        (optional) (S, p0, n)
    """
    traj = np.asarray(trajectory_12d, dtype=float)
    transposed = False
    if traj.ndim != 2:
        raise ValueError(f"Expected a 2D array; got shape {traj.shape}")
    if traj.shape[1] == 12:
        traj_t = traj
    elif traj.shape[0] == 12:
        traj_t = traj.T
        transposed = True
    else:
        raise ValueError(f"Expected shape (T,12) or (12,T); got {traj.shape}")

    S, p0, n = compute_mirror_transform_from_heads(
        controlled_head_positions=controlled_head_positions,
        observed_head_positions=observed_head_positions,
        up=up,
    )

    mirrored = np.array(traj_t, copy=True)

    # Mirror positions: p' = p0 + S (p - p0)
    for pos_slice in pos_slices:
        P = mirrored[:, pos_slice]
        mirrored[:, pos_slice] = (P - p0) @ S.T + p0

    # Mirror orientations: R' = S R S  (where R from rotvec)
    try:
        from scipy.spatial.transform import Rotation
    except Exception as exc:
        raise ImportError("SciPy is required to mirror rotation vectors") from exc

    for rot_slice in rotvec_slices:
        rotvec = mirrored[:, rot_slice]  # (T, 3)
        rot_mats = Rotation.from_rotvec(rotvec).as_matrix()  # (T, 3, 3)
        rot_mats_m = np.einsum("ij,tjk,kl->til", S, rot_mats, S)
        rotvec_m = Rotation.from_matrix(rot_mats_m).as_rotvec()
        mirrored[:, rot_slice] = rotvec_m

    if transposed:
        mirrored = mirrored.T

    if return_transform:
        return mirrored, (S, p0, n)
    return mirrored

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