import numpy as np

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