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

def compute_cutoff(euclidean_distance, cutoff_margin=0.05, steady_state_window=50):
    """Compute the handover location cutoff in order to process only the approach to handover phase of the interaction."""
    cutoff = np.mean(euclidean_distance[-steady_state_window]) # Since receiver will have grasped the object in the last frames, this can be used as the cutoff
    cutoff_margin = np.std(euclidean_distance[-steady_state_window])
    for i in range(len(euclidean_distance)):
        if euclidean_distance[i] < cutoff + cutoff_margin: 
            return i
    return None