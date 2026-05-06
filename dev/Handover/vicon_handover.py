from sys import prefix

from dev.util import *
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_date = "2026_04_27"
    data_dir = Path(__file__).parent / data_date
    # Import data from csv files in the data directory and store in a dictionary of numpy arrays.
    data_ = csv_to_dict(data_dir)

    start_indices_ = {}
    for name in data_:
        start_indices_[name] = get_traj_start_indices(data_[name])

    trajectories_ = segment_trajectories(data_, start_indices_, print_info=False)

    controlled_agent = "Baton"
    observed_agent = "Jacquie_Hand"
    controlled_general_pose = "Hat_Giver"
    observed_general_pose = "Hat_Receiver"

    # Plot the trajectories of the controlled and observed agents for each trajectory segment.
    if False:
        for i in range(0, len(start_indices_[controlled_agent])):
            plt.figure()
            plt.title(f"Trajectory {i+1} of {controlled_agent} and {observed_agent}")
            plt.plot(trajectories_[f"{controlled_agent}_{i}"][:, 0], trajectories_[f"{controlled_agent}_{i}"][:, 1], label=f"{controlled_agent} x")
            plt.plot(trajectories_[f"{controlled_agent}_{i}"][:, 0], trajectories_[f"{controlled_agent}_{i}"][:, 2], label=f"{controlled_agent} y")
            plt.plot(trajectories_[f"{controlled_agent}_{i}"][:, 0], trajectories_[f"{controlled_agent}_{i}"][:, 3], label=f"{controlled_agent} z")
            plt.plot(trajectories_[f"{observed_agent}_{i}"][:, 0], trajectories_[f"{observed_agent}_{i}"][:, 1], label=f"{observed_agent} x")
            plt.plot(trajectories_[f"{observed_agent}_{i}"][:, 0], trajectories_[f"{observed_agent}_{i}"][:, 2], label=f"{observed_agent} y")
            plt.plot(trajectories_[f"{observed_agent}_{i}"][:, 0], trajectories_[f"{observed_agent}_{i}"][:, 3], label=f"{observed_agent} z")
            plt.xlabel("Time (s)")
            plt.ylabel("Position (m)")
            plt.legend()
            plt.grid()
            plt.show()
    
    euclidean_distance = {}
    for i in range(0, len(start_indices_[controlled_agent])):
        euclidean_distance[f"{i}"] = compute_euclidean_distance(trajectories_[f"{controlled_agent}_{i}"][:, 1:4], trajectories_[f"{observed_agent}_{i}"][:, 1:4])

    # Compute the handover location cutoff for each trajectory segment in order to process only the approach to handover phase of the interaction.
    approach_cutoff_indices = []
    for i in range(0, len(start_indices_[controlled_agent])):
        if i == len(start_indices_[controlled_agent]) - 1: # Final trajectory for initial data collection had high noise and will be dismissed
            break
        approach_cutoff_indices.append(compute_cutoff(euclidean_distance[f"{i}"]))

    # Plot cutoff values with euclidean distance for each trajectory segment.
    if False:
        for i in range(0, len(approach_cutoff_indices)):
            plt.figure()
            plt.title(f"Euclidean Distance between {controlled_agent} and {observed_agent} for Trajectory {i+1}")
            plt.plot(trajectories_[f"{controlled_agent}_{i}"][:, 0], euclidean_distance[f"{i}"], label="Euclidean Distance")
            plt.vlines(x=trajectories_[f"{controlled_agent}_{i}"][approach_cutoff_indices[i], 0], ymin= min(euclidean_distance[f"{i}"]), ymax=max(euclidean_distance[f"{i}"]), color="red", linestyle="--", label="Handover Location Cutoff")
            plt.xlabel("Time (s)")
            plt.ylabel("Distance (m)")
            plt.legend()
            plt.grid()  
            plt.show()

    # If it is desired to save all objects for each trajectory into combined csv files. If not, can train ebip directly following
    if False:
        for i in range(0, len(approach_cutoff_indices)):
            combined_data = np.zeros((approach_cutoff_indices[i], 1 + 7*len(data_))) # Time + 7 values (position and orientation) for each agent
            combined_data[:,0] = trajectories_[f"{controlled_agent}_{i}"][:approach_cutoff_indices[i], 0] # Time
            suffix = f"_{i}"
            for name in trajectories_:
                if name.endswith(suffix):
                    if name.removesuffix(suffix) == controlled_agent:
                        combined_data[:, 1:8] = trajectories_[name][:approach_cutoff_indices[i], 1:8] # Position and orientation of controlled agent

                    elif name.removesuffix(suffix) == observed_agent:
                        combined_data[:, 8:15] = trajectories_[name][:approach_cutoff_indices[i], 1:8] # Position and orientation of observed agent

                    elif name.removesuffix(suffix) == controlled_general_pose:
                        combined_data[:, 15:22] = trajectories_[name][:approach_cutoff_indices[i], 1:8] # Position and orientation of controlled agent's general pose

                    elif name.removesuffix(suffix) == observed_general_pose:
                        combined_data[:, 22:29] = trajectories_[name][:approach_cutoff_indices[i], 1:8] # Position and orientation of observed agent's general pose

            zero_cols = np.all(combined_data[:, 1:] == 0.0, axis=0)
            if np.any(zero_cols):
                print(f"Warning: Trajectory {i+1} has {np.sum(zero_cols)} all-zero pose columns.")

            np.savetxt(data_dir / "processed" / f"processed_traj_{i+1}.csv", combined_data, delimiter=",", header="Time,Controlled_X,Controlled_Y,Controlled_Z,Controlled_Qw,Controlled_Qx,Controlled_Qy,Controlled_Qz,Observed_X,Observed_Y,Observed_Z,Observed_Qw,Observed_Qx,Observed_Qy,Observed_Qz,Controlled_General_X,Controlled_General_Y,Controlled_General_Z,Controlled_General_Qw,Controlled_General_Qx,Controlled_General_Qy,Controlled_General_Qz,Observed_General_X,Observed_General_Y,Observed_General_Z,Observed_General_Qw,Observed_General_Qx,Observed_General_Qy,Observed_General_Qz", comments="")

    
    