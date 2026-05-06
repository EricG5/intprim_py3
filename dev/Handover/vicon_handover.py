from sys import prefix

from dev.util import *
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import intprim

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
    
    # for i in range(0, len(approach_cutoff_indices)):
    #     plt.figure()
    #     plt.plot(trajectories_[f"{observed_agent}_0"][:approach_cutoff_indices[0], 0], trajectories_[f"{observed_agent}_0"][:approach_cutoff_indices[0], 3], label=f"{observed_agent} z")
    #     plt.show()

    interaction_start_indices = []
    for i in range(0, len(approach_cutoff_indices)):
        interaction_start_indices.append(get_interaction_start_indices(trajectories_[f"{observed_agent}_{i}"][:approach_cutoff_indices[i]], steady_state_window=200, min_consecutive=50, direction="up"))
    #     plt.figure()
    #     plt.plot(trajectories_[f"{observed_agent}_{i}"][:approach_cutoff_indices[i], 0], trajectories_[f"{observed_agent}_{i}"][:approach_cutoff_indices[i], 3], label=f"{observed_agent} z")
    #     plt.vlines(x=trajectories_[f"{observed_agent}_{i}"][interaction_start_indices[i], 0], ymin= min(trajectories_[f"{observed_agent}_{i}"][:approach_cutoff_indices[i], 3]), ymax=max(trajectories_[f"{observed_agent}_{i}"][:approach_cutoff_indices[i], 3]), color="red", linestyle="--", label="Interaction Start")
    #     plt.show()
        
    # print(f"Interaction start indices: {interaction_start_indices}")

    


    combined_data = {}
    for i in range(0, len(approach_cutoff_indices)):
        combined_data[f"{i}"] = np.zeros((approach_cutoff_indices[i]-interaction_start_indices[i], 1 + 7*len(data_))) # Time + 7 values (position and orientation) for each agent
        combined_data[f"{i}"][:,0] = trajectories_[f"{controlled_agent}_{i}"][interaction_start_indices[i]:approach_cutoff_indices[i], 0] # Time
        suffix = f"_{i}"
        for name in trajectories_:
            if name.endswith(suffix):
                if name.removesuffix(suffix) == controlled_agent:
                    combined_data[f"{i}"][:, 1:8] = trajectories_[name][interaction_start_indices[i]:approach_cutoff_indices[i], 1:8] # Position and orientation of controlled agent

                elif name.removesuffix(suffix) == observed_agent:
                    combined_data[f"{i}"][:, 8:15] = trajectories_[name][interaction_start_indices[i]:approach_cutoff_indices[i], 1:8] # Position and orientation of observed agent

                elif name.removesuffix(suffix) == controlled_general_pose:
                    combined_data[f"{i}"][:, 15:22] = trajectories_[name][interaction_start_indices[i]:approach_cutoff_indices[i], 1:8] # Position and orientation of controlled agent's general pose

                elif name.removesuffix(suffix) == observed_general_pose:
                    combined_data[f"{i}"][:, 22:29] = trajectories_[name][interaction_start_indices[i]:approach_cutoff_indices[i], 1:8] # Position and orientation of observed agent's general pose

        zero_cols = np.all(combined_data[f"{i}"][:, 1:] == 0.0, axis=0)
        if np.any(zero_cols):
            print(f"Warning: Trajectory {i+1} has {np.sum(zero_cols)} all-zero pose columns.")

        # Save the combined data if required, data can be directly processed below.
        if False:
            np.savetxt(data_dir / "processed" / f"processed_traj_{i+1}.csv", combined_data[f"{i}"], delimiter=",", header="Time,Controlled_X,Controlled_Y,Controlled_Z,Controlled_Qw,Controlled_Qx,Controlled_Qy,Controlled_Qz,Observed_X,Observed_Y,Observed_Z,Observed_Qw,Observed_Qx,Observed_Qy,Observed_Qz,Controlled_General_X,Controlled_General_Y,Controlled_General_Z,Controlled_General_Qw,Controlled_General_Qx,Controlled_General_Qy,Controlled_General_Qz,Observed_General_X,Observed_General_Y,Observed_General_Z,Observed_General_Qw,Observed_General_Qx,Observed_General_Qy,Observed_General_Qz", comments="")

    ## eBIP ##
    
    training_trajectories = []
    for i in range(0, len(approach_cutoff_indices)):
        training_data = np.zeros((combined_data[f"{i}"].shape[0], 12)) # 6 values (position and orientation) for each agent
        training_data[:, :3] = combined_data[f"{i}"][:, 1:4] # Position of controlled agent
        training_data[:, 3:6] = rotation_dim_reduction(combined_data[f"{i}"][:, 4:8]) # Orientation of controlled agent
        training_data[:, 6:9] = combined_data[f"{i}"][:, 8:11] # Position of observed agent
        training_data[:, 9:12] = rotation_dim_reduction(combined_data[f"{i}"][:, 11:15]) # Orientation of observed agent
        training_trajectories.append(training_data.T)
    

    # Visualize the trajectories
    # for i in range(0, len(approach_cutoff_indices)):
    #     visualize_pose_trajectories_matplotlib(combined_data[f"{i}"][:, 0], training_trajectories[i][:,:6], training_trajectories[i][:,6:12])


    # Define DOFs: 12 total — 6 per agent (pos + rotvec)
    dof_names = np.array([
        "X (Controlled)", "Y (Controlled)", "Z (Controlled)",
        "RX (Controlled)", "RY (Controlled)", "RZ (Controlled)",
        "X (Observed)", "Y (Observed)", "Z (Observed)",
        "RX (Observed)", "RY (Observed)", "RZ (Observed)",
    ])

    selection = intprim.basis.Selection(dof_names)

    for trajectory in training_trajectories:
        selection.add_demonstration(trajectory)

    # aic, bic = selection.get_information_criteria(np.array([0, 1], dtype = np.int32))
    # selection.get_best_model(aic, bic)

    basis_model_sigmoidal = intprim.basis.SigmoidalModel(11, 0.01, dof_names)
    

    primitive = intprim.BayesianInteractionPrimitive(basis_model_sigmoidal)

    for trajectory in training_trajectories:
        primitive.add_demonstration(trajectory)

    mean, upper_bound, lower_bound = primitive.get_probability_distribution()
    # intprim.util.visualization.plot_distribution(dof_names, mean, upper_bound, lower_bound)

    # Export model - TO DO

    observation_noise = np.diag(selection.get_model_mse(basis_model_sigmoidal, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])))
    phase_velocity_mean, phase_velocity_var = intprim.examples.get_phase_stats(training_trajectories)
    # print(f"Phase velocity mean: {phase_velocity_mean}, Phase velocity variance: {phase_velocity_var}")

    #Define a filter to use. Here we use an ensemble Kalman filter
    filter = intprim.filter.spatiotemporal.EnsembleKalmanFilter(
    basis_model = basis_model_sigmoidal,
    initial_phase_mean = [0.0, phase_velocity_mean],
    initial_phase_var = [1e-4, phase_velocity_var],
    proc_var = 1e-8,
    initial_ensemble = primitive.basis_weights)

    evaluate_6d_matplotlib(primitive, filter, training_trajectories[1], observation_noise)




    plt.show()