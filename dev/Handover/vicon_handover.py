from pathlib import Path

import intprim
import matplotlib.pyplot as plt
import numpy as np

from dev.util.pose_helper import rotation_dim_reduction, rotation_dim_reduction_continuous
from dev.util.process import (
    apply_local_axis_rotation_offset_to_quats,
    apply_reflection_to_positions,
    apply_reflection_to_quats,
    compute_cutoff,
    compute_euclidean_distance,
    csv_to_dict,
    ensure_quaternion_hemisphere_continuity,
    get_interaction_start_indices,
    get_traj_start_indices,
    mirror_trajectory_quats_using_head_plane,
    rebase_to_head_midpoint_floor_yaw_quat,
    segment_trajectories,
)
from dev.util.visualize import evaluate_6d_matplotlib

if __name__ == "__main__":
    np.random.seed(213413414)
    data_date = "2026_05_8"
    data_dir = Path(__file__).parent / data_date
    # Import data from csv files in the data directory and store in a dictionary of numpy arrays.
    data_ = csv_to_dict(data_dir)

    start_indices_ = {}
    for name in data_:
        start_indices_[name] = get_traj_start_indices(data_[name])
    

    trajectories_ = segment_trajectories(data_, start_indices_, print_info=False)

    controlled_agent = "Baton"
    observed_agent = "Receiver_Hand"
    controlled_general_pose = "Hat_Giver"
    observed_general_pose = "Hat_Receiver"

    mirror_to_left_handed = True
    apply_post_mirror_z_pi = True  # Apply constant local-axis rotation offset to both agents
    post_mirror_offset_axis = "z"  # only used when apply_post_mirror_z_pi=True
    post_mirror_offset_angle = np.pi - np.pi/16
    rebase_to_head_midpoint_floor = True

    rebase_yaw_mode = "legacy_euler"  # or: "projected_forward"
    rebase_body_forward_axis = "x"    # used only when rebase_yaw_mode == "projected_forward"

    continuous_rotvec = True

    debug_axes = False

    # Plot the trajectories of the controlled and observed agents for each trajectory segment.
    if False:
        # for i in range(0, len(start_indices_[controlled_agent])):
        for i in range(0,3):
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
    if True:
        training_trajectories = []
        training_ctrl_quats = []
        testing_trajectories = []
        n_test_trajectories = 5
        for i in range(0, len(approach_cutoff_indices)):
            if i == 39:
                continue
            time_vec = combined_data[f"{i}"][:, 0]
            # Keep rotations as quaternions through all geometric transforms.
            ctrl_pos = combined_data[f"{i}"][:, 1:4]
            ctrl_quat = combined_data[f"{i}"][:, 4:8]   # (qw,qx,qy,qz)
            obs_pos = combined_data[f"{i}"][:, 8:11]
            obs_quat = combined_data[f"{i}"][:, 11:15]  # (qw,qx,qy,qz)

            ctrl_head_pos = combined_data[f"{i}"][:, 15:18]
            obs_head_pos = combined_data[f"{i}"][:, 22:25]
            obs_head_quat = combined_data[f"{i}"][:, 25:29]  # (qw,qx,qy,qz) yaw source

            if debug_axes and i == 0:
                from scipy.spatial.transform import Rotation as _R
                R0 = _R.from_quat(np.array([ctrl_quat[0, 1], ctrl_quat[0, 2], ctrl_quat[0, 3], ctrl_quat[0, 0]])).as_matrix()
                print("[dbg] raw ctrl axes (world): x,y,z =", R0[:, 0], R0[:, 1], R0[:, 2])

            if mirror_to_left_handed:
                ctrl_pos, ctrl_quat, obs_pos, obs_quat, (S, p_plane, n_plane) = mirror_trajectory_quats_using_head_plane(
                    ctrl_pos,
                    ctrl_quat,
                    obs_pos,
                    obs_quat,
                    controlled_head_positions=ctrl_head_pos,
                    observed_head_positions=obs_head_pos,
                    up=(0.0, 0.0, 1.0),
                    return_transform=True,
                )

                # Mirror head streams consistently (for anchor computation).
                ctrl_head_pos = apply_reflection_to_positions(ctrl_head_pos, S=S, p0=p_plane)
                obs_head_pos = apply_reflection_to_positions(obs_head_pos, S=S, p0=p_plane)
                obs_head_quat = apply_reflection_to_quats(obs_head_quat, S=S)

                if debug_axes and i == 0:
                    from scipy.spatial.transform import Rotation as _R
                    R0 = _R.from_quat(np.array([ctrl_quat[0, 1], ctrl_quat[0, 2], ctrl_quat[0, 3], ctrl_quat[0, 0]])).as_matrix()
                    print("[dbg] mirrored ctrl axes (world): x,y,z =", R0[:, 0], R0[:, 1], R0[:, 2])

                

            if rebase_to_head_midpoint_floor:
                ctrl_pos, ctrl_quat, obs_pos, obs_quat = rebase_to_head_midpoint_floor_yaw_quat(
                    ctrl_pos,
                    ctrl_quat,
                    obs_pos,
                    obs_quat,
                    controlled_head_positions=ctrl_head_pos,
                    observed_head_positions=obs_head_pos,
                    observed_head_quats=obs_head_quat,
                    yaw_mode=rebase_yaw_mode,
                    body_forward_axis=rebase_body_forward_axis,
                    midpoint_time="mean",
                    floor_z=0.0,
                )

                if debug_axes and i == 0:
                    from scipy.spatial.transform import Rotation as _R
                    R0 = _R.from_quat(np.array([ctrl_quat[0, 1], ctrl_quat[0, 2], ctrl_quat[0, 3], ctrl_quat[0, 0]])).as_matrix()
                    print("[dbg] rebased ctrl axes (world): x,y,z =", R0[:, 0], R0[:, 1], R0[:, 2])

            if apply_post_mirror_z_pi:
                ctrl_quat = apply_local_axis_rotation_offset_to_quats(
                    ctrl_quat,
                    axis=post_mirror_offset_axis,
                    angle_rad=post_mirror_offset_angle,
                )

                obs_quat = apply_local_axis_rotation_offset_to_quats(
                    obs_quat,
                    axis=post_mirror_offset_axis,
                    angle_rad=post_mirror_offset_angle,
                )

                if debug_axes and i == 0:
                    from scipy.spatial.transform import Rotation as _R

                    R0 = _R.from_quat(
                        np.array([ctrl_quat[0, 1], ctrl_quat[0, 2], ctrl_quat[0, 3], ctrl_quat[0, 0]])
                    ).as_matrix()
                    print("[dbg] post-offset ctrl axes (world): x,y,z =", R0[:, 0], R0[:, 1], R0[:, 2])

            # Enforce quaternion sign continuity before converting to rotvec.
            ctrl_quat = ensure_quaternion_hemisphere_continuity(ctrl_quat)
            obs_quat = ensure_quaternion_hemisphere_continuity(obs_quat)

            # Final EBIP input: 12D (pos + rotvec) per timestep.
            training_data = np.zeros((ctrl_pos.shape[0], 12))
            training_data[:, 0:3] = ctrl_pos
            training_data[:, 3:6] = rotation_dim_reduction_continuous(ctrl_quat) if continuous_rotvec else rotation_dim_reduction(ctrl_quat)
            training_data[:, 6:9] = obs_pos
            training_data[:, 9:12] = rotation_dim_reduction_continuous(obs_quat) if continuous_rotvec else rotation_dim_reduction(obs_quat)
            if i <= len(approach_cutoff_indices) - n_test_trajectories - 1:
                training_trajectories.append(training_data.T) # (12, T) for EBIP convention
            else:
                # Store time alongside the 12 DoFs for convenient export/debug.
                # Shape: (13, T) = [time; 12 DoFs]
                testing_trajectories.append(np.vstack((time_vec[None, :], training_data.T)))

            training_ctrl_quats.append(ctrl_quat[0,:])

        # Visualize the trajectories
        # for i in range(0, len(approach_cutoff_indices)):
        #     visualize_pose_trajectories_matplotlib(combined_data[f"{i}"][:, 0], training_trajectories[i][:,:6], training_trajectories[i][:,6:12])

        mean_ctrl_start = np.zeros(3)
        mean_ctrl_start[0] = np.mean([traj[0, 0] for traj in training_trajectories])
        mean_ctrl_start[1] = np.mean([traj[1, 0] for traj in training_trajectories])
        mean_ctrl_start[2] = np.mean([traj[2, 0] for traj in training_trajectories])
        print(f"Mean starting position of controlled agent across training trajectories: {mean_ctrl_start}")
        print(f"+X axis forward from controlled agent's starting position")

        # print(f"Starting quat shape of entire list: {np.array(training_ctrl_quats).shape}")
        mean_ctrl_quat_start = np.mean(training_ctrl_quats, axis=0)
        print(f"Mean starting quaternion of controlled agent across training trajectories: {mean_ctrl_quat_start} (qw, qx, qy, qz)")



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

        # basis_model_sigmoidal = intprim.basis.SigmoidalModel(11, 0.01, dof_names)
        basis_model_gaussian = intprim.basis.GaussianModel(5, 0.19, dof_names)
        

        primitive = intprim.BayesianInteractionPrimitive(basis_model_gaussian)

        for trajectory in training_trajectories:
            primitive.add_demonstration(trajectory)

        mean, upper_bound, lower_bound = primitive.get_probability_distribution()
        # intprim.util.visualization.plot_distribution(dof_names, mean, upper_bound, lower_bound)

        # Export model + test_trajectory csv for evaluation in ros2
        model_dir = Path(__file__).parent / "models"
        model_dir.mkdir(exist_ok=True)
        model_file = model_dir / "mirrored_handover_2026_05_8.bip"
        # primitive.export_data(model_file)

        test_save_index = 3
        # testing_trajectories entries are (13, T): [time; 12 DoFs]
        test_trajectory_save = testing_trajectories[test_save_index].T
        test_trajectory_file = model_dir / "mirrored_handover_test_trajectory.csv"
        # np.savetxt(test_trajectory_file, test_trajectory_save, delimiter=",", header="Time, Controlled_X, Controlled_Y, Controlled_Z, Controlled_RX, Controlled_RY, Controlled_RZ, Observed_X, Observed_Y, Observed_Z, Observed_RX, Observed_RY, Observed_RZ", comments="")

        observation_noise = np.diag(selection.get_model_mse(basis_model_gaussian, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])))
        # mat = observation_noise
        # print(np.array2string(mat.diagonal(), formatter={"float_kind": lambda x: f"{x:.7f}"}))
        phase_velocity_mean, phase_velocity_var = intprim.examples.get_phase_stats(training_trajectories)
        # print(f"Phase velocity mean: {phase_velocity_mean}, Phase velocity variance: {phase_velocity_var:.7f}")

        #Define a filter to use. Here we use an ensemble Kalman filter
        filter = intprim.filter.spatiotemporal.EnsembleKalmanFilter(
        basis_model = basis_model_gaussian,
        initial_phase_mean = [0.0, phase_velocity_mean],
        initial_phase_var = [1e-4, phase_velocity_var],
        proc_var = 1e-8,
        initial_ensemble = primitive.basis_weights)

        # Strip time row before evaluation; evaluate_6d_matplotlib expects (12, T).
        evaluate_6d_matplotlib(primitive, filter, testing_trajectories[3][1:, :], observation_noise)




    plt.show()