import time
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.spatial.transform import Rotation as R

try:
	import pyvista as pv
except ImportError:
	pv = None


def _require_pyvista():
	if pv is None:
		raise ImportError("pyvista is required for visualization. Install with 'pip install pyvista'.")


def _mean_dt(timestamps, n=20, default=0.004):
	timestamps = np.asarray(timestamps)
	n = min(int(n), len(timestamps))
	if n >= 2:
		return float(np.mean(np.diff(timestamps[:n])))
	return float(default)


def visualize_pose_trajectory(
	pose_matrix,
	pause_secs=0.004,
	title="Pose trajectory",
	*,
	step=1,
	realtime=True,
	color="dodgerblue",
	line_width=3,
	point_size=12,
):
	"""Visualize a timestamped pose trajectory in 3D using PyVista.

	Args:
		pose_matrix: (T, 8) array where columns are:
					 [timestamp, x, y, z, qw, qx, qy, qz].
		pause_secs: Minimum seconds per frame (fallback).
		step:        Decimation factor.
		realtime:    If True, pace using timestamps when possible.
		title:       Window title.
	"""
	visualize_pose_trajectories(
		[pose_matrix],
		pause_secs=pause_secs,
		title=title,
		step=step,
		realtime=realtime,
		colors=[color],
		line_width=line_width,
		point_size=point_size,
	)


def visualize_pose_trajectories(
	pose_matrices,
	pause_secs=0.004,
	title="Pose trajectories",
	*,
	step=1,
	realtime=True,
	colors=None,
	line_width=3,
	point_size=12,
):
	"""Visualize multiple timestamped pose trajectories in 3D using PyVista.

	Trajectories are advanced by index (frame 0, 1, 2, ...) and rendered together.
	If `realtime` is enabled, timing is based on the first trajectory's timestamps.

	Args:
		pose_matrices: Iterable of (T, 8) arrays.
		pause_secs: Minimum seconds per frame (fallback); replaced by mean dt of the
			first 20 samples of the first trajectory when possible.
		step: Decimation factor.
		realtime: If True, pace using timestamps when possible.
	"""
	_require_pyvista()
	if pose_matrices is None:
		raise ValueError("pose_matrices must be a non-empty iterable")

	step = max(int(step), 1)
	pose_matrices = list(pose_matrices)
	if not pose_matrices:
		raise ValueError("pose_matrices must be a non-empty iterable")

	if colors is None:
		colors = ["dodgerblue", "orange", "limegreen", "violet", "tomato"]

	positions_list = []
	timestamps_list = []
	for pm in pose_matrices:
		pm = np.asarray(pm)
		if pm.ndim != 2 or pm.shape[1] != 8:
			raise ValueError("each pose_matrix must have shape (T, 8)")
		timestamps_list.append(pm[::step, 0])
		positions_list.append(pm[::step, 1:4])

	timestamps_ref = timestamps_list[0]
	pause_secs = _mean_dt(timestamps_ref, n=20, default=pause_secs)

	plotter = pv.Plotter()
	plotter.add_axes()
	plotter.add_title(title)

	point_actors = []
	for idx, positions in enumerate(positions_list):
		color = colors[idx % len(colors)]
		traj_line = pv.lines_from_points(positions)
		plotter.add_mesh(traj_line, color=color, line_width=line_width)
		point_actor = plotter.add_mesh(pv.PolyData(positions[:1]), color=color, point_size=point_size)
		point_actors.append(point_actor)

	plotter.reset_camera()
	plotter.show(auto_close=False, interactive_update=True)

	max_len = max(p.shape[0] for p in positions_list)
	for i in range(max_len):
		frame_start = time.perf_counter()
		for traj_idx, positions in enumerate(positions_list):
			if i < positions.shape[0]:
				point_actors[traj_idx].mapper.SetInputData(pv.PolyData(positions[i].reshape(1, 3)))

		plotter.update()

		if not realtime:
			continue

		if i + 1 < len(timestamps_ref):
			desired = max(pause_secs, float(timestamps_ref[i + 1] - timestamps_ref[i]))
		else:
			desired = pause_secs

		elapsed = time.perf_counter() - frame_start
		to_sleep = desired - elapsed
		if to_sleep > 0.0:
			time.sleep(to_sleep)

	plotter.close()


def visualize_pose_trajectories_matplotlib(
	timestamps,
	pose_a,
	pose_b,
	*,
	step=1,
	realtime=True,
	colors=("dodgerblue", "orange"),
	line_width=2.0,
	point_size=40,
	arrow_scale=0.1,
):
	"""Visualize two pose trajectories with Matplotlib (3D).

	Args:
		timestamps: (T,) array of timestamps.
		pose_a: (T, 6) array [x, y, z, rx, ry, rz].
		pose_b: (T, 6) array [x, y, z, rx, ry, rz].
		step: Decimation factor.
		realtime: If True, pace using mean dt of pose_a timestamps.
		arrow_scale: Scale applied to rotation vectors when plotting arrows.
	"""
	timestamps = np.asarray(timestamps)
	pose_a = np.asarray(pose_a)
	pose_b = np.asarray(pose_b)
	if timestamps.ndim != 1:
		raise ValueError("timestamps must have shape (T,)")
	if pose_a.ndim != 2 or pose_a.shape[1] != 6:
		raise ValueError("pose_a must have shape (T, 6)")
	if pose_b.ndim != 2 or pose_b.shape[1] != 6:
		raise ValueError("pose_b must have shape (T, 6)")

	step = max(int(step), 1)
	timestamps = timestamps[::step]
	pose_a = pose_a[::step]
	pose_b = pose_b[::step]

	mean_dt = _mean_dt(timestamps, n=20, default=0.004)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	ax.set_title("Pose trajectories")
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")

	pos_a = pose_a[:, 0:3]
	pos_b = pose_b[:, 0:3]
	rot_a = pose_a[:, 3:6]
	rot_b = pose_b[:, 3:6]

	def _axes_from_rotvec(rot_vec):
		axes = R.from_rotvec(rot_vec).as_matrix()
		return axes[:, 0], axes[:, 1], axes[:, 2]

	ax.plot(pos_a[:, 0], pos_a[:, 1], pos_a[:, 2], color=colors[0], linewidth=line_width)
	ax.plot(pos_b[:, 0], pos_b[:, 1], pos_b[:, 2], color=colors[1], linewidth=line_width)

	point_a = ax.scatter([pos_a[0, 0]], [pos_a[0, 1]], [pos_a[0, 2]], s=point_size, color=colors[0])
	point_b = ax.scatter([pos_b[0, 0]], [pos_b[0, 1]], [pos_b[0, 2]], s=point_size, color=colors[1])

	ax_a, ay_a, az_a = _axes_from_rotvec(rot_a[0])
	frame_a = ax.quiver(
		[pos_a[0, 0]] * 3, [pos_a[0, 1]] * 3, [pos_a[0, 2]] * 3,
		[ax_a[0], ay_a[0], az_a[0]], [ax_a[1], ay_a[1], az_a[1]], [ax_a[2], ay_a[2], az_a[2]],
		length=arrow_scale,
		color=["red", "green", "blue"],
		normalize=False,
	)
	ax_b, ay_b, az_b = _axes_from_rotvec(rot_b[0])
	frame_b = ax.quiver(
		[pos_b[0, 0]] * 3, [pos_b[0, 1]] * 3, [pos_b[0, 2]] * 3,
		[ax_b[0], ay_b[0], az_b[0]], [ax_b[1], ay_b[1], az_b[1]], [ax_b[2], ay_b[2], az_b[2]],
		length=arrow_scale,
		color=["red", "green", "blue"],
		normalize=False,
	)

	max_len = max(pos_a.shape[0], pos_b.shape[0])
	for i in range(max_len):
		frame_start = time.perf_counter()
		if i < pos_a.shape[0]:
			point_a._offsets3d = ([pos_a[i, 0]], [pos_a[i, 1]], [pos_a[i, 2]])
			frame_a.remove()
			ax_a, ay_a, az_a = _axes_from_rotvec(rot_a[i])
			frame_a = ax.quiver(
				[pos_a[i, 0]] * 3, [pos_a[i, 1]] * 3, [pos_a[i, 2]] * 3,
				[ax_a[0], ay_a[0], az_a[0]], [ax_a[1], ay_a[1], az_a[1]], [ax_a[2], ay_a[2], az_a[2]],
				length=arrow_scale,
				color=["red", "green", "blue"],
				normalize=False,
			)
		if i < pos_b.shape[0]:
			point_b._offsets3d = ([pos_b[i, 0]], [pos_b[i, 1]], [pos_b[i, 2]])
			frame_b.remove()
			ax_b, ay_b, az_b = _axes_from_rotvec(rot_b[i])
			frame_b = ax.quiver(
				[pos_b[i, 0]] * 3, [pos_b[i, 1]] * 3, [pos_b[i, 2]] * 3,
				[ax_b[0], ay_b[0], az_b[0]], [ax_b[1], ay_b[1], az_b[1]], [ax_b[2], ay_b[2], az_b[2]],
				length=arrow_scale,
				color=["red", "green", "blue"],
				normalize=False,
			)

		plt.pause(0.001)
		if not realtime:
			continue

		if i + 1 < len(timestamps):
			desired = max(mean_dt, float(timestamps[i + 1] - timestamps[i]))
		else:
			desired = mean_dt

		elapsed = time.perf_counter() - frame_start
		to_sleep = desired - elapsed
		if to_sleep > 0.0:
			time.sleep(to_sleep)

	plt.show()


def evaluate_6d_matplotlib(
	primitive,
	filter,
	test_trajectory,
	observation_noise,
	*,
	time_step=1,
	pause_secs=0.04,
	observe_controlled_start=False,
	drop_redundant_stationary_obs=True,
	stationary_pos_eps=1e-3,
	stationary_rot_eps=5e-3,
):
	"""Run online EBIP inference and visualize controlled vs observed trajectories.

	Args:
		primitive: Trained BayesianInteractionPrimitive.
		filter: Filter template (deep-copied per run).
		test_trajectory: (12, T) array [ctrl pos+rotvec, obs pos+rotvec].
		observation_noise: (12, 12) observation noise matrix.
		time_step: Observations per inference call.
		pause_secs: Pause between viewer updates.
		observe_controlled_start: If True, observe controlled position at t=0.
	"""
	traj = np.asarray(test_trajectory)
	if traj.ndim != 2 or traj.shape[0] != 12:
		raise ValueError("test_trajectory must have shape (12, T)")

	observed_dof_indices = np.array([6, 7, 8, 9, 10, 11], dtype=np.int32)
	observed_dof_indices_with_ctrl_pos = np.array([0, 1, 2, 6, 7, 8, 9, 10, 11], dtype=np.int32)

	def _is_stationary_step(traj, prev_col, curr_col, active_dofs):
		if prev_col < 0 or curr_col < 0:
			return False
		pos_mask = np.isin(active_dofs, np.array([0, 1, 2, 6, 7, 8], dtype=np.int32))
		rot_mask = np.isin(active_dofs, np.array([3, 4, 5, 9, 10, 11], dtype=np.int32))
		diffs = np.abs(traj[active_dofs, curr_col] - traj[active_dofs, prev_col])
		if np.any(pos_mask) and np.max(diffs[pos_mask]) > stationary_pos_eps:
			return False
		if np.any(rot_mask) and np.max(diffs[rot_mask]) > stationary_rot_eps:
			return False
		return True

	traj_partial = np.array(traj, copy=True)
	traj_partial[0:6, :] = 0.0
	if observe_controlled_start:
		traj_partial[0:3, 0] = traj[0:3, 0]

	new_filter = copy.deepcopy(filter)
	primitive.set_filter(new_filter)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection="3d")
	ax.set_title("EBIP inference (controlled vs observed)")
	ax.set_xlabel("X")
	ax.set_ylabel("Y")
	ax.set_zlabel("Z")

	obs_pos = traj[6:9, :].T
	ctrl_pos = traj[0:3, :].T
	obs_rot = traj[9:12, :].T
	ctrl_rot = traj[3:6, :].T
	obs_line, = ax.plot(obs_pos[:, 0], obs_pos[:, 1], obs_pos[:, 2], color="orange", alpha=0.3)
	gen_line, = ax.plot([], [], [], color="dodgerblue", linewidth=2.0)
	obs_point = ax.scatter([obs_pos[0, 0]], [obs_pos[0, 1]], [obs_pos[0, 2]], color="orange", s=40)
	gen_point = ax.scatter([ctrl_pos[0, 0]], [ctrl_pos[0, 1]], [ctrl_pos[0, 2]], color="dodgerblue", s=40)

	def _axes_from_rotvec(rot_vec):
		axes = R.from_rotvec(rot_vec).as_matrix()
		return axes[:, 0], axes[:, 1], axes[:, 2]

	obs_ax, obs_ay, obs_az = _axes_from_rotvec(obs_rot[0])
	obs_frame = ax.quiver(
		[obs_pos[0, 0]] * 3, [obs_pos[0, 1]] * 3, [obs_pos[0, 2]] * 3,
		[obs_ax[0], obs_ay[0], obs_az[0]], [obs_ax[1], obs_ay[1], obs_az[1]], [obs_ax[2], obs_ay[2], obs_az[2]],
		length=0.05,
		color=["red", "green", "blue"],
		normalize=False,
	)
	ctrl_ax, ctrl_ay, ctrl_az = _axes_from_rotvec(ctrl_rot[0])
	ctrl_frame = ax.quiver(
		[ctrl_pos[0, 0]] * 3, [ctrl_pos[0, 1]] * 3, [ctrl_pos[0, 2]] * 3,
		[ctrl_ax[0], ctrl_ay[0], ctrl_az[0]], [ctrl_ax[1], ctrl_ay[1], ctrl_az[1]], [ctrl_ax[2], ctrl_ay[2], ctrl_az[2]],
		length=0.05,
		color=["red", "green", "blue"],
		normalize=False,
	)

	prev_observed_index = 0
	for observed_index in range(time_step, traj.shape[1], time_step):
		is_first_step = (prev_observed_index == 0)
		active_dofs = (
			observed_dof_indices_with_ctrl_pos
			if (is_first_step and observe_controlled_start)
			else observed_dof_indices
		)

		if drop_redundant_stationary_obs and not is_first_step:
			prev_col = (prev_observed_index - 1) if prev_observed_index > 0 else 0
			curr_col = observed_index - 1
			if _is_stationary_step(traj, prev_col, curr_col, active_dofs):
				obs_segment = obs_pos[:observed_index]
				obs_line.set_data(obs_segment[:, 0], obs_segment[:, 1])
				obs_line.set_3d_properties(obs_segment[:, 2])
				obs_point._offsets3d = ([obs_segment[-1, 0]], [obs_segment[-1, 1]], [obs_segment[-1, 2]])
				obs_frame.remove()
				obs_ax, obs_ay, obs_az = _axes_from_rotvec(obs_rot[observed_index - 1])
				obs_frame = ax.quiver(
					[obs_segment[-1, 0]] * 3, [obs_segment[-1, 1]] * 3, [obs_segment[-1, 2]] * 3,
					[obs_ax[0], obs_ay[0], obs_az[0]], [obs_ax[1], obs_ay[1], obs_az[1]], [obs_ax[2], obs_ay[2], obs_az[2]],
					length=0.05,
					color=["red", "green", "blue"],
					normalize=False,
				)
				plt.pause(pause_secs)
				prev_observed_index = observed_index
				continue

		gen_trajectory, phase, mean, var = primitive.generate_probable_trajectory_recursive(
			traj_partial[:, prev_observed_index:observed_index],
			observation_noise,
			active_dofs,
			num_samples=traj_partial.shape[1] - observed_index,
		)

		gen_pos = gen_trajectory[0:3, :].T
		gen_rot = gen_trajectory[3:6, :].T
		obs_segment = obs_pos[:observed_index]
		gen_line.set_data(gen_pos[:, 0], gen_pos[:, 1])
		gen_line.set_3d_properties(gen_pos[:, 2])
		obs_line.set_data(obs_segment[:, 0], obs_segment[:, 1])
		obs_line.set_3d_properties(obs_segment[:, 2])
		gen_point._offsets3d = ([gen_pos[-1, 0]], [gen_pos[-1, 1]], [gen_pos[-1, 2]])
		obs_point._offsets3d = ([obs_segment[-1, 0]], [obs_segment[-1, 1]], [obs_segment[-1, 2]])
		obs_frame.remove()
		obs_ax, obs_ay, obs_az = _axes_from_rotvec(obs_rot[observed_index - 1])
		obs_frame = ax.quiver(
			[obs_segment[-1, 0]] * 3, [obs_segment[-1, 1]] * 3, [obs_segment[-1, 2]] * 3,
			[obs_ax[0], obs_ay[0], obs_az[0]], [obs_ax[1], obs_ay[1], obs_az[1]], [obs_ax[2], obs_ay[2], obs_az[2]],
			length=0.05,
			color=["red", "green", "blue"],
			normalize=False,
		)
		ctrl_frame.remove()
		ctrl_ax, ctrl_ay, ctrl_az = _axes_from_rotvec(gen_rot[0])
		ctrl_frame = ax.quiver(
			[gen_pos[0, 0]] * 3, [gen_pos[0, 1]] * 3, [gen_pos[0, 2]] * 3,
			[ctrl_ax[0], ctrl_ay[0], ctrl_az[0]], [ctrl_ax[1], ctrl_ay[1], ctrl_az[1]], [ctrl_ax[2], ctrl_ay[2], ctrl_az[2]],
			length=0.05,
			color=["red", "green", "blue"],
			normalize=False,
		)

		plt.pause(pause_secs)
		prev_observed_index = observed_index

	plt.show()
