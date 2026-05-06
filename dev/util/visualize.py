import time
import numpy as np

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
