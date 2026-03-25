import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import animation
from scipy.spatial.transform import Rotation


def quat_to_rotmat(q):
    """Convert a quaternion [x, y, z, w] (scipy / CSV convention) to a 3x3 rotation matrix."""
    return Rotation.from_quat(q).as_matrix()  # scipy uses [x,y,z,w] — same as CSV q0,q1,q2,q3


def plot_pose_trajectory(
    positions,
    quaternions,
    positions2=None,
    quaternions2=None,
    every_n=10,
    arrow_length=0.02,
    labels=("Giver", "Taker"),
    title="6D Pose Trajectory",
    show=True,
):
    """Plot 3D position paths with orientation triads at sampled points.

    Each triad is a set of three arrows (R=X, G=Y, B=Z) showing the local
    frame orientation derived from the quaternion.

    Args:
        positions:    (T, 3) array of XYZ positions for agent 1.
        quaternions:  (T, 4) array of [x, y, z, w] quaternions for agent 1 (CSV convention).
        positions2:   Optional (T, 3) for agent 2.
        quaternions2: Optional (T, 4) for agent 2.
        every_n:      Draw a triad every N timesteps (avoids clutter).
        arrow_length: Length of each axis arrow in world units.
        labels:       Tuple of legend labels for each agent.
        title:        Figure title.
        show:         If True, call plt.show().

    Returns:
        (fig, ax)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    fig.suptitle(title)

    axis_colors = ["red", "green", "blue"]  # X, Y, Z

    def _draw_agent(pos, quat, color, label):
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, lw=2, label=label)
        ax.scatter([pos[0, 0]], [pos[0, 1]], [pos[0, 2]], color=color, s=60, marker="o", zorder=5)
        ax.scatter([pos[-1, 0]], [pos[-1, 1]], [pos[-1, 2]], color=color, s=60, marker="x", zorder=5)
        for i in range(0, len(pos), every_n):
            R = quat_to_rotmat(quat[i])
            origin = pos[i]
            for col_idx in range(3):
                direction = R[:, col_idx] * arrow_length
                ax.quiver(
                    origin[0], origin[1], origin[2],
                    direction[0], direction[1], direction[2],
                    color=axis_colors[col_idx], arrow_length_ratio=0.2, linewidth=1.5,
                )

    _draw_agent(np.asarray(positions), np.asarray(quaternions), "#ff6a6a", labels[0])
    if positions2 is not None and quaternions2 is not None:
        _draw_agent(np.asarray(positions2), np.asarray(quaternions2), "#6ba3ff", labels[1])

    ax.legend(loc="best")
    ax.set_box_aspect(None)

    if show:
        plt.show()

    return fig, ax


def plot_quaternion_components(
    quaternions,
    quaternions2=None,
    labels=("Giver", "Taker"),
    title="Quaternion components over phase",
    show=True,
):
    """Plot quaternion [w, x, y, z] components as 4 subplots over normalized phase.

    Args:
        quaternions:  (T, 4) array of [x, y, z, w] (CSV convention: q0,q1,q2,q3).
        quaternions2: Optional second agent.
        labels:       Legend labels.
        show:         If True, call plt.show().

    Returns:
        (fig, axes)
    """
    quat = np.asarray(quaternions)
    phase = np.linspace(0, 1, len(quat))
    comp_names = ["x", "y", "z", "w"]  # matches CSV column order: q0,q1,q2,q3

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(title)

    for i, (ax_i, name) in enumerate(zip(axes, comp_names)):
        ax_i.plot(phase, quat[:, i], color="#ff6a6a", label=labels[0])
        if quaternions2 is not None:
            q2 = np.asarray(quaternions2)
            phase2 = np.linspace(0, 1, len(q2))
            ax_i.plot(phase2, q2[:, i], color="#6ba3ff", label=labels[1])
        ax_i.set_ylabel(f"q_{name}")
        ax_i.legend(loc="best", fontsize=8)
        ax_i.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Phase")

    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes


class LiveInteractionViewer:
    """Persistent 3D viewer that updates in-place each inference step.

    Creates a single figure on construction and updates the plotted lines
    via set_data / set_3d_properties on every call to update(), avoiding
    the overhead of spawning a new window at each timestep.

    Usage::

        viewer = LiveInteractionViewer(mean_interaction=mean_traj)
        for each_step:
            viewer.update(gen_trajectory_world, observed_partial_world)
        viewer.keep_open()   # block at the end so the final frame stays visible
    """

    _C_OBS        = "#6ba3ff"
    _C_CTRL_INF   = "#ff6a6a"
    _C_TAKER_INF  = "#ffa94d"
    _C_CTRL_MEAN  = "#85d87f"
    _C_TAKER_MEAN = "#9ad0f5"

    def __init__(self, mean_interaction=None, observed_agent="taker", title="Live interaction", pause_secs=0.05):
        """
        Args:
            mean_interaction: optional (6, T) array — used for static axis limits
                              and drawn as a translucent reference.
            observed_agent:   'taker' or 'controlled'.
            title:            Window suptitle.
            pause_secs:       Seconds to pause after each update so the GUI can
                              render.  Increase to slow the animation down.
        """
        self.observed_agent = observed_agent
        self.mean_interaction = np.asarray(mean_interaction) if mean_interaction is not None else None
        self._limits_fixed = False
        self._pause_secs = pause_secs

        plt.ion()
        self.fig = plt.figure(figsize=(9, 7))
        self.ax  = self.fig.add_subplot(111, projection="3d")
        self.fig.suptitle(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        obs_label = "Taker observed" if observed_agent == "taker" else "Controlled observed"
        (self._line_obs,)       = self.ax.plot([], [], [], color=self._C_OBS,      lw=3.0, label=obs_label)
        (self._line_ctrl_inf,)  = self.ax.plot([], [], [], "--", color=self._C_CTRL_INF,  lw=2.0, label="Controlled inferred")
        (self._line_taker_inf,) = self.ax.plot([], [], [], "--", color=self._C_TAKER_INF, lw=2.0, label="Taker inferred")
        (self._dot_obs,)        = self.ax.plot([], [], [], "o",  color=self._C_OBS,       ms=6)
        (self._dot_ctrl,)       = self.ax.plot([], [], [], "o",  color="red",             ms=10, zorder=5, label="Controlled current")
        (self._dot_taker,)      = self.ax.plot([], [], [], "o",  color=self._C_TAKER_INF, ms=6)

        if self.mean_interaction is not None and self.mean_interaction.shape[0] >= 6:
            cm = self.mean_interaction[0:3]
            tm = self.mean_interaction[3:6]
            self.ax.plot(cm[0], cm[1], cm[2], color=self._C_CTRL_MEAN,  lw=1.5, label="Controlled mean", alpha=0.5)
            self.ax.plot(tm[0], tm[1], tm[2], color=self._C_TAKER_MEAN, lw=1.5, label="Taker mean",       alpha=0.5)
            self._set_limits_from(self.mean_interaction)

        self.ax.legend(loc="best", fontsize=8)
        self.fig.canvas.draw()
        plt.pause(self._pause_secs)

    def update(self, gen_trajectory, observed_partial):
        """Redraw with the latest inference output.

        Args:
            gen_trajectory:   (6, T_future) predicted interaction in world coords.
            observed_partial: (6, t_obs)    observation history in world coords.
        """
        gen_trajectory   = np.asarray(gen_trajectory)
        observed_partial = np.asarray(observed_partial)

        ctrl_inf  = gen_trajectory[0:3]
        taker_inf = gen_trajectory[3:6]
        obs = observed_partial[3:6] if self.observed_agent == "taker" else observed_partial[0:3]

        def _set(line, r):
            line.set_data(r[0], r[1])
            line.set_3d_properties(r[2])

        def _set_dot(dot, r):
            dot.set_data([r[0, -1]], [r[1, -1]])
            dot.set_3d_properties([r[2, -1]])

        _set(self._line_ctrl_inf,  ctrl_inf)
        _set(self._line_taker_inf, taker_inf)

        if obs.shape[1] > 0:
            _set(self._line_obs, obs)
            _set_dot(self._dot_obs,   obs)          # taker current position (last observed)
            # Controlled current position = first column of predicted future (index 0)
            self._dot_ctrl.set_data([ctrl_inf[0, 0]], [ctrl_inf[1, 0]])
            self._dot_ctrl.set_3d_properties([ctrl_inf[2, 0]])
            _set_dot(self._dot_taker, taker_inf)    # taker predicted end

        if not self._limits_fixed:
            xs = np.concatenate([ctrl_inf[0], taker_inf[0], obs[0]])
            ys = np.concatenate([ctrl_inf[1], taker_inf[1], obs[1]])
            zs = np.concatenate([ctrl_inf[2], taker_inf[2], obs[2]])
            self.ax.set_xlim(*self._pad(xs))
            self.ax.set_ylim(*self._pad(ys))
            self.ax.set_zlim(*self._pad(zs))

        self.fig.canvas.draw()
        plt.pause(self._pause_secs)

    def keep_open(self):
        """Block until the viewer window is closed (call once after the loop)."""
        plt.ioff()
        plt.show(block=True)

    def _set_limits_from(self, interaction):
        mx = np.concatenate([interaction[0], interaction[3]])
        my = np.concatenate([interaction[1], interaction[4]])
        mz = np.concatenate([interaction[2], interaction[5]])
        self.ax.set_xlim(*self._pad(mx))
        self.ax.set_ylim(*self._pad(my))
        self.ax.set_zlim(*self._pad(mz))
        self._limits_fixed = True

    @staticmethod
    def _pad(values, ratio=0.05):
        vmin, vmax = np.min(values), np.max(values)
        span = max(vmax - vmin, 1e-6)
        return vmin - ratio * span, vmax + ratio * span


class Live6DInteractionViewer:
    """Persistent 3D viewer for 12-DOF (pos + rotvec) interaction trajectories.

    Shows 3D position paths with orientation triads (RGB arrows for local X/Y/Z)
    at sampled points.  Updates in-place each inference step.

    Data layout (12, T):
        rows 0:3  = controlled position XYZ
        rows 3:6  = controlled rotation vector [rx, ry, rz]
        rows 6:9  = taker position XYZ
        rows 9:12 = taker rotation vector [rx, ry, rz]

    Usage::

        viewer = Live6DInteractionViewer(mean_interaction=mean_traj)
        for each_step:
            viewer.update(gen_trajectory_world, observed_partial_world)
        viewer.keep_open()
    """

    _C_OBS        = "#6ba3ff"
    _C_CTRL_INF   = "#ff6a6a"
    _C_TAKER_INF  = "#ffa94d"
    _C_CTRL_MEAN  = "#85d87f"
    _C_TAKER_MEAN = "#9ad0f5"
    _AXIS_COLORS  = ["red", "green", "blue"]  # X, Y, Z

    def __init__(self, mean_interaction=None, observed_agent="taker",
                 title="6D BIP live inference", pause_secs=0.05,
                 triad_every_n=10, arrow_length=0.015):
        """
        Args:
            mean_interaction: optional (12, T) array for static reference + axis limits.
            observed_agent:   'taker' or 'controlled'.
            title:            Window title.
            pause_secs:       Seconds to pause between updates.
            triad_every_n:    Draw an orientation triad every N timesteps.
            arrow_length:     Length of each triad arrow in world units.
        """
        self.observed_agent = observed_agent
        self.mean_interaction = np.asarray(mean_interaction) if mean_interaction is not None else None
        self._limits_fixed = False
        self._pause_secs = pause_secs
        self._triad_every_n = triad_every_n
        self._arrow_length = arrow_length
        self._triad_artists = []  # quiver artists to remove on redraw

        plt.ion()
        self.fig = plt.figure(figsize=(10, 8))
        self.ax  = self.fig.add_subplot(111, projection="3d")
        self.fig.suptitle(title)
        self.ax.set_xlabel("X"); self.ax.set_ylabel("Y"); self.ax.set_zlabel("Z")

        obs_label = "Taker observed" if observed_agent == "taker" else "Controlled observed"
        (self._line_obs,)       = self.ax.plot([], [], [], color=self._C_OBS,       lw=3.0, label=obs_label)
        (self._line_ctrl_inf,)  = self.ax.plot([], [], [], "--", color=self._C_CTRL_INF,  lw=2.0, label="Controlled inferred")
        (self._line_taker_inf,) = self.ax.plot([], [], [], "--", color=self._C_TAKER_INF, lw=2.0, label="Taker inferred")
        (self._dot_obs,)        = self.ax.plot([], [], [], "o",  color=self._C_OBS,       ms=6)
        (self._dot_ctrl,)       = self.ax.plot([], [], [], "o",  color="red",             ms=10, zorder=5, label="Controlled current")
        (self._dot_taker,)      = self.ax.plot([], [], [], "o",  color=self._C_TAKER_INF, ms=6)

        # Draw mean reference (position only)
        if self.mean_interaction is not None and self.mean_interaction.shape[0] >= 12:
            cm = self.mean_interaction[0:3]
            tm = self.mean_interaction[6:9]
            self.ax.plot(cm[0], cm[1], cm[2], color=self._C_CTRL_MEAN,  lw=1.5, label="Controlled mean", alpha=0.5)
            self.ax.plot(tm[0], tm[1], tm[2], color=self._C_TAKER_MEAN, lw=1.5, label="Taker mean",       alpha=0.5)
            self._set_limits_from(self.mean_interaction)

        self.ax.legend(loc="best", fontsize=8)
        self.fig.canvas.draw()
        plt.pause(self._pause_secs)

    def _draw_triads(self, positions, rotvecs, color_tint):
        """Draw orientation triads along a trajectory, return the quiver artists."""
        artists = []
        for i in range(0, positions.shape[1], self._triad_every_n):
            artists += self._draw_single_triad(positions[:, i], rotvecs[:, i])
        return artists

    def _draw_single_triad(self, position, rotvec):
        """Draw a single orientation triad at a given position, return quiver artists."""
        R = Rotation.from_rotvec(rotvec).as_matrix()
        artists = []
        for col_idx in range(3):
            direction = R[:, col_idx] * self._arrow_length
            q = self.ax.quiver(
                position[0], position[1], position[2],
                direction[0], direction[1], direction[2],
                color=self._AXIS_COLORS[col_idx], arrow_length_ratio=0.2, linewidth=1.5,
            )
            artists.append(q)
        return artists

    def _clear_triads(self):
        """Remove previously drawn triads."""
        for a in self._triad_artists:
            a.remove()
        self._triad_artists = []

    def update(self, gen_trajectory, observed_partial):
        """Redraw with the latest inference output.

        Args:
            gen_trajectory:   (12, T_future) predicted interaction in world coords.
            observed_partial: (12, t_obs)    observation history in world coords.
        """
        gen_trajectory   = np.asarray(gen_trajectory)
        observed_partial = np.asarray(observed_partial)

        ctrl_pos  = gen_trajectory[0:3]
        ctrl_rot  = gen_trajectory[3:6]
        taker_pos = gen_trajectory[6:9]
        taker_rot = gen_trajectory[9:12]

        if self.observed_agent == "taker":
            obs_pos = observed_partial[6:9]
        else:
            obs_pos = observed_partial[0:3]

        def _set(line, r):
            line.set_data(r[0], r[1])
            line.set_3d_properties(r[2])

        def _set_dot(dot, r):
            dot.set_data([r[0, -1]], [r[1, -1]])
            dot.set_3d_properties([r[2, -1]])

        _set(self._line_ctrl_inf,  ctrl_pos)
        _set(self._line_taker_inf, taker_pos)

        if obs_pos.shape[1] > 0:
            _set(self._line_obs, obs_pos)
            _set_dot(self._dot_obs, obs_pos)
            self._dot_ctrl.set_data([ctrl_pos[0, 0]], [ctrl_pos[1, 0]])
            self._dot_ctrl.set_3d_properties([ctrl_pos[2, 0]])
            _set_dot(self._dot_taker, taker_pos)

        # Redraw orientation triads — only at current agent positions
        self._clear_triads()
        # Controlled: current position = first column of predicted future
        self._triad_artists += self._draw_single_triad(ctrl_pos[:, 0], ctrl_rot[:, 0])
        # Taker: current orientation = last observed position and rotation
        if obs_pos.shape[1] > 0:
            obs_rot = observed_partial[9:12] if self.observed_agent == "taker" else observed_partial[3:6]
            self._triad_artists += self._draw_single_triad(obs_pos[:, -1], obs_rot[:, -1])
        else:
            self._triad_artists += self._draw_single_triad(taker_pos[:, 0], taker_rot[:, 0])

        if not self._limits_fixed:
            xs = np.concatenate([ctrl_pos[0], taker_pos[0], obs_pos[0]])
            ys = np.concatenate([ctrl_pos[1], taker_pos[1], obs_pos[1]])
            zs = np.concatenate([ctrl_pos[2], taker_pos[2], obs_pos[2]])
            self.ax.set_xlim(*self._pad(xs))
            self.ax.set_ylim(*self._pad(ys))
            self.ax.set_zlim(*self._pad(zs))

        self.fig.canvas.draw()
        plt.pause(self._pause_secs)

    def keep_open(self):
        """Block until the viewer window is closed."""
        plt.ioff()
        plt.show(block=True)

    def _set_limits_from(self, interaction):
        mx = np.concatenate([interaction[0], interaction[6]])
        my = np.concatenate([interaction[1], interaction[7]])
        mz = np.concatenate([interaction[2], interaction[8]])
        self.ax.set_xlim(*self._pad(mx))
        self.ax.set_ylim(*self._pad(my))
        self.ax.set_zlim(*self._pad(mz))
        self._limits_fixed = True

    @staticmethod
    def _pad(values, ratio=0.05):
        vmin, vmax = np.min(values), np.max(values)
        span = max(vmax - vmin, 1e-6)
        return vmin - ratio * span, vmax + ratio * span


def plot_animate(traj1, traj2, interval=20, show=True, labels=None):
    """Plot and animate two 3D trajectories.

    Args:
        traj1: Array-like of shape (T, 3) for trajectory 1.
        traj2: Array-like of shape (T, 3) for trajectory 2.
        n: Number of samples to animate from the start of each trajectory.
        interval: Delay between frames in milliseconds.
        show: If True, calls plt.show().
        labels: Optional tuple/list of two strings for legend labels.

    Returns:
        (fig, ax, ani): Matplotlib figure, 3D axis, and FuncAnimation.
    """
    traj1 = np.asarray(traj1)
    traj2 = np.asarray(traj2)

    if traj1.ndim != 2 or traj1.shape[1] < 3:
        raise ValueError("traj1 must be a 2D array with at least 3 columns")
    if traj2.ndim != 2 or traj2.shape[1] < 3:
        raise ValueError("traj2 must be a 2D array with at least 3 columns")

    x1 = traj1[:, 0]
    y1 = traj1[:, 1]
    z1 = traj1[:, 2]
    x2 = traj2[:, 0]
    y2 = traj2[:, 1]
    z2 = traj2[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Use combined limits so both stay in view
    xmin = min(np.min(x1), np.min(x2))
    xmax = max(np.max(x1), np.max(x2))
    ymin = min(np.min(y1), np.min(y2))
    ymax = max(np.max(y1), np.max(y2))
    zmin = min(np.min(z1), np.min(z2))
    zmax = max(np.max(z1), np.max(z2))

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Two animated lines + dots
    line1, = ax.plot([], [], [], lw=2)
    dot1, = ax.plot([], [], [], marker='o', markersize=5)
    line2, = ax.plot([], [], [], lw=2)
    dot2, = ax.plot([], [], [], marker='o', markersize=5)

    if labels is not None:
        if len(labels) != 2:
            raise ValueError("labels must be a sequence of two strings")
        line1.set_label(labels[0])
        line2.set_label(labels[1])
        ax.legend(loc='best')

    def init():
        for artist in (line1, dot1, line2, dot2):
            artist.set_data([], [])
            artist.set_3d_properties([])
        return line1, dot1, line2, dot2

    def update(frame):
        # Keep animation safe if lengths differ
        f1 = min(frame, len(x1))
        f2 = min(frame, len(x2))

        # Trajectory 1
        if f1 > 0:
            line1.set_data(x1[:f1], y1[:f1])
            line1.set_3d_properties(z1[:f1])
            dot1.set_data([x1[f1 - 1]], [y1[f1 - 1]])
            dot1.set_3d_properties([z1[f1 - 1]])

        # Trajectory 2
        if f2 > 0:
            line2.set_data(x2[:f2], y2[:f2])
            line2.set_3d_properties(z2[:f2])
            dot2.set_data([x2[f2 - 1]], [y2[f2 - 1]])
            dot2.set_3d_properties([z2[f2 - 1]])

        return line1, dot1, line2, dot2

    # Frames: animate until the longer trajectory finishes
    N = max(len(x1), len(x2))
    ani = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=N, interval=interval, blit=False
    )

    if show:
        plt.show()

    return fig, ax, ani


def plot_partial_trajectory_3d(
    interaction,
    partial_observed_interaction,
    mean_interaction=None,
    show=True,
    observed_agent="controlled",
    show_mean=True,
):
    """Plot a partial 3D interaction with two 3D trajectories.

    Expected data format (matches your handover code):
        interaction: (6, T) where
            rows 0:3 = controlled XYZ
            rows 3:6 = taker XYZ
        partial_observed_interaction: (6, t_obs) prefix of observed samples
        mean_interaction: optional (6, T_mean)
        show_mean: if True, plot the mean trajectories (if provided)

    This plots:
        - observed agent (solid)
        - controlled inferred (dashed)
        - taker inferred (dashed)
        - optional mean for both (when show_mean=True)
    """
    interaction = np.asarray(interaction)
    partial_observed_interaction = np.asarray(partial_observed_interaction)

    if interaction.ndim != 2 or interaction.shape[0] < 6:
        raise ValueError("interaction must be a 2D array with at least 6 rows (controlled XYZ + taker XYZ)")
    if partial_observed_interaction.ndim != 2 or partial_observed_interaction.shape[0] < 6:
        raise ValueError("partial_observed_interaction must be a 2D array with at least 6 rows")

    ctrl_inf = interaction[0:3, :]
    taker_inf = interaction[3:6, :]

    if observed_agent == "controlled":
        obs = partial_observed_interaction[0:3, :]
        obs_label = "Controlled observed"
    elif observed_agent == "taker":
        obs = partial_observed_interaction[3:6, :]
        obs_label = "Taker observed"
    else:
        raise ValueError("observed_agent must be 'controlled' or 'taker'")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot observed + inferred trajectories
    ax.plot(obs[0], obs[1], obs[2], color="#6ba3ff", label=obs_label, linewidth=3.0)
    ax.plot(ctrl_inf[0], ctrl_inf[1], ctrl_inf[2], "--", color="#ff6a6a", label="Controlled inferred", linewidth=2.0)
    ax.plot(taker_inf[0], taker_inf[1], taker_inf[2], "--", color="#ffa94d", label="Taker inferred", linewidth=2.0)

    if show_mean and mean_interaction is not None:
        mean_interaction = np.asarray(mean_interaction)
        if mean_interaction.ndim == 2 and mean_interaction.shape[0] >= 6:
            ctrl_mean = mean_interaction[0:3, :]
            taker_mean = mean_interaction[3:6, :]
            ax.plot(ctrl_mean[0], ctrl_mean[1], ctrl_mean[2], color="#85d87f", label="Controlled mean", linewidth=1.5)
            ax.plot(taker_mean[0], taker_mean[1], taker_mean[2], color="#9ad0f5", label="Taker mean", linewidth=1.5)

    def _minmax_with_pad(values, pad_ratio=0.05):
        vmin = np.min(values)
        vmax = np.max(values)
        span = vmax - vmin
        if span == 0:
            span = 1e-6
        pad = pad_ratio * span
        return vmin - pad, vmax + pad

    # Axis limits:
    # - If a mean interaction is provided, use its full extents so limits remain static across partial updates
    #   (independent of whether the mean is currently plotted).
    # - Otherwise, fall back to dynamic limits from the current observed/inferred data.
    if mean_interaction is not None and mean_interaction.ndim == 2 and mean_interaction.shape[0] >= 6:
        mx = np.concatenate([mean_interaction[0], mean_interaction[3]])
        my = np.concatenate([mean_interaction[1], mean_interaction[4]])
        mz = np.concatenate([mean_interaction[2], mean_interaction[5]])
        ax.set_xlim(*_minmax_with_pad(mx))
        ax.set_ylim(*_minmax_with_pad(my))
        ax.set_zlim(*_minmax_with_pad(mz))
    else:
        xs = np.concatenate([ctrl_inf[0], taker_inf[0], obs[0]])
        ys = np.concatenate([ctrl_inf[1], taker_inf[1], obs[1]])
        zs = np.concatenate([ctrl_inf[2], taker_inf[2], obs[2]])
        ax.set_xlim(*_minmax_with_pad(xs))
        ax.set_ylim(*_minmax_with_pad(ys))
        ax.set_zlim(*_minmax_with_pad(zs))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.suptitle("Probable 3D interaction")
    ax.legend(loc="best")

    if show:
        plt.show(block=False)

    return fig, ax