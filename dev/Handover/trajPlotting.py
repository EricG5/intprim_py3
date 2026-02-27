import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
from matplotlib import animation


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