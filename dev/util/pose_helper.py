from scipy.spatial.transform import Rotation as R # Needs updated scipy version for scalar_first=True (current apt version does not have access)
import numpy as np

def rotation_dim_reduction(trajectory):
    """Reduce the dimensionality of the trajectory data by converting quaternions to angle-axis representation."""
    reduced_trajectory = np.zeros((trajectory.shape[0], 3)) # 3 orientation
    for i in range(trajectory.shape[0]):
        quat = trajectory[i, :] # Assuming quaternion is in the order [qw, qx, qy, qz]
        try:
            r = R.from_quat(quat, scalar_first=True) # SciPy >= 1.10
        except TypeError:
            quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=float)
            r = R.from_quat(quat_xyzw) # Older SciPy expects [qx, qy, qz, qw]
        rot_vec = r.as_rotvec() # Convert to angle-axis representation (rotation vector)
        reduced_trajectory[i, :] = rot_vec

    return reduced_trajectory


def rotation_dim_reduction_continuous(quats_wxyz):
    """Convert a quaternion time series to a *continuous* rotation-vector series.

    SciPy's `as_rotvec()` returns a principal rotation vector, which can jump by
    approximately 2π when the underlying rotation stays smooth (especially near
    the π boundary). This function chooses the equivalent rotvec at each step
    (v, v±2π*axis) that stays closest to the previous sample.

    Args:
        quats_wxyz: (T,4) quaternions in (qw,qx,qy,qz)

    Returns:
        (T,3) rotation vectors (angle-axis) with reduced discontinuities.
    """
    quats = np.asarray(quats_wxyz, dtype=float)
    if quats.ndim != 2 or quats.shape[1] != 4:
        raise ValueError(f"Expected quats shape (T,4); got {quats.shape}")

    # Hemisphere continuity first (q and -q represent same rotation)
    quats_fix = np.array(quats, copy=True)
    for i in range(1, quats_fix.shape[0]):
        if float(np.dot(quats_fix[i], quats_fix[i - 1])) < 0.0:
            quats_fix[i] *= -1.0

    # SciPy compatibility: scalar_first may not exist
    try:
        rot = R.from_quat(quats_fix, scalar_first=True)
    except TypeError:
        quat_xyzw = np.column_stack((quats_fix[:, 1], quats_fix[:, 2], quats_fix[:, 3], quats_fix[:, 0]))
        rot = R.from_quat(quat_xyzw)

    rotvecs = rot.as_rotvec()
    if rotvecs.shape[0] <= 1:
        return rotvecs

    two_pi = 2.0 * np.pi
    out = np.array(rotvecs, copy=True)
    prev = out[0]
    for i in range(1, out.shape[0]):
        v = out[i]
        angle = float(np.linalg.norm(v))
        if angle < 1e-12:
            prev = v
            continue

        axis = v / angle
        candidates = (v - two_pi * axis, v, v + two_pi * axis)
        best = min(candidates, key=lambda c: float(np.linalg.norm(c - prev)))
        out[i] = best
        prev = best
    return out