from scipy.spatial.transform import Rotation as R # Needs updated scipy version for scalar_first=True (current apt version does not have access)
import numpy as np

def rotation_dim_reduction(trajectory):
    """Reduce the dimensionality of the trajectory data by converting quaternions to angle-axis representation."""
    reduced_trajectory = np.zeros((trajectory.shape[0], 3)) # 3 orientation
    for i in range(trajectory.shape[0]):
        quat = trajectory[i, :] # Assuming quaternion is in the order [qw, qx, qy, qz]
        r = R.from_quat(quat, scalar_first=True) # Create a rotation object from the quaternion
        rot_vec = r.as_rotvec() # Convert to angle-axis representation (rotation vector)
        reduced_trajectory[i, :] = rot_vec

    return reduced_trajectory