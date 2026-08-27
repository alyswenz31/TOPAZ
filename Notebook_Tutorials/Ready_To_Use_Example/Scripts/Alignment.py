'''

    These are the methods that calculate the change in a cell's velocity/directionality
    There are two main ways this is done: Flores Alignment and Haughey Alignment.

    Flores Alignment --> You can include and exclude a weighting for distance (i.e., the
    impact of alignment varies inversely with the distance of the two object); however, we
    assume that only at a single point (complete orthogonality) will cells not bother to align.
    Also, these cells only align with cells in a neighborhood.

    Haughey Alignment --> Again, you can include a weighting for the distance of cells in a
    neighborhood, but the difference is that there is a region for which cells don't align.
    This also assumes that more orthogonal cell directions will not impact one another, but
    the cut-off to alignment is not 90 degrees but something like 45 - 135 degrees, or so.

'''

import numpy as np


# all_symbols = []
# for name in dir():
#     if not name.startswith('_'):
#         all_symbols.append(name)
# __all__ = all_symbols


def _projection_matrix(vel):
    vnorm2 = np.linalg.norm(vel, axis=1) ** 2
    dot_mat = np.sum(vel * vel[:, np.newaxis], axis=2)
    return (dot_mat / vnorm2[:, np.newaxis]).T

def _calc_scale(vel):
    vel_norms = np.linalg.norm(vel, axis=1)[:,np.newaxis]
    vel_norm_3D = vel_norms[:, np.newaxis] # Nx1x1 velocity norms
    vel_outer3D = np.abs(vel[:,:,np.newaxis]) * np.abs(vel[:,np.newaxis]) # Nx2x2 matrix, outerproduct of velocities
    vel_norm_3D_squared = vel_norm_3D ** 2
    scale = vel_outer3D / vel_norm_3D_squared # Nx2x2 Matrix that will back out the effect of magnitude in alignment=
    return scale

def flores_align(pos: np.ndarray, vel: np.ndarray, N_radius: float, weight: float, dist_bool = False):
    """
    Calculate the alignment with no cutoff region. Only additional parameters are "weight" and
    neighbordhood radius. Specify whether to scale for distance

    Args:
        pos (ndarray): A n x 2 numpy array of [x position, y position]
        vel (ndarray): A n x 2 numpy array of [x velocity, y velocity]
        N_radiis (float): The radius of the neighborhood of each cell
        weight (float): The sensitivity to alignment
        dist_bool (bool): True if scaling for distance from neighbor

    Returns:
        ndarray: A n x 2 scaled, alignment vector [x Align, y Align]
    """

    meps= np.finfo(float).eps

    # Calculate pairwise distances
    distances = np.linalg.norm(pos[:, np.newaxis] - pos, axis=2) # NxN matrix, [i, j] distance from each other

    # Compute weights using vectorized operations
    valid_indices = np.int0(np.logical_and(meps < distances, distances < N_radius)) # NxN with 0 if not neighbor, 1 if neighbor
    vel_norms = np.linalg.norm(vel, axis=1)[:,np.newaxis] # Nx1 vector of the velocity norms
    vel_ratios = vel_norms.T / vel_norms # NxN matrix of vel[j] / vel[i]
    projs = _projection_matrix(vel) # NxN matrix with [i, j] = dot(vel[i], vel[j]) / dot(vel[j], vel[j])
    weight_mat = weight * vel_ratios * projs * valid_indices # NxN matrix of scaling, ready for multipliciation with neighbor velocity

    if dist_bool:
        weight_mat = np.divide(weight_mat, distances, out=np.zeros_like(weight_mat), where=(distances != 0))
    
    align_vec = weight_mat @ vel # Nx2 of the scaled alignment velocity
    vel_norm_3D = vel_norms[:, np.newaxis] # Nx1x1 velocity norms
    vel_outer3D = np.abs(vel[:,:,np.newaxis]) * np.abs(vel[:,np.newaxis]) # Nx2x2 matrix, outerproduct of velocities
    vel_norm_3D_squared = vel_norm_3D ** 2
    scale = vel_outer3D / vel_norm_3D_squared # Nx2x2 Matrix that will back out the effect of magnitude in alignment
    normed_A = np.squeeze(scale @ align_vec[:,:,np.newaxis]) # Nx2 matrix where the only the change in alignment not a change in magnitude

    return normed_A


def haughey_align(pos: np.ndarray, vel: np.ndarray, N_radius: float, weight: float, cut_off: float, dist_bool = False):
    """
    Calculate the alignment with a cutoff region. For a given "cut-off" which is to be between -1 and 1,
    we can infer a cutoff region. If cut-off = 0 then cut-off at +-45 degrees from 0 and 180. If cut-off
    is -1, then there is no effect of alignment, and if cut-off is 1 then only at 90 degrees is there a cut-off

    Args:
        pos (ndarray): A n x 2 numpy array of [x position, y position]
        vel (ndarray): A n x 2 numpy array of [x velocity, y velocity]
        N_radiis (float): The radius of the neighborhood of each cell
        weight (float): The sensitivity to alignment
        cut_off (float): The cut-off angle about 0 and 180, range (-1, 1)
        dist_bool (bool): True if scaling for distance from neighbor

    Returns:
        ndarray: A n x 2 scaled, alignment vector [x Align, y Align]
    """

    meps= np.finfo(float).eps

    # Calculate pairwise distances
    distances = np.linalg.norm(pos[:, np.newaxis] - pos, axis=2) # NxN matrix, [i, j] distance from each other

    # Compute weights using vectorized operations
    valid_indices = np.int0(np.logical_and(meps < distances, distances < N_radius)) # NxN with 0 if not neighbor, 1 if neighbor
    vel_norms = np.linalg.norm(vel, axis=1)[:,np.newaxis] # Nx1 vector of the velocity norms
    vel_ratios = vel_norms.T / vel_norms # NxN matrix of vel[j] / vel[i]
    projs = _projection_matrix(vel) # NxN matrix with [i, j] = dot(vel[i], vel[j]) / dot(vel[j], vel[j])
    weight_mat = (weight / (1 + cut_off)) * vel_ratios * projs * valid_indices

    if dist_bool:
        weight_mat = np.divide(weight_mat, distances, out=np.zeros_like(weight_mat), where=(distances != 0))

    # Calculate the cosine double angle which can be done with cross products and dot products
    cross_mat = np.abs(np.cross(vel[:, np.newaxis], vel)) ** 2 # Pairwise square of magnitude of dot product
    dot_mat = np.dot(vel, vel.T) ** 2 # Pairwise dot product squared
    cos2theta = (dot_mat - cross_mat) / (dot_mat + cross_mat) # Equation for cos(2* delta_theta)
    cut_off_matt = np.clip(cos2theta, 0, None) # NxN Filled with cos2theta or 0 if cos2theta < 0

    weight_mat *= cut_off_matt
    align_vec = weight_mat @ vel
    scale = _calc_scale(vel)
    normed_A = np.squeeze(scale @ align_vec[:,:,np.newaxis])
    return normed_A

