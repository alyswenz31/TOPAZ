import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
import concurrent.futures
from scipy.spatial import cKDTree

# =========================
# Global objects (per worker)
# =========================
Theta_ref = None
X_ref = None
Theta_ref_tree = None

# =========================
# Worker initializer
# =========================

def init_worker():
    """
    Runs ONCE per worker process.
    Loads reference data and builds the KD-tree.
    """
    global Theta_ref, X_ref, Theta_ref_tree

    Theta_ref = np.load('Simulated_Grid/ODE_Align/all_params.npy')
    X_ref = np.load('Simulated_Grid/ODE_Align/all_crockers_flattened.npy').astype(float)

    Theta_ref_tree = cKDTree(Theta_ref)

# =========================
# Nearest neighbors via KD-tree
# =========================

def aabc_find_nearest_neighbors(theta_star, k):
    """
    Returns distances and indices of the k nearest neighbors
    plus the (k+1)-th distance for kernel scaling.
    """
    dists, idx = Theta_ref_tree.query(theta_star, k=k+1)

    return (
        dists[:k],      # k nearest distances
        idx[:k],        # k nearest indices
        dists[k]        # (k+1)-th distance
    )

# =========================
# AABC resampling
# =========================

def aabc_resample(theta_star, k, iSample, NUM_SAMPLE):
    theta_star = np.asarray(theta_star)

    # -------------------------------
    # 1. Nearest neighbors
    # -------------------------------
    k_dists, k_idx, k1_dist = aabc_find_nearest_neighbors(theta_star, k)

    if k1_dist == 0:
        raise ValueError("theta_star identical to reference parameter")

    # -------------------------------
    # 2. Epanechnikov weights
    # -------------------------------
    ratio_sq = (k_dists / k1_dist) ** 2
    weights = (3/4) * (1 / k1_dist) * (1 - ratio_sq)
    weights[k_dists >= k1_dist] = 0.0

    if np.all(weights == 0):
        weights[:] = 1.0 / k

    weights = np.maximum(weights, 1e-12)

    # -------------------------------
    # 3. Dirichlet draw
    # -------------------------------
    phi = np.random.dirichlet(weights)

    # -------------------------------
    # 4. Synthetic dataset
    # -------------------------------
    x_star = np.zeros_like(X_ref[0], dtype=float)
    for i in range(k):
        x_star += phi[i] * X_ref[k_idx[i]]

    # -------------------------------
    # 5. Save output
    # -------------------------------
    base_dir = f'./Simulated_Grid/ODE_Align/sample_aabc_{NUM_SAMPLE}'
    run_dir = os.path.join(base_dir, f'run_{iSample+1}')

    os.makedirs(run_dir, exist_ok=True)

    np.save(os.path.join(run_dir, 'theta_star.npy'), theta_star)
    np.save(os.path.join(run_dir, 'crocker_angles.npy'), x_star.reshape(100, 200, 2))

# =========================
# Wrapper for multiprocessing
# =========================

def simulation_wrapper(args):
    C, L, W, k, iSample, NUM_SAMPLE = args
    theta_star = [C, L, W]
    aabc_resample(theta_star, k, iSample, NUM_SAMPLE)

# =========================
# Main execution
# =========================

if __name__ == '__main__':

    ## UPDATE NUMBER OF SAMPLES HERE (Ex. 25000, 25001, etc)
    NUM_SAMPLE = 50000
    k = 5

    tasks = []
    for iSample in range(NUM_SAMPLE):
        # This is the smaller denser grid range for C=0.7, L=2.5 but adjust as needed
        C = np.random.uniform(0.5, 3.0)
        L = np.random.uniform(0.1, 0.8)
        W = 0.0 
        tasks.append((C, L, W, k, iSample, NUM_SAMPLE))

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=4,
        initializer=init_worker
    ) as executor:
        executor.map(simulation_wrapper, tasks)
