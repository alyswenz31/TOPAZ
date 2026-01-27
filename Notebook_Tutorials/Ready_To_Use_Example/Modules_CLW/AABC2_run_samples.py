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

def init_worker(C_idx, L_idx, W_idx,):
    """
    Runs ONCE per worker process.
    Loads reference data and builds the KD-tree.
    """
    global Theta_ref, X_ref, Theta_ref_tree

    BASE_DIR = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/'
    Theta_ref = np.load(os.path.join(BASE_DIR, "all_params.npy"))
    X_ref = np.load(os.path.join(BASE_DIR, "all_crockers_flattened.npy")).astype(float)

    Theta_ref_tree = cKDTree(Theta_ref)


# =========================
# Nearest neighbors via KD-tree
# =========================

def aabc_find_nearest_neighbors(theta_star, k):
    dists, idx = Theta_ref_tree.query(theta_star, k=k+1)

    return (
        dists[:k],
        idx[:k],
        dists[k]
    )

# =========================
# AABC resampling
# =========================

def aabc_resample(theta_star, k, iSample, NUM_SAMPLE, C_idx, L_idx, W_idx):
    theta_star = np.asarray(theta_star)

    k_dists, k_idx, k1_dist = aabc_find_nearest_neighbors(theta_star, k)

    if k1_dist == 0:
        raise ValueError("theta_star identical to reference parameter")

    ratio_sq = (k_dists / k1_dist) ** 2
    weights = (3/4) * (1 / k1_dist) * (1 - ratio_sq)
    weights[k_dists >= k1_dist] = 0.0

    if np.all(weights == 0):
        weights[:] = 1.0 / k

    weights = np.maximum(weights, 1e-12)

    phi = np.random.dirichlet(weights)

    x_star = np.zeros_like(X_ref[0], dtype=float)
    for i in range(k):
        x_star += phi[i] * X_ref[k_idx[i]]

    base_dir = f"./sample_aabc_{NUM_SAMPLE}"
    run_dir = os.path.join(base_dir, f'run_{iSample+1}')
    os.makedirs(run_dir, exist_ok=True)

    np.save(os.path.join(run_dir, 'theta_star.npy'), theta_star)
    np.save(
        os.path.join(run_dir, 'crocker_angles.npy'),
        x_star.reshape(100, 200, 2)
    )

# =========================
# Wrapper for multiprocessing
# =========================

def simulation_wrapper(args):
    C, L, W, k, iSample, NUM_SAMPLE, C_idx, L_idx, W_idx = args
    theta_star = [C, L, W]
    aabc_resample(theta_star, k, iSample, NUM_SAMPLE, C_idx, L_idx, W_idx)

# =========================
# Public API: run locally
# =========================

def run_samples_aabc(NUM_SAMPLES, k, C_idx, L_idx, W_idx, C_lower_val, C_upper_val, L_lower_val, L_upper_val, W_lower_val, W_upper_val, n_workers=4, use_multiprocessing=False):
    """
    Run AABC sampling locally.

    Parameters
    ----------
    NUM_SAMPLES : int
        Number of AABC samples
    k : int
        Number of nearest neighbors
    n_workers : int
        Number of processes (ignored if use_multiprocessing=False)
    use_multiprocessing : bool
        Whether to use ProcessPoolExecutor

    C - Ratio of magnitude of attractive and respulive forces
    L - Ratio of range of attractive and respulive forces
    W - Alignment parameter

    C_lower_val - Value of lower bound for C grid 
    C_upper_val - Value of upper bound for C grid 
    L_lower_val - Value of lower bound for L grid 
    L_upper_val - Value of upper bound for L grid 
    W_lower_val - Value of lower bound for W grid 
    W_upper_val - Value of upper bound for W grid 

    Returns
    -------
    samples : np.ndarray, shape (NUM_SAMPLES, 3)
        The sampled theta_star values
    """

    # Store sampled parameters for return
    samples = np.zeros((NUM_SAMPLES, 3))

    tasks = []
    for iSample in range(NUM_SAMPLES):
        C = np.random.uniform(C_lower_val, C_upper_val)
        L = np.random.uniform(L_lower_val, L_upper_val)
        W = np.random.uniform(W_lower_val, W_upper_val)

        samples[iSample] = [C, L, W]
        tasks.append((C, L, W, k, iSample, NUM_SAMPLES, C_idx, L_idx, W_idx))

    if use_multiprocessing:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_worker
        ) as executor:
            list(executor.map(simulation_wrapper, tasks))
    else:
        # Useful for debugging / notebooks
        init_worker(C_idx, L_idx, W_idx)
        for args in tasks:
            simulation_wrapper(args)

    return samples
