import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
import concurrent.futures
from scipy.spatial import cKDTree
import sys

from Scripts.aabc_utils import (
    AABC_VERSION,
    PARAM_LOWER,
    PARAM_UPPER,
    scale_theta,
    write_json,
)

# =========================
# Global objects (per worker)
# =========================
Theta_ref = None
X_ref = None
Theta_ref_tree = None

DEFAULT_SEED = 20260818
DIRICHLET_CONCENTRATION = 20.0

# =========================
# Worker initializer
# =========================

def init_worker():
    """
    Runs ONCE per worker process.
    Loads reference data and builds the KD-tree.
    """
    global Theta_ref, X_ref, Theta_ref_tree

    Theta_ref = np.load('all_params.npy')
    print("params_shape: " + str(Theta_ref.shape))
    X_ref = np.load('all_crockers_flattened.npy').astype(float)
    print("crockers_shape: " + str(X_ref.shape))

    Theta_ref_tree = cKDTree(scale_theta(Theta_ref))

# =========================
# Nearest neighbors via KD-tree
# =========================

def aabc_find_nearest_neighbors(theta_star, k):
    """
    Returns distances and indices of the k nearest neighbors
    plus the (k+1)-th distance for kernel scaling.
    """
    dists, idx = Theta_ref_tree.query(scale_theta(theta_star), k=k + 1)

    return (
        dists[:k],      # k nearest distances
        idx[:k],        # k nearest indices
        dists[k]        # (k+1)-th distance
    )

# =========================
# AABC resampling
# =========================

def aabc_resample(theta_star, k, iSample, NUM_SAMPLE, task_seed):
    theta_star = np.asarray(theta_star)
    rng = np.random.default_rng(task_seed)

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
    kernel_weights = np.maximum(1.0 - ratio_sq, 0.0)

    if not np.any(kernel_weights > 0.0):
        kernel_probabilities = np.full(k, 1.0 / k)
    else:
        kernel_probabilities = kernel_weights / kernel_weights.sum()

    # -------------------------------
    # 3. Dirichlet draw
    # -------------------------------
    alpha = np.maximum(
        DIRICHLET_CONCENTRATION * kernel_probabilities,
        1e-12,
    )
    phi = rng.dirichlet(alpha)

    # -------------------------------
    # 4. Synthetic dataset
    # -------------------------------
    x_star = np.zeros_like(X_ref[0], dtype=float)
    for i in range(k):
        x_star += phi[i] * X_ref[k_idx[i]]

    # -------------------------------
    # 5. Save output
    # -------------------------------
    base_dir = f'./sample_aabc_{NUM_SAMPLE}'
    run_dir = os.path.join(base_dir, f'run_{iSample+1}')

    os.makedirs(run_dir, exist_ok=True)

    np.save(os.path.join(run_dir, 'theta_star.npy'), theta_star)
    np.save(os.path.join(run_dir, 'crocker_angles.npy'), x_star.reshape(100, 200, 2))

# =========================
# Wrapper for multiprocessing
# =========================

def simulation_wrapper(args):
    C, L, W, k, iSample, NUM_SAMPLE, task_seed = args
    theta_star = [C, L, W]
    aabc_resample(theta_star, k, iSample, NUM_SAMPLE, task_seed)

# =========================
# Main execution
# =========================

if __name__ == '__main__':

    # NUM_SAMPLE = 100000
    NUM_SAMPLE = int(sys.argv[1])
    k = 5
    base_seed = int(os.environ.get("AABC_SEED", DEFAULT_SEED))
    proposal_rng = np.random.default_rng(base_seed)

    base_dir = f"sample_aabc_{NUM_SAMPLE}"
    os.makedirs(base_dir, exist_ok=True)
    existing_run_numbers = []
    for name in os.listdir(base_dir):
        if name.startswith("run_") and name[4:].isdigit():
            existing_run_numbers.append(int(name[4:]))
    if existing_run_numbers and max(existing_run_numbers) > NUM_SAMPLE:
        raise RuntimeError(
            f"{base_dir} contains runs beyond {NUM_SAMPLE}; refusing to mix "
            "stale samples into this batch"
        )
    write_json(
        os.path.join(base_dir, "aabc_metadata.json"),
        {
            "aabc_version": AABC_VERSION,
            "samples": NUM_SAMPLE,
            "neighbors": k,
            "dirichlet_concentration": DIRICHLET_CONCENTRATION,
            "seed": base_seed,
            "parameter_lower": PARAM_LOWER,
            "parameter_upper": PARAM_UPPER,
        },
    )

    tasks = []
    for iSample in range(NUM_SAMPLE):
        C, L, W = proposal_rng.uniform(PARAM_LOWER, PARAM_UPPER)
        task_seed = np.random.SeedSequence([base_seed, iSample]).generate_state(1)[0]
        tasks.append((C, L, W, k, iSample, NUM_SAMPLE, task_seed))

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=4,
        initializer=init_worker
    ) as executor:
        # Consume the iterator so worker exceptions reach the parent process.
        for _ in executor.map(simulation_wrapper, tasks, chunksize=20):
            pass
