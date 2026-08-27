import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
import concurrent.futures
from scipy.spatial import cKDTree
from Scripts.aabc_utils import AABC_VERSION, write_json

# =========================
# Global objects (per worker)
# =========================
Theta_ref = None
X_ref = None
Theta_ref_tree = None
PARAM_LOWER = None
PARAM_SCALE = None
DEFAULT_SEED = 20260818
DIRICHLET_CONCENTRATION = 20.0

# =========================
# Worker initializer
# =========================

def scale_theta(theta):
    values = np.asarray(theta, dtype=float)
    varying = PARAM_SCALE > 0
    return (values[..., varying] - PARAM_LOWER[varying]) / PARAM_SCALE[varying]

def init_worker(C_idx, L_idx, W_idx, parameter_lower, parameter_upper):
    """
    Runs ONCE per worker process.
    Loads reference data and builds the KD-tree.
    """
    global Theta_ref, X_ref, Theta_ref_tree, PARAM_LOWER, PARAM_SCALE

    PARAM_LOWER = np.asarray(parameter_lower, dtype=float)
    PARAM_SCALE = np.asarray(parameter_upper, dtype=float) - PARAM_LOWER

    BASE_DIR = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/'
    Theta_ref = np.load(os.path.join(BASE_DIR, "all_params.npy"))
    X_ref = np.load(os.path.join(BASE_DIR, "all_crockers_flattened.npy")).astype(float)

    Theta_ref_tree = cKDTree(scale_theta(Theta_ref))


# =========================
# Nearest neighbors via KD-tree
# =========================

def aabc_find_nearest_neighbors(theta_star, k):
    dists, idx = Theta_ref_tree.query(scale_theta(theta_star), k=k+1)

    return (
        dists[:k],
        idx[:k],
        dists[k]
    )

# =========================
# AABC resampling
# =========================

def aabc_resample(theta_star, k, iSample, NUM_SAMPLE, C_idx, L_idx, W_idx, task_seed):
    theta_star = np.asarray(theta_star)
    rng = np.random.default_rng(task_seed)

    k_dists, k_idx, k1_dist = aabc_find_nearest_neighbors(theta_star, k)

    if k1_dist == 0:
        raise ValueError("theta_star identical to reference parameter")

    ratio_sq = (k_dists / k1_dist) ** 2
    kernel_weights = np.maximum(1.0 - ratio_sq, 0.0)
    if not np.any(kernel_weights > 0.0):
        kernel_probabilities = np.full(k, 1.0 / k)
    else:
        kernel_probabilities = kernel_weights / kernel_weights.sum()
    alpha = np.maximum(DIRICHLET_CONCENTRATION * kernel_probabilities, 1e-12)
    phi = rng.dirichlet(alpha)

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
    C, L, W, k, iSample, NUM_SAMPLE, C_idx, L_idx, W_idx, task_seed = args
    theta_star = [C, L, W]
    aabc_resample(theta_star, k, iSample, NUM_SAMPLE, C_idx, L_idx, W_idx, task_seed)

# =========================
# Public API: run locally
# =========================

def run_samples_aabc(NUM_SAMPLES, k, C_idx, L_idx, W_idx, parameter_bounds, n_workers=4, use_multiprocessing=False, seed=DEFAULT_SEED):
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

    parameter_bounds - Values of lower and upper bounds for C, L, and W grids 

    Returns
    -------
    samples : np.ndarray, shape (NUM_SAMPLES, 3)
        The sampled theta_star values
    """

    # Store sampled parameters for return
    samples = np.zeros((NUM_SAMPLES, 3))

    [C_lower_val, C_upper_val] = parameter_bounds[0]
    [L_lower_val, L_upper_val] = parameter_bounds[1]
    [W_lower_val, W_upper_val] = parameter_bounds[2]

    proposal_rng = np.random.default_rng(seed)
    base_dir = f"sample_aabc_{NUM_SAMPLES}"
    os.makedirs(base_dir, exist_ok=True)
    existing = [int(name[4:]) for name in os.listdir(base_dir)
                if name.startswith("run_") and name[4:].isdigit()]
    if existing and max(existing) > NUM_SAMPLES:
        raise RuntimeError(f"{base_dir} contains stale runs beyond {NUM_SAMPLES}")
    write_json(os.path.join(base_dir, "aabc_metadata.json"), {
        "aabc_version": AABC_VERSION, "samples": NUM_SAMPLES,
        "neighbors": k, "dirichlet_concentration": DIRICHLET_CONCENTRATION,
        "seed": seed,
        "parameter_lower": [C_lower_val, L_lower_val, W_lower_val],
        "parameter_upper": [C_upper_val, L_upper_val, W_upper_val],
    })

    tasks = []
    for iSample in range(NUM_SAMPLES):
        C = proposal_rng.uniform(C_lower_val, C_upper_val)
        L = proposal_rng.uniform(L_lower_val, L_upper_val)
        W = proposal_rng.uniform(W_lower_val, W_upper_val)
        task_seed = np.random.SeedSequence([seed, iSample]).generate_state(1)[0]

        samples[iSample] = [C, L, W]
        tasks.append((C, L, W, k, iSample, NUM_SAMPLES, C_idx, L_idx, W_idx, task_seed))

    if use_multiprocessing:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=init_worker,
            initargs=(C_idx, L_idx, W_idx,
                      [C_lower_val, L_lower_val, W_lower_val],
                      [C_upper_val, L_upper_val, W_upper_val]),
        ) as executor:
            for _ in executor.map(simulation_wrapper, tasks, chunksize=20):
                pass
    else:
        # Useful for debugging / notebooks
        init_worker(C_idx, L_idx, W_idx,
                    [C_lower_val, L_lower_val, W_lower_val],
                    [C_upper_val, L_upper_val, W_upper_val])
        for args in tasks:
            simulation_wrapper(args)

    return samples
