from pathlib import Path

import numpy as np


def run_samples_aabc(
    reference_path,
    output_path,
    parameter_bounds,
    num_aabc_samples,
    k=5,
    seed=2026,
):
    """Generate model-agnostic AABC parameter/CROCKER samples."""
    reference_path = Path(reference_path)
    output_path = Path(output_path)
    theta_ref = np.load(reference_path / "all_params.npy")
    x_ref = np.load(reference_path / "all_crockers_flattened.npy")
    crocker_shape = tuple(np.load(reference_path / "crocker_shape.npy").astype(int))
    bounds = np.asarray(parameter_bounds, dtype=float)

    if theta_ref.ndim != 2 or x_ref.ndim != 2 or len(theta_ref) != len(x_ref):
        raise ValueError("Reference parameter and CROCKER arrays must be aligned 2-D arrays")
    if bounds.shape != (theta_ref.shape[1], 2):
        raise ValueError("parameter_bounds must have one [lower, upper] row per parameter")
    if np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Every lower parameter bound must be smaller than its upper bound")
    if not 1 <= k < len(theta_ref):
        raise ValueError("k must be at least 1 and smaller than the reference sample count")

    rng = np.random.default_rng(seed)
    sampled_params = rng.uniform(
        bounds[:, 0], bounds[:, 1], size=(num_aabc_samples, bounds.shape[0])
    )
    output_path.mkdir(parents=True, exist_ok=True)

    for index, theta_star in enumerate(sampled_params, start=1):
        all_distances = np.linalg.norm(theta_ref - theta_star, axis=1)
        neighbor_indices = np.argsort(all_distances)[: k + 1]
        distances = all_distances[neighbor_indices]
        bandwidth = distances[-1]
        if bandwidth <= 0:
            weights = np.ones(k)
        else:
            ratios = distances[:k] / bandwidth
            weights = np.maximum(0.75 * (1.0 - ratios**2), 1e-12)
        mixture = rng.dirichlet(weights)
        x_star = mixture @ x_ref[neighbor_indices[:k]]

        run_path = output_path / f"run_{index}"
        run_path.mkdir(parents=True, exist_ok=True)
        np.save(run_path / "theta_star.npy", theta_star)
        np.save(run_path / "crocker_angles.npy", x_star.reshape(crocker_shape))

    np.save(output_path / "sampled_params.npy", sampled_params)
    return sampled_params
