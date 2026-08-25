from pathlib import Path

import numpy as np


def _load_flattened_crocker_group(root_path):
    root_path = Path(root_path)
    files = sorted(
        root_path.glob("run_*/crocker_angles.npy"),
        key=lambda path: int(path.parent.name.rsplit("_", 1)[-1]),
    )
    if not files:
        raise FileNotFoundError(f"No run_*/crocker_angles.npy files found under {root_path}")
    arrays = [np.asarray(np.load(path), dtype=float) for path in files]
    expected_shape = arrays[0].shape
    if any(array.shape != expected_shape for array in arrays):
        raise ValueError(f"All CROCKER arrays in {root_path} must have shape {expected_shape}")
    return np.vstack([array.reshape(1, -1) for array in arrays])


def statistical_verification(
    group_a_path,
    group_b_path,
    output_csv,
    permutations=999,
    seed=2026,
):
    """Run PERMANOVA, energy-distance, and MMD two-sample tests."""
    try:
        import pandas as pd
        from hyppo.ksample import Energy, MMD
        from sklearn.metrics import pairwise_distances
        from skbio.stats.distance import DistanceMatrix, permanova
    except ImportError as error:
        raise ImportError(
            "Statistical verification requires pandas, scikit-learn, scikit-bio, and hyppo."
        ) from error

    group_a = _load_flattened_crocker_group(group_a_path)
    group_b = _load_flattened_crocker_group(group_b_path)
    if group_a.shape[1] != group_b.shape[1]:
        raise ValueError("The two groups must have the same flattened CROCKER dimension")

    data = np.vstack([group_a, group_b])
    labels = np.array(["model_a"] * len(group_a) + ["model_b"] * len(group_b))
    distance_matrix = DistanceMatrix(pairwise_distances(data, metric="euclidean"))
    perma = permanova(distance_matrix, labels, permutations=permutations, seed=seed)
    energy_stat, energy_p = Energy().test(
        group_a, group_b, reps=permutations, workers=1, random_state=seed
    )
    mmd_stat, mmd_p = MMD().test(
        group_a, group_b, reps=permutations, workers=1, random_state=seed
    )

    results = pd.DataFrame(
        [
            {
                "test": "PERMANOVA",
                "statistic": perma["test statistic"],
                "p_value": perma["p-value"],
            },
            {"test": "Energy", "statistic": energy_stat, "p_value": energy_p},
            {"test": "MMD", "statistic": mmd_stat, "p_value": mmd_p},
        ]
    )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    return results
