from pathlib import Path

import numpy as np


def _crocker_loss(candidate, target):
    candidate = np.asarray(candidate, dtype=float)
    target = np.asarray(target, dtype=float)
    if candidate.shape != target.shape:
        raise ValueError(f"CROCKER shape mismatch: {candidate.shape} != {target.shape}")
    return np.linalg.norm(candidate.reshape(-1) - target.reshape(-1))


def compute_losses_aabc(target_path, abc_path, aabc_path, output_path):
    """Calculate and save aligned losses/parameters for ABC and AABC runs."""
    target = np.load(target_path)
    records = []
    for method, root, parameter_file in (
        ("ABC", Path(abc_path), "pars.npy"),
        ("AABC", Path(aabc_path), "theta_star.npy"),
    ):
        if not root.exists():
            continue
        run_dirs = sorted(
            (path for path in root.iterdir() if path.is_dir() and path.name.startswith("run_")),
            key=lambda path: int(path.name.rsplit("_", 1)[-1]),
        )
        for run_dir in run_dirs:
            theta = np.load(run_dir / parameter_file).reshape(-1)
            candidate = np.load(run_dir / "crocker_angles.npy")
            records.append((method, theta, _crocker_loss(candidate, target)))

    if not records:
        raise FileNotFoundError("No ABC or AABC run directories were found")
    methods = np.asarray([record[0] for record in records])
    parameters = np.vstack([record[1] for record in records])
    losses = np.asarray([record[2] for record in records])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, methods=methods, parameters=parameters, losses=losses)
    return methods, parameters, losses
