from pathlib import Path
import re

import numpy as np

from Modules_General.ABC3_compute_losses import compute_crocker_error


def _numbered_run_directories(root):
    root = Path(root)
    if not root.is_dir():
        return []
    runs = []
    for path in root.iterdir():
        match = re.fullmatch(r"run_(\d+)", path.name)
        if match and path.is_dir():
            runs.append((int(match.group(1)), path))
    return [path for _, path in sorted(runs)]


def compute_losses_aabc(target_path, abc_path, aabc_path, output_path):
    """Calculate aligned, normalized losses for complete ABC and AABC runs."""
    target = np.load(target_path, allow_pickle=True)
    records = []
    for method, root, parameter_file in (
        ("ABC", abc_path, "pars.npy"),
        ("AABC", aabc_path, "theta_star.npy"),
    ):
        for run_dir in _numbered_run_directories(root):
            parameter_path = run_dir / parameter_file
            crocker_path = run_dir / "crocker_angles.npy"
            if not parameter_path.is_file() or not crocker_path.is_file():
                continue
            theta = np.asarray(np.load(parameter_path), dtype=float).reshape(-1)
            candidate = np.load(crocker_path, allow_pickle=True)
            records.append((method, theta, compute_crocker_error(target, candidate)))
    if not records:
        raise FileNotFoundError("No complete ABC or AABC runs were found")
    methods = np.asarray([record[0] for record in records])
    parameters = np.vstack([record[1] for record in records])
    losses = np.asarray([record[2] for record in records], dtype=float)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, methods=methods, parameters=parameters, losses=losses)
    return methods, parameters, losses
