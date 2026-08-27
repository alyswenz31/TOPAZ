"""Model adapter for the AABC-median CROCKER comparison step."""

from pathlib import Path

import numpy as np

from Modules_General.ABC3_compute_losses import compute_crocker_error


def _trajectory_to_crocker(trajectory_path):
    """Convert a model-specific trajectory into a CROCKER array."""
    raise NotImplementedError(
        "Provide trajectory_to_crocker=... or implement _trajectory_to_crocker "
        "in Modules_you_replace/AABC6_run_ABC_crocker.py"
    )


def run_ABC_crocker_aabc(
    tda_crocker_angles_path,
    df_aabc_path,
    aabc_crocker_angles_path,
    crocker_difference_aabc_path,
    trajectory_to_crocker=None,
    crocker_plotter=None,
):
    """Create, validate, and save the AABC CROCKER comparison.

    ``trajectory_to_crocker`` receives ``df_aabc_path`` and must return a
    CROCKER array calculated with the same settings as the ground truth.
    ``crocker_plotter`` is optional and receives ``(target, estimate)``.
    """
    target_path = Path(tda_crocker_angles_path)
    trajectory_path = Path(df_aabc_path)
    if not target_path.is_file():
        raise FileNotFoundError(f"Missing ground-truth CROCKER: {target_path}")
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Missing AABC trajectory: {trajectory_path}")

    converter = _trajectory_to_crocker if trajectory_to_crocker is None else trajectory_to_crocker
    target = np.asarray(np.load(target_path, allow_pickle=True), dtype=float)
    estimate = np.asarray(converter(trajectory_path), dtype=float)
    if target.shape != estimate.shape:
        raise ValueError(
            f"CROCKER shapes differ: ground truth {target.shape}, AABC {estimate.shape}"
        )
    if not np.all(np.isfinite(estimate)):
        raise ValueError("The AABC CROCKER contains non-finite values")

    crocker_path = Path(aabc_crocker_angles_path)
    difference_path = Path(crocker_difference_aabc_path)
    crocker_path.parent.mkdir(parents=True, exist_ok=True)
    difference_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(crocker_path, estimate)
    np.save(difference_path, target - estimate)
    np.save(
        difference_path.with_name("posterior_median_loss_aabc.npy"),
        compute_crocker_error(target, estimate),
    )

    if crocker_plotter is not None:
        crocker_plotter(target, estimate)
    return estimate
