from pathlib import Path

import numpy as np


def compute_crocker_error(true_metric, pred_metric):
    """Return an equally weighted, scale-normalized MSE across components."""
    true_values = np.asarray(true_metric, dtype=float)
    pred_values = np.asarray(pred_metric, dtype=float)
    if true_values.shape != pred_values.shape:
        raise ValueError(
            f"CROCKER shapes differ: {true_values.shape} != {pred_values.shape}"
        )
    if true_values.ndim < 2:
        raise ValueError("CROCKER arrays must have at least two dimensions")
    if not np.all(np.isfinite(true_values)) or not np.all(np.isfinite(pred_values)):
        return float("nan")
    if true_values.ndim > 2:
        losses = []
        for component in range(true_values.shape[-1]):
            target = true_values[..., component]
            candidate = pred_values[..., component]
            scale = max(float(np.max(np.abs(target))), 1.0)
            losses.append(float(np.mean(((candidate - target) / scale) ** 2)))
        return float(np.mean(losses))
    scale = max(float(np.max(np.abs(true_values))), 1.0)
    return float(np.mean(((pred_values - true_values) / scale) ** 2))


def compute_losses(num_samples, tda_crocker_angles_path,
                   abc_crocker_angles_and_pars_path, sample_losses_angles_path):
    """Calculate normalized losses for up to ``num_samples`` complete ABC runs."""
    target = np.load(tda_crocker_angles_path, allow_pickle=True)
    root = Path(abc_crocker_angles_and_pars_path)
    losses = {}
    for sample_number in range(1, num_samples + 1):
        run_dir = root / f"run_{sample_number}"
        parameter_path = run_dir / "pars.npy"
        crocker_path = run_dir / "crocker_angles.npy"
        if not parameter_path.is_file() or not crocker_path.is_file():
            continue
        losses[str(sample_number)] = {
            "sampled_pars": np.load(parameter_path, allow_pickle=True),
            "loss": compute_crocker_error(
                target, np.load(crocker_path, allow_pickle=True)
            ),
        }
    if not losses:
        raise FileNotFoundError(f"No complete ABC runs found under {root}")
    output_path = Path(sample_losses_angles_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, losses)
    return losses
