"""Shared constants and numerical utilities for the AABC pipeline."""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path

import numpy as np
PARAM_LOWER = np.array([0.4, 1.0, 0.0], dtype=float)
PARAM_UPPER = np.array([1.3, 3.0, 0.0], dtype=float)
PARAM_SCALE = PARAM_UPPER - PARAM_LOWER
VARYING_PARAMS = PARAM_SCALE > 0.0
ABC_SAMPLE_SIZE = 2_100
ACCEPTANCE_PERCENTILE = 1.0
LOSS_VERSION = "normalized_mse_v1"
AABC_VERSION = "scaled_params_dirichlet20_v1"


def scale_theta(theta: np.ndarray) -> np.ndarray:
    """Scale parameters with nonzero prior ranges to the unit interval."""
    values = np.asarray(theta, dtype=float)
    if values.shape[-1] != 3:
        raise ValueError(f"Expected final parameter dimension 3, got {values.shape}")
    if not np.any(VARYING_PARAMS):
        raise ValueError("At least one parameter must have a nonzero prior range")
    return (
        values[..., VARYING_PARAMS] - PARAM_LOWER[VARYING_PARAMS]
    ) / PARAM_SCALE[VARYING_PARAMS]


def compute_crocker_loss(true_metric: np.ndarray, pred_metric: np.ndarray) -> float:
    """Return an equally weighted, scale-normalized MSE across Betti numbers."""
    true_values = np.asarray(true_metric, dtype=float)
    pred_values = np.asarray(pred_metric, dtype=float)

    if true_values.shape != pred_values.shape:
        raise ValueError(
            f"Crocker shapes differ: {true_values.shape} != {pred_values.shape}"
        )
    if true_values.ndim < 2:
        raise ValueError("Crocker arrays must have at least two dimensions")
    if not np.all(np.isfinite(true_values)) or not np.all(np.isfinite(pred_values)):
        return float("nan")

    # The last dimension indexes Betti numbers for the active pipeline.
    if true_values.ndim > 2:
        component_losses = []
        for betti_index in range(true_values.shape[-1]):
            true_betti = true_values[..., betti_index]
            pred_betti = pred_values[..., betti_index]
            scale = max(float(np.max(np.abs(true_betti))), 1.0)
            component_losses.append(
                float(np.mean(((pred_betti - true_betti) / scale) ** 2))
            )
        return float(np.mean(component_losses))

    scale = max(float(np.max(np.abs(true_values))), 1.0)
    return float(np.mean(((pred_values - true_values) / scale) ** 2))


def record_theta(record: dict) -> np.ndarray:
    """Extract C, L, W from either the legacy six-vector or a three-vector."""
    parameters = np.asarray(record["sampled_pars"], dtype=float)
    if parameters.size >= 6:
        return parameters[3:6]
    if parameters.size == 3:
        return parameters
    raise ValueError(f"Unsupported sampled_pars length: {parameters.size}")


def unpack_loss_records(records: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convert a loss dictionary into finite parameter and loss arrays."""
    parameters = []
    losses = []
    for record in records.values():
        parameters.append(record_theta(record))
        losses.append(float(record["loss"]))

    theta = np.asarray(parameters, dtype=float)
    loss = np.asarray(losses, dtype=float)
    keep = np.isfinite(loss) & np.all(np.isfinite(theta), axis=1)
    return theta[keep], loss[keep]


def fixed_tolerance(folder: str | os.PathLike, base_losses: np.ndarray) -> float:
    """Load or create the fixed ABC tolerance for the current loss definition."""
    tolerance_path = Path(folder) / f"abc_tolerance_{LOSS_VERSION}.npy"
    metadata_path = Path(folder) / f"abc_tolerance_{LOSS_VERSION}.json"

    finite_losses = np.asarray(base_losses, dtype=float)
    finite_losses = finite_losses[np.isfinite(finite_losses)]
    if finite_losses.size == 0:
        raise RuntimeError("The base ABC library contains no finite losses")

    fingerprint = hashlib.sha256(
        np.ascontiguousarray(finite_losses).view(np.uint8)
    ).hexdigest()
    if tolerance_path.exists() and metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("base_loss_sha256") == fingerprint:
            return float(np.load(tolerance_path))

    tolerance = float(np.percentile(finite_losses, ACCEPTANCE_PERCENTILE))
    np.save(tolerance_path, tolerance)
    write_json(
        metadata_path,
        {
            "acceptance_percentile": ACCEPTANCE_PERCENTILE,
            "base_loss_sha256": fingerprint,
            "base_samples": finite_losses.size,
            "loss_version": LOSS_VERSION,
            "tolerance": tolerance,
        },
    )
    return tolerance


def write_json(path: str | os.PathLike, data: dict) -> None:
    """Write JSON containing native Python/NumPy scalar values."""
    def default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Cannot serialize {type(value)!r}")

    with open(path, "w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, default=default)
        stream.write("\n")


def latest_corrected_run(folder: str | os.PathLike) -> tuple[int, Path]:
    """Return the manifest-backed checkpoint with the most AABC samples."""
    candidates = []
    for entry in Path(folder).iterdir():
        manifest_path = entry / "checkpoint_manifest.json"
        if not entry.is_dir() or not manifest_path.exists():
            continue
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("loss_version") != LOSS_VERSION:
            continue
        candidates.append((int(manifest["aabc_samples"]), entry))

    if not candidates:
        raise ValueError(f"No corrected AABC checkpoints found in {folder}")
    _, run_path = max(candidates, key=lambda item: item[0])
    run_number = int(run_path.name.split("_", 1)[1])
    return run_number, run_path
