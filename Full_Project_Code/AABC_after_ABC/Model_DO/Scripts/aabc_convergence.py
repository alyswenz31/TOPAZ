"""Posterior-based convergence diagnostics for cumulative AABC checkpoints."""

from __future__ import annotations

import os
import re
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

from Scripts.aabc_utils import PARAM_SCALE, write_json


QUANTILES = np.array([0.05, 0.25, 0.5, 0.75, 0.95])


def symmetric_relative_percent(old: float, new: float) -> float:
    denominator = max((abs(old) + abs(new)) / 2.0, 1e-12)
    return abs(new - old) / denominator * 100.0


def checkpoint_number(name: str) -> int | None:
    match = re.fullmatch(r"run_(\d+)", name)
    return int(match.group(1)) if match else None


def load_checkpoints(folder: str | os.PathLike) -> list[tuple[int, np.ndarray]]:
    checkpoints = []
    for entry in Path(folder).iterdir():
        run_number = checkpoint_number(entry.name)
        accepted_path = entry / "accepted_params.npy"
        manifest_path = entry / "checkpoint_manifest.json"
        if (
            run_number is None
            or not entry.is_dir()
            or not accepted_path.exists()
            or not manifest_path.exists()
        ):
            continue
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        samples = np.load(accepted_path)
        if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) == 0:
            continue
        checkpoints.append((int(manifest["aabc_samples"]), samples))
    return sorted(checkpoints, key=lambda item: item[0])



# Folder-derived convergence plot title
def build_plot_title(folder: str | Path) -> str:
    """Construct the plot title from the model, case, and Widx folders."""
    script_path = Path(__file__).resolve()

    model_name = next(
        (
            part
            for part in script_path.parts
            if part in {"Model_AL", "Model_DO"}
        ),
        None,
    )

    case_match = next(
        (
            re.fullmatch(r"Cidx_(\d+)_Lidx_(\d+)", part)
            for part in script_path.parts
            if part.startswith("Cidx_")
        ),
        None,
    )

    w_match = re.fullmatch(r"Widx_(\d+)", Path(folder).name)

    if model_name is None:
        raise ValueError(
            f"Could not determine Model_AL or Model_DO from {script_path}"
        )

    if case_match is None:
        raise ValueError(
            f"Could not determine Cidx and Lidx from {script_path}"
        )

    if w_match is None:
        raise ValueError(
            f"Could not determine Widx from folder {folder!r}"
        )

    c_index = int(case_match.group(1))
    l_index = int(case_match.group(2))
    w_index = int(w_match.group(1))

    c_value = (c_index + 1) / 10
    l_value = (l_index + 1) / 10
    w_value = w_index / 100

    # Show Widx_00 as 0.0, but retain the second decimal in Widx_05.
    w_text = "0.0" if w_index == 0 else f"{w_value:.2f}"

    display_model_name = model_name.replace("_", " ")

    return (
        f"{display_model_name}: "
        f"C={c_value:.1f}, "
        f"L={l_value:.1f}, "
        f"W={w_text}"
    )


def analyze_posterior_convergence(
    folder: str | os.PathLike,
    model_label: str,
    threshold: float = 0.01,
    required_consecutive: int = 3,
) -> dict:
    checkpoints = load_checkpoints(folder)
    comparisons = []
    consecutive = 0

    for (old_n, old), (new_n, new) in zip(checkpoints, checkpoints[1:]):
        old_quantiles = np.quantile(old, QUANTILES, axis=0)
        new_quantiles = np.quantile(new, QUANTILES, axis=0)
        active_parameters = PARAM_SCALE > 0

        quantile_change = (
            np.abs(
                new_quantiles[:, active_parameters]
                - old_quantiles[:, active_parameters]
            )
            / PARAM_SCALE[active_parameters]
        )

        marginal_wasserstein = np.array([
            wasserstein_distance(old[:, index], new[:, index])
            / PARAM_SCALE[index]
            for index in np.flatnonzero(active_parameters)
        ])
        passed = bool(
            np.max(quantile_change) < threshold
            and np.max(marginal_wasserstein) < threshold
        )
        consecutive = consecutive + 1 if passed else 0
        comparisons.append(
            {
                "old_aabc_samples": old_n,
                "new_aabc_samples": new_n,
                "max_quantile_change": float(np.max(quantile_change)),
                "quantile_change_by_parameter": np.max(
                    quantile_change, axis=0
                ).tolist(),
                "wasserstein_by_parameter": marginal_wasserstein.tolist(),
                "passed": passed,
                "consecutive_passes": consecutive,
            }
        )

    converged = bool(
        comparisons
        and comparisons[-1]["consecutive_passes"] >= required_consecutive
    )
    result = {
        "model": model_label,
        "threshold": threshold,
        "required_consecutive": required_consecutive,
        "converged": converged,
        "checkpoints": [number for number, _ in checkpoints],
        "comparisons": comparisons,
    }

    suffix = Path(folder).name.split("_")[-1]
    write_json(Path(folder) / f"posterior_convergence_{suffix}.json", result)

    if comparisons:
        x_values = [item["new_aabc_samples"] for item in comparisons]
        quantile_values = [item["max_quantile_change"] for item in comparisons]
        wasserstein_values = [
            max(item["wasserstein_by_parameter"]) for item in comparisons
        ]
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.plot(x_values, quantile_values, marker="o", label="Max quantile change")
        axis.plot(x_values, wasserstein_values, marker="o", label="Max Wasserstein")
        axis.axhline(threshold, color="black", linestyle="--", label="Threshold")
        axis.set_xlabel("Cumulative AABC samples")
        axis.set_ylabel("Parameter-range-scaled change")
        axis.set_title(build_plot_title(folder))
        axis.grid(True)
        axis.legend()
        figure.tight_layout()
        figure.savefig(Path(folder) / "posterior_convergence.pdf")
        plt.close(figure)

    return result


def write_secondary_rss_diagnostic(folder: str | os.PathLike) -> None:
    """Retain RSS change as a secondary diagnostic with a safe denominator."""
    values = []
    for entry in Path(folder).iterdir():
        run_number = checkpoint_number(entry.name)
        if run_number is None or not entry.is_dir():
            continue
        path = entry / f"bic_results_{run_number}.npy"
        if not path.exists():
            continue
        try:
            result = np.load(path, allow_pickle=True)
            values.append((run_number, float(result[1])))
        except (OSError, ValueError, IndexError):
            continue

    values.sort(key=lambda item: item[0])
    relative = [
        symmetric_relative_percent(old[1], new[1])
        for old, new in zip(values, values[1:])
    ]
    suffix = Path(folder).name.split("_")[-1]
    np.save(Path(folder) / f"rss_relative_changes_{suffix}.npy", relative)
