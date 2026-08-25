"""Combine collision-free AABC batches and construct corrected AL posteriors."""

from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Scripts.aabc_utils import (
    AABC_VERSION,
    ABC_SAMPLE_SIZE,
    LOSS_VERSION,
    unpack_loss_records,
    write_json,
)


CS = np.linspace(0.1, 3.0, 30)
LS = np.linspace(0.1, 3.0, 30)
WS = np.linspace(0.0, 0.1, 11)
PARS_IDC = [(6, 24, 0), (6, 24, 5)]


def discover_batch_files(folder: Path) -> list[tuple[int, Path]]:
    pattern = re.compile(
        rf"aabc_loss_batch_(\d+)_{re.escape(AABC_VERSION)}_"
        rf"{re.escape(LOSS_VERSION)}\.npy"
    )
    batches = []
    for entry in folder.iterdir():
        match = pattern.fullmatch(entry.name)
        if match:
            batches.append((int(match.group(1)), entry))
    return sorted(batches, key=lambda item: item[0])


def find_base_loss_path(folder: Path) -> Path:
    """Prefer versioned base losses, then accept the original unversioned file."""
    candidates = [
        folder / f"sample_losses_angles_{ABC_SAMPLE_SIZE}_{LOSS_VERSION}.npy",
        folder / f"sample_losses_angles_{ABC_SAMPLE_SIZE}.npy",
    ]

    for candidate in candidates:
        if candidate.is_file():
            print(f"Using base losses: {candidate}")
            return candidate

    checked = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Missing base ABC losses. Checked:\n"
        f"  {checked}\n"
        "Run Step 3 or copy the matching base-loss file into this folder."
    )


def load_all_losses(folder: Path) -> tuple[dict, dict, dict]:
    base_path = find_base_loss_path(folder)
    base_data = np.load(base_path, allow_pickle=True).item()
    data = dict(base_data)
    next_index = max(map(int, data.keys()), default=0) + 1
    batch_manifest = []
    num_aabc_samples = 0

    for batch_label, batch_path in discover_batch_files(folder):
        batch = np.load(batch_path, allow_pickle=True).item()
        before = len(data)
        for old_key in sorted(batch, key=lambda value: int(value)):
            data[str(next_index)] = batch[old_key]
            next_index += 1
        added = len(data) - before
        if added != len(batch):
            raise RuntimeError(f"Failed to append every record from {batch_path}")
        num_aabc_samples += added
        batch_manifest.append(
            {
                "label": batch_label,
                "filename": batch_path.name,
                "records": added,
            }
        )

    if len(data) != len(base_data) + num_aabc_samples:
        raise RuntimeError("Combined record count does not match base plus AABC counts")

    counts = {
        "abc_samples": len(base_data),
        "aabc_samples": num_aabc_samples,
        "total_samples": len(data),
        "unique_keys": len(set(data)),
    }
    return data, counts, {"batches": batch_manifest}


def build_map(
    accepted_parameters: np.ndarray,
    c_grid: np.ndarray,
    l_grid: np.ndarray,
    w_grid: np.ndarray,
) -> np.ndarray:
    result = np.zeros((len(c_grid), len(l_grid), len(w_grid)), dtype=int)
    for c_value, l_value, w_value in accepted_parameters:
        c_index = np.argmin(np.abs(c_grid - c_value))
        l_index = np.argmin(np.abs(l_grid - l_value))
        w_index = np.argmin(np.abs(w_grid - w_value))
        result[c_index, l_index, w_index] += 1
    return result


def plot_map(
    sample_map: np.ndarray,
    c_grid: np.ndarray,
    l_grid: np.ndarray,
    w_grid: np.ndarray,
    truth: np.ndarray,
    median: np.ndarray,
    num_aabc_samples: int,
    title_prefix: str,
    output_directory: Path,
    tag: str,
) -> None:
    c_mesh, l_mesh = np.meshgrid(c_grid, l_grid, indexing="ij")
    true_w_index = np.argmin(np.abs(w_grid - truth[2]))
    median_w_index = np.argmin(np.abs(w_grid - median[2]))
    output_directory.mkdir(parents=True, exist_ok=True)

    for each_w in range(len(w_grid)):
        figure, axis = plt.subplots(figsize=(6, 6), dpi=400)
        axis.contourf(c_mesh, l_mesh, sample_map[:, :, each_w])
        if each_w == true_w_index:
            axis.scatter(
                truth[0], truth[1], c="w", edgecolor="k", marker="*", s=500,
                label="True",
            )
        if each_w == median_w_index:
            axis.scatter(
                median[0], median[1], c="k", marker="o", s=180,
                label="ABC median",
            )
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"{title_prefix}\n"
            f"True: C={truth[0]:.3f}, L={truth[1]:.3f}, W={truth[2]:.3f}\n"
            f"Median: C={median[0]:.3f}, L={median[1]:.3f}, W={median[2]:.3f}\n"
            f"W plot value={w_grid[each_w]:.3f}"
        )
        axis.set_xlabel("C")
        axis.set_ylabel("L")
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
        figure.tight_layout()
        figure.savefig(
            output_directory
            / f"ABC_{num_aabc_samples}_{tag}_at_w{each_w:02d}.pdf",
            bbox_inches="tight",
        )
        plt.close(figure)


def process_case(c_index: int, l_index: int, w_index: int) -> None:
    truth = np.array([CS[c_index], LS[l_index], WS[w_index]])
    folder = Path(f"Widx_{w_index:02d}")
    data, counts, manifest = load_all_losses(folder)
    parameters, losses = unpack_loss_records(data)

    # Select exactly the lowest-loss 1% from all combined finite samples.
    valid_indices = np.flatnonzero(np.isfinite(losses))
    if valid_indices.size == 0:
        raise RuntimeError(f"No finite losses found in {folder}")

    number_to_accept = max(1, int(np.ceil(0.01 * valid_indices.size)))
    ranked_valid_indices = valid_indices[
        np.argsort(losses[valid_indices], kind="stable")
    ]
    accepted_indices = ranked_valid_indices[:number_to_accept]
    threshold = losses[accepted_indices[-1]]

    print(
        f"{folder}: accepted {len(accepted_indices)} of {len(valid_indices)} "
        f"finite samples ({100 * len(accepted_indices) / len(valid_indices):.3f}%), "
        f"threshold={threshold}"
    )

    accepted_parameters = parameters[accepted_indices]
    accepted_losses = losses[accepted_indices]
    median = np.median(accepted_parameters, axis=0)
    quantile_levels = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    quantiles = np.quantile(accepted_parameters, quantile_levels, axis=0)

    # Use the true total record count for a collision-resistant corrected run name.
    run_dir = folder / f"run_{counts['total_samples']}"
    if run_dir.exists() and not (run_dir / "checkpoint_manifest.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite legacy output directory {run_dir}. "
            "Move or rename it before creating the corrected checkpoint."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "sample_losses_angles.npy", data)
    np.save(run_dir / "accepted_params.npy", accepted_parameters)
    np.save(run_dir / "accepted_losses.npy", accepted_losses)
    np.save(run_dir / "medians.npy", median)
    np.savetxt(run_dir / "medians.txt", median, delimiter=",")
    np.save(run_dir / "posterior_quantiles.npy", quantiles)
    np.save(run_dir / "acceptance_threshold.npy", threshold)

    checkpoint_manifest = {
        **counts,
        **manifest,
        "loss_version": LOSS_VERSION,
        "aabc_version": AABC_VERSION,
        "acceptance_percentile": 1.0,
        "acceptance_reference": "all_finite_combined_losses",
        "finite_samples": len(valid_indices),
        "acceptance_threshold": threshold,
        "accepted_samples": len(accepted_parameters),
        "quantile_levels": quantile_levels,
        "truth": truth,
        "median": median,
    }
    write_json(run_dir / "checkpoint_manifest.json", checkpoint_manifest)

    c_grid_11 = np.linspace(0.4, 1.3, 10)
    l_grid_11 = np.linspace(1.0, 3.0, 11)
    w_grid_11 = np.linspace(0.0, 0.0, 1)
    map_11 = build_map(accepted_parameters, c_grid_11, l_grid_11, w_grid_11)
    posterior_11 = map_11.astype(float) / len(accepted_parameters)
    np.save(run_dir / "sample_counts_11.npy", map_11)
    np.save(run_dir / "posteriors_11.npy", posterior_11)
    if not np.isclose(posterior_11.sum(), 1.0):
        raise RuntimeError("Posterior 11 did not normalize to one")

    c_grid_30 = np.linspace(0.4, 1.3, 10)
    l_grid_30 = np.linspace(1.0, 3.0, 21)
    w_grid_30 = np.linspace(0.0, 0.0, 1)
    map_30 = build_map(accepted_parameters, c_grid_30, l_grid_30, w_grid_30)
    posterior_30 = map_30.astype(float) / len(accepted_parameters)
    np.save(run_dir / "sample_counts_30.npy", map_30)
    np.save(run_dir / "posteriors_30.npy", posterior_30)
    if not np.isclose(posterior_30.sum(), 1.0):
        raise RuntimeError("Posterior 30 did not normalize to one")

    plot_map(
        map_11, c_grid_11, l_grid_11, w_grid_11, truth, median,
        counts["aabc_samples"], "Posterior 11", run_dir / "ABC_angle_plot_11",
        "11",
    )
    plot_map(
        map_30, c_grid_30, l_grid_30, w_grid_30, truth, median,
        counts["aabc_samples"], "Posterior 30", run_dir / "ABC_angle_plot_30",
        "30",
    )
    print(f"Saved corrected checkpoint {run_dir}: {counts}")


def main() -> None:
    for indices in PARS_IDC:
        process_case(*indices)
    print("Done")


if __name__ == "__main__":
    main()
