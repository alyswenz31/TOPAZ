"""Compute versioned ABC and AABC Crocker losses for the active AL pipeline."""

from __future__ import annotations

import concurrent.futures
import re
import sys
from pathlib import Path

import numpy as np

from Scripts.aabc_utils import (
    AABC_VERSION,
    ABC_SAMPLE_SIZE,
    LOSS_VERSION,
    compute_crocker_loss,
)


DATA_COLS = ("x", "y", "angle")
MAX_ABC_SAMPLE = ABC_SAMPLE_SIZE
PARS_IDC = [(6, 24, 0), (6, 24, 5)]


def numbered_run_directories(folder: Path) -> list[Path]:
    runs = []
    for entry in folder.iterdir():
        match = re.fullmatch(r"run_(\d+)", entry.name)
        if match and entry.is_dir():
            runs.append((int(match.group(1)), entry))
    return [entry for _, entry in sorted(runs, key=lambda item: item[0])]


def true_crocker_path(w_index: int) -> Path:
    suffix = "angles" if "angle" in DATA_COLS else "velocities"
    return Path(f"Widx_{w_index:02d}") / f"crocker_{suffix}.npy"


def base_loss_path(w_index: int) -> Path:
    return (
        Path(f"Widx_{w_index:02d}")
        / f"sample_losses_angles_{ABC_SAMPLE_SIZE}_{LOSS_VERSION}.npy"
    )


def aabc_batch_loss_path(w_index: int, batch_label: int) -> Path:
    return (
        Path(f"Widx_{w_index:02d}")
        / f"aabc_loss_batch_{batch_label}_{AABC_VERSION}_{LOSS_VERSION}.npy"
    )


def build_base_losses(w_index: int, true_crocker: np.ndarray) -> dict:
    output_path = base_loss_path(w_index)

    existing_paths = [
        output_path,  # Current versioned filename
        Path(f"Widx_{w_index:02d}")
        / f"sample_losses_angles_{ABC_SAMPLE_SIZE}.npy",
    ]

    for existing_path in existing_paths:
        if existing_path.exists():
            print(f"Using existing base losses: {existing_path}")
            return np.load(existing_path, allow_pickle=True).item()

    # Only use sample_21000 if no existing loss file was found.
    losses = {}
    sample_folder = Path(f"sample_{MAX_ABC_SAMPLE}")

    for sample_number in range(1, ABC_SAMPLE_SIZE + 1):
        run_dir = sample_folder / f"run_{sample_number}"
        parameter_path = run_dir / "pars.npy"
        prediction_path = run_dir / "crocker_angles.npy"

        if not parameter_path.exists() or not prediction_path.exists():
            continue

        parameters = np.load(parameter_path, allow_pickle=True)
        prediction = np.load(prediction_path, allow_pickle=True)

        losses[str(sample_number)] = {
            "sampled_pars": parameters,
            "loss": compute_crocker_loss(true_crocker, prediction),
        }

    if not losses:
        checked = ", ".join(str(path) for path in existing_paths)
        raise RuntimeError(
            f"No existing base-loss file found. Checked: {checked}. "
            f"No base ABC simulations found in {sample_folder} either."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, losses)
    print(f"Saved {len(losses)} base-loss records to {output_path}")
    return losses


def build_aabc_batch_losses(
    w_index: int,
    batch_label: int,
    true_crocker: np.ndarray,
) -> Path:
    output_path = aabc_batch_loss_path(w_index, batch_label)
    if output_path.exists():
        print(f"Using existing {output_path}")
        return output_path

    source_folder = Path(f"sample_aabc_{batch_label}")
    if not source_folder.is_dir():
        raise FileNotFoundError(f"Missing AABC sample folder: {source_folder}")

    batch_losses = {}
    run_directories = numbered_run_directories(source_folder)
    if len(run_directories) != batch_label:
        raise RuntimeError(
            f"Expected {batch_label} run directories in {source_folder}, "
            f"found {len(run_directories)}"
        )
    for local_index, run_dir in enumerate(run_directories, start=1):
        parameter_path = run_dir / "theta_star.npy"
        prediction_path = run_dir / "crocker_angles.npy"
        if not parameter_path.exists() or not prediction_path.exists():
            continue

        theta = np.asarray(np.load(parameter_path, allow_pickle=True), dtype=float)
        prediction = np.load(prediction_path, allow_pickle=True)
        batch_losses[str(local_index)] = {
            "sampled_pars": theta,
            "loss": compute_crocker_loss(true_crocker, prediction),
        }

    if not batch_losses:
        raise RuntimeError(f"No complete AABC runs found in {source_folder}")
    np.save(output_path, batch_losses)
    print(f"Saved {len(batch_losses)} records to {output_path}")
    return output_path


def run_compute_distance(args: tuple[tuple[int, int, int], int]) -> None:
    (_, _, w_index), batch_label = args
    true_crocker = np.load(true_crocker_path(w_index), allow_pickle=True)
    build_base_losses(w_index, true_crocker)
    build_aabc_batch_losses(w_index, batch_label, true_crocker)


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        raise SystemExit("Usage: python ABC_S03_...py AABC_BATCH_LABEL")
    batch_label = int(arguments[0])

    tasks = [(parameter_indices, batch_label) for parameter_indices in PARS_IDC]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
        for _ in executor.map(run_compute_distance, tasks):
            pass


if __name__ == "__main__":
    main()
