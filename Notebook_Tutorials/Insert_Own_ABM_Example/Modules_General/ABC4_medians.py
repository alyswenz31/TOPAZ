from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _validate_inputs(parameters, losses, parameter_grids, parameter_names):
    parameters = np.asarray(parameters, dtype=float)
    losses = np.asarray(losses, dtype=float)
    grids = [np.asarray(grid, dtype=float) for grid in parameter_grids]
    if parameters.ndim != 2 or parameters.shape[1] != len(grids):
        raise ValueError("Parameter columns must match the number of parameter grids")
    if losses.shape != (len(parameters),):
        raise ValueError("There must be exactly one loss per parameter vector")
    if len(grids) != 3:
        raise ValueError("Posterior slice plotting currently requires three parameters")
    if any(grid.ndim != 1 or len(grid) == 0 for grid in grids):
        raise ValueError("Every parameter grid must be a non-empty 1-D array")
    if parameter_names is None:
        parameter_names = [f"Par{i + 1}" for i in range(len(grids))]
    if len(parameter_names) != len(grids):
        raise ValueError("parameter_names must match parameter_grids")
    keep = np.isfinite(losses) & np.all(np.isfinite(parameters), axis=1)
    if not np.any(keep):
        raise ValueError("No finite parameter/loss records were found")
    return parameters[keep], losses[keep], grids, list(parameter_names)


def _posterior_from_arrays(parameters, losses, posterior_path, median_path,
                           parameter_grids, parameter_names=None,
                           truth_indices=None, acceptance_percentile=1.0):
    parameters, losses, grids, names = _validate_inputs(
        parameters, losses, parameter_grids, parameter_names
    )
    number_to_accept = max(
        1, int(np.ceil((acceptance_percentile / 100.0) * len(losses)))
    )
    accepted_indices = np.argsort(losses, kind="stable")[:number_to_accept]
    accepted = parameters[accepted_indices]
    medians = np.median(accepted, axis=0)

    median_path = Path(median_path)
    median_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(median_path, medians)
    np.save(median_path.with_name("accepted_params" + median_path.suffix), accepted)

    counts = np.zeros(tuple(len(grid) for grid in grids), dtype=int)
    for theta in accepted:
        index = tuple(np.argmin(np.abs(grid - value)) for grid, value in zip(grids, theta))
        counts[index] += 1
    posterior = counts.astype(float) / len(accepted)
    np.save(median_path.with_name("posterior" + median_path.suffix), posterior)

    truth = None
    if truth_indices is not None:
        if len(truth_indices) != 3:
            raise ValueError("truth_indices must contain one index per parameter")
        truth = np.array([grid[index] for grid, index in zip(grids, truth_indices)])

    posterior_path = Path(posterior_path)
    posterior_path.mkdir(parents=True, exist_ok=True)
    first_mesh, second_mesh = np.meshgrid(grids[0], grids[1], indexing="ij")
    median_slice = int(np.argmin(np.abs(grids[2] - medians[2])))
    truth_slice = None if truth is None else int(np.argmin(np.abs(grids[2] - truth[2])))
    for slice_index, slice_value in enumerate(grids[2]):
        figure, axis = plt.subplots(figsize=(6, 6), dpi=200)
        axis.contourf(first_mesh, second_mesh, counts[:, :, slice_index])
        if truth is not None and slice_index == truth_slice:
            axis.scatter(truth[0], truth[1], c="white", edgecolor="black",
                         marker="*", s=300, label="True")
        if slice_index == median_slice:
            axis.scatter(medians[0], medians[1], c="black", marker="o",
                         s=120, label="Posterior median")
        axis.set_xlabel(names[0])
        axis.set_ylabel(names[1])
        axis.set_title(f"{names[2]} = {slice_value:g}")
        axis.set_aspect("equal", adjustable="box")
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
        figure.tight_layout()
        figure.savefig(
            posterior_path / f"posterior_density_slice_at_{names[2]}_{slice_index:02d}.png",
            bbox_inches="tight",
        )
        plt.close(figure)
    return medians


def compute_medians_and_densities(sample_losses_angles_path,
                                  abc_posterior_densities_path, median_path,
                                  parameter_grids, parameter_names=None,
                                  truth_indices=None, acceptance_percentile=1.0):
    records = np.load(sample_losses_angles_path, allow_pickle=True).item()
    parameters = []
    losses = []
    for record in records.values():
        parameters.append(np.asarray(record["sampled_pars"], dtype=float).reshape(-1))
        losses.append(float(record["loss"]))
    return _posterior_from_arrays(
        np.vstack(parameters), np.asarray(losses), abc_posterior_densities_path,
        median_path, parameter_grids, parameter_names, truth_indices,
        acceptance_percentile,
    )
