import numpy as np

from Modules_General.ABC4_medians import _posterior_from_arrays


def compute_medians_and_densities_aabc(
    sample_losses_angles_aabc_path,
    aabc_posterior_densities_path,
    medians_aabc_path,
    parameter_grids,
    parameter_names=None,
    truth_indices=None,
    acceptance_percentile=1.0,
):
    """Build the combined ABC+AABC posterior and return parameter medians."""
    with np.load(sample_losses_angles_aabc_path, allow_pickle=True) as data:
        parameters = data["parameters"]
        losses = data["losses"]
    return _posterior_from_arrays(
        parameters,
        losses,
        aabc_posterior_densities_path,
        medians_aabc_path,
        parameter_grids,
        parameter_names,
        truth_indices,
        acceptance_percentile,
    )
