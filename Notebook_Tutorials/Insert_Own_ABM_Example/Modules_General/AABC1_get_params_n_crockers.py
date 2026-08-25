from pathlib import Path

import numpy as np


def get_params_n_crockers_aabc(samples_path, output_path, num_parameters=None):
    """Stack parameters and flattened CROCKER arrays from ``run_*`` folders."""
    samples_path = Path(samples_path)
    output_path = Path(output_path)
    run_dirs = sorted(
        (path for path in samples_path.iterdir() if path.is_dir() and path.name.startswith("run_")),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories found in {samples_path}")

    params, crockers = [], []
    crocker_shape = None
    for run_dir in run_dirs:
        theta = np.asarray(np.load(run_dir / "pars.npy"), dtype=float).reshape(-1)
        crocker = np.asarray(np.load(run_dir / "crocker_angles.npy"), dtype=float)
        if num_parameters is not None and theta.size != num_parameters:
            raise ValueError(
                f"{run_dir}: expected {num_parameters} parameters, found {theta.size}"
            )
        if crocker_shape is None:
            crocker_shape = crocker.shape
        elif crocker.shape != crocker_shape:
            raise ValueError(
                f"{run_dir}: CROCKER shape {crocker.shape} does not match {crocker_shape}"
            )
        params.append(theta)
        crockers.append(crocker.reshape(-1))

    output_path.mkdir(parents=True, exist_ok=True)
    params = np.vstack(params)
    crockers = np.vstack(crockers)
    np.save(output_path / "all_params.npy", params)
    np.save(output_path / "all_crockers_flattened.npy", crockers)
    np.save(output_path / "crocker_shape.npy", np.asarray(crocker_shape, dtype=int))
    return params, crockers, crocker_shape
