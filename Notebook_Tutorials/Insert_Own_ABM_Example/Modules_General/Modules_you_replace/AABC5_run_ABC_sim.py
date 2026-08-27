"""Model adapter for running an ABM at the AABC posterior medians.

Replace ``simulate_abm`` with your model-specific simulation function, or edit
``_simulate_abm`` directly. The surrounding loading, validation, and output
handling mirrors the corresponding CLW AABC step.
"""

from pathlib import Path

import numpy as np


def _simulate_abm(parameters, t0, tf, dt, num_agents):
    """Run one model-specific simulation and return its trajectory dataframe."""
    raise NotImplementedError(
        "Provide simulate_abm=... or implement _simulate_abm in "
        "Modules_you_replace/AABC5_run_ABC_sim.py"
    )


def run_ABC_sim_aabc(
    medians_aabc_path,
    T0,
    TF,
    DT,
    num_agents,
    output_path="./df_AABC.pkl",
    simulate_abm=None,
):
    """Run the user's ABM at the saved AABC medians.

    ``simulate_abm`` must accept ``(parameters, T0, TF, DT, num_agents)`` and
    return a pandas-like dataframe with a ``to_pickle`` method. It may instead
    save ``output_path`` itself and return ``None``.
    """
    if TF <= T0:
        raise ValueError("TF must be greater than T0")
    if DT <= 0:
        raise ValueError("DT must be positive")
    if num_agents <= 0:
        raise ValueError("num_agents must be positive")

    median_path = Path(medians_aabc_path)
    if not median_path.is_file():
        raise FileNotFoundError(f"Missing AABC median file: {median_path}")
    medians = np.asarray(np.load(median_path), dtype=float).reshape(-1)
    if medians.size == 0 or not np.all(np.isfinite(medians)):
        raise ValueError("AABC medians must be a non-empty finite parameter vector")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simulation_function = _simulate_abm if simulate_abm is None else simulate_abm
    trajectory = simulation_function(medians, T0, TF, DT, num_agents)

    if trajectory is not None:
        if not hasattr(trajectory, "to_pickle"):
            raise TypeError("simulate_abm must return a dataframe-like object or None")
        trajectory.to_pickle(output_path)
    elif not output_path.is_file():
        raise RuntimeError(
            "simulate_abm returned None but did not create the requested output file: "
            f"{output_path}"
        )
    return trajectory
