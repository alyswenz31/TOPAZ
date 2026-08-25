import math
import numpy as np
import os
import re

from Scripts.aabc_utils import latest_corrected_run

pars_idc = [(6, 24, 0), (6, 24, 5)]

k = 3  # C, L, W

def get_latest_run(folder):
    run_number, _ = latest_corrected_run(folder)
    return run_number



def compute_information_criteria(folder):

    try:
        crocker_diffs = np.load(
            os.path.join(folder, "crocker_differences.npy"),
            allow_pickle=True
        )

    except Exception:
        print(f"Could not load {folder}")
        return None, None

    n = crocker_diffs.size
    RSS = max(float(np.sum(crocker_diffs**2)), np.finfo(float).tiny)
    sigma_sq = RSS / n

    Log_L = (-(n/2)*math.log(2*np.pi)-(n/2)*math.log(sigma_sq)-RSS/(2*sigma_sq))

    AIC = 2*(k+1) - 2*Log_L
    BIC = (k+1)*math.log(n) - 2*Log_L

    return [AIC, RSS], [BIC, RSS]


for Cidx, Lidx, Widx in pars_idc:

    folder = f'Widx_{str(Widx).zfill(2)}'

    sample_size = get_latest_run(folder)

    run_dir=f'{folder}/run_{sample_size}'

    AIC_results, BIC_results = compute_information_criteria(run_dir)

    np.save(f'{run_dir}/aic_results_{sample_size}.npy', AIC_results)
    np.savetxt(f'{run_dir}/aic_results_{sample_size}.txt', AIC_results, delimiter=",")

    np.save(f'{run_dir}/bic_results_{sample_size}.npy', BIC_results)
    np.savetxt(f'{run_dir}/bic_results_{sample_size}.txt', BIC_results, delimiter=",")

    print(f'Saved {run_dir} at sample {sample_size}')

print("Finished AIC + BIC")
