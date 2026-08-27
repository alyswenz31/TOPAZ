import math
import numpy as np


def _calc_information_criteria(num_parameters, crocker_diff_path,
                               bic_path, aic_path):
    crocker_diffs = np.load(crocker_diff_path, allow_pickle=True)
    n = crocker_diffs.size
    rss = max(float(np.sum(crocker_diffs**2)), np.finfo(float).tiny)
    sigma_sq = rss / n
    log_likelihood = (-(n/2) * math.log(2*math.pi)-(n/2)*math.log(sigma_sq)-rss/(2*sigma_sq))
    
    num_fitted_parameters = num_parameters + 1
    aic = 2 * num_fitted_parameters - 2 * log_likelihood
    bic = num_fitted_parameters * math.log(n) - 2 * log_likelihood
    
    bic_results = [bic, rss]
    aic_results = [aic, rss]
    np.save(bic_path, bic_results)
    np.save(aic_path, aic_results)
    return bic_results, aic_results


def calc_bic_and_aic(num_parameters, crocker_diff_path, bic_path, aic_path):
    return _calc_information_criteria(
        num_parameters, crocker_diff_path, bic_path, aic_path
    )


def calc_bic_and_aic_aabc(num_parameters, crocker_diff_path,
                          bic_path, aic_path):
    return _calc_information_criteria(
        num_parameters, crocker_diff_path, bic_path, aic_path
    )

