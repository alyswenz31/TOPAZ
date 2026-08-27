import math
import numpy as np


def _calc_information_criteria(C_idx, L_idx, W_idx, suffix):
    # num parameters
    k = 3 #C, L, W
    
    # likelihood function 
    # error derived from difference in true and simulated (median value) crocker plots 
    folder = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)
    error_path = folder + '/crocker_differences' + suffix + '.npy'

     # L_hat = Residual sum of squares
    crocker_diffs = np.load(error_path,allow_pickle=True)

    # do RSS
    n = crocker_diffs.size
    RSS = max(float(np.sum(crocker_diffs**2)), np.finfo(float).tiny)
    sig_sq = RSS/n
    Log_L = -(n/2)*math.log(2*math.pi)-(n/2)*math.log(sig_sq)-(RSS/(2*sig_sq))
    
    # Include the residual variance as an additional fitted parameter.
    num_fitted_parameters = k + 1
    AIC = 2 * num_fitted_parameters - 2 * Log_L
    BIC = num_fitted_parameters * math.log(n) - 2 * Log_L

    AIC_results = [AIC, RSS]
    BIC_results = [BIC, RSS]

    np.save(folder + '/aic_results' + suffix + '.npy', AIC_results)
    np.save(folder + '/bic_results' + suffix + '.npy', BIC_results)

    return BIC_results, AIC_results


def calc_bic_and_aic(C_idx, L_idx, W_idx):
    return _calc_information_criteria(C_idx, L_idx, W_idx, '')


def calc_bic_and_aic_aabc(C_idx, L_idx, W_idx):
    return _calc_information_criteria(C_idx, L_idx, W_idx, '_aabc')
