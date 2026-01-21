import math
import numpy as np


def calc_bic(num_parameters, num_data_pts, crocker_diff_path, bic_path):
    # num parameters
    k = num_parameters
    
    # num samples 
    n = num_data_pts
    
    # likelihood function 
    # error derived from difference in true and simulated (median value) crocker plots 
    error_path = crocker_diff_path

     # L_hat = Residual sum of squares
    crocker_diffs = np.load(error_path,allow_pickle=True)

    # do RSS
    RSS = np.sum(crocker_diffs**2)
    sig_sq = 1
    Log_L = -(n/2)*math.log(2*math.pi)-(n/2)*math.log(sig_sq)-(RSS/(2*sig_sq))
    
    # put it all together into BIC 
    BIC = k * math.log(n) - 2 * Log_L 

    BIC_results = [BIC, RSS]

    np.save(bic_path,BIC_results)

    return BIC_results


