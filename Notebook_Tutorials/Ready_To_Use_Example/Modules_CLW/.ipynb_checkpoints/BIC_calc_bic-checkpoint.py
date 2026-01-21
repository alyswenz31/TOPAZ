import math
import numpy as np


def calc_bic(C_idx,L_idx,W_idx):
    # num parameters
    k = 3 #C, L, W
    
    # num samples 
    # either want 
        # total num time points (122) x num approximations (11x11x11=1331?) x 2 #122 is len fo time_vec 
        # total num time points (122) x num approximations (30x30x11=9900?) x 2
        # total num time points x 2
        # 2
    n = 100 * 200 * 2
    
    # likelihood function 
    # error derived from difference in true and simulated (median value) crocker plots 
    error_path = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/crocker_differences.npy'

     # L_hat = Residual sum of squares
    crocker_diffs = np.load(error_path,allow_pickle=True)

    # do RSS
    RSS = np.sum(crocker_diffs**2)
    sig_sq = 1
    Log_L = -(n/2)*math.log(2*math.pi)-(n/2)*math.log(sig_sq)-(RSS/(2*sig_sq))
    
    # put it all together into BIC 
    BIC = k * math.log(n) - 2 * Log_L 

    BIC_results = [BIC, RSS]

    np.save('Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/bic_results.npy',BIC_results)

    return BIC_results


