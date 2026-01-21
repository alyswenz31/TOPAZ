import math
import numpy as np


def BIC_fun(Cidx, Lidx, Widx, CLW_path):
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
    error_path = CLW_path + '/crocker_differences.npy'

     # L_hat = Residual sum of squares
    crocker_diffs = np.load(error_path,allow_pickle=True)

    # do RSS
    RSS = np.sum(crocker_diffs**2)
    sig_sq = 1
    Log_L = -(n/2)*math.log(2*math.pi)-(n/2)*math.log(sig_sq)-(RSS/(2*sig_sq))
    
    # put it all together into BIC 
    BIC = k * math.log(n) - 2 * Log_L 

    return [BIC, RSS]

# pars_idc = [(1,2,0), (1,2,4), (6,1,0), (6,1,4), (3,9,0), (3,9,4)]
pars_idc = [(17,3,0),(6,24,0),(19,0,0),(8,5,0),(4,4,0),(1,14,0),(14,6,0),(24,24,0),(19,14,0),(17,3,5),(6,24,5),(19,0,5),(8,5,5),(4,4,5),(1,14,5),(14,6,5),(24,24,5),(19,14,5)]

for pars_idx in pars_idc:
    Cidx, Lidx, Widx = pars_idx
    # BIC_results = []
    CLW_path = './Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1'
    BIC_results = BIC_fun(Cidx, Lidx, Widx, CLW_path)
    # BIC_results.append(BIC)
    np.save(CLW_path + '/bic_results_final_OGw00.npy',BIC_results)
    np.savetxt(CLW_path+'/bic_results_final_OGw00.txt', BIC_results, delimiter=",") 
