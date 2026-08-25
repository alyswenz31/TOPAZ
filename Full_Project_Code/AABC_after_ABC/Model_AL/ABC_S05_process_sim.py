import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
from ripser import ripser
import scipy
import re

import concurrent.futures
from scipy.integrate import ode
import glob
import imageio as io
from itertools import repeat

from Scripts.DorsognaNondim_Align import *
from Scripts.crocker import *
from Scripts.aabc_utils import latest_corrected_run

def get_latest_run(folder):
    run_number, _ = latest_corrected_run(folder)
    return run_number

def run_simulation(pars, ic_vec, time_vec, iRUN, run_dir, opt_alg=None):
    SIGMA, ALPHA, BETA, C_idx, C, L_idx, L, W_idx, W = pars
    T0 = np.min(time_vec)
    TF = np.max(time_vec)
    DT = time_vec[1] - time_vec[0]

    par_dir = 'Widx_'+str(W_idx).zfill(2)
    #Where to save the runs
    if SIGMA == 0:
        FIGURE_PATH = './'+par_dir+'/'
    elif SIGMA > 0:
        FIGURE_PATH = './'+par_dir+'/'
    else:
        raise ValueError("{0} is an invalid value for SIGMA".format(SIGMA))

    if not os.path.isdir(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)
    
    if opt_alg is not None:
        if opt_alg == "NM":
            pickle_path = os.path.join(run_dir,'df_NM.pkl')
        elif opt_alg == "ABC":
            pickle_path = os.path.join(run_dir,'df_ABC.pkl')

    #Simulate using appropriate integrator
    MODEL_CLASS = DorsognaNondim
    model = MODEL_CLASS(sigma=SIGMA,alpha=ALPHA,beta=BETA,
                       c=C,l=L,w=W)
    if SIGMA == 0:
        model.ode_rk4(ic_vec,T0,TF,DT)
    elif SIGMA > 0:
        model.sde_maruyama(ic_vec,T0,TF,return_time=DT)
    else:
        raise ValueError("{0} is an invalid value for SIGMA".format(SIGMA))

    #Save results as dataframe
    results = model.results_to_df(time_vec)
    results.to_pickle(pickle_path)
    
    #Plot gif of simulated positions
    model.position_gif(run_dir,time_vec)

Cs = np.linspace(0.1,3.0,30)
Ls = np.linspace(0.1,3.0,30)
Ws = np.linspace(0.0,0.1,11)


pars_idc = [(6, 24, 0), (6, 24, 5)]

#What time to use as initial
T0 = 1
#What time to end the simulation
TF = 21
#How often to make a new frame of data
DT = 1/6
#Make time vector
time_vec = np.arange(T0,TF+DT,DT)
#Initial conditions
rng = np.random.default_rng()

num_agents = 300
ic_vec = np.load('ic_vec.npy',allow_pickle=True)

#Stochastic diffusivity parameter
SIGMA = 0 #0.05
#alpha
ALPHA = 1.0
BETA = 0.5
iRUN = 0

for pars_idx in pars_idc:
    Cidx, Lidx, Widx = pars_idx
    C_true = Cs[Cidx]
    L_true = Ls[Lidx]
    W_true = Ws[Widx]
    
    # Get ABC results:
    folder = f'./Widx_{str(Widx).zfill(2)}'

    sample_size = get_latest_run(folder)

    run_dir = f'{folder}/run_{sample_size}'

    print(f"Using run {sample_size}")

    # Step 4 owns posterior selection and uses one fixed base-library tolerance.
    # Reuse its saved median instead of recomputing a moving top-1% posterior.
    C_median, L_median, W_median = np.load(
        f"{run_dir}/medians.npy", allow_pickle=True
    )
    
    pars = [SIGMA, ALPHA, BETA, Cidx, C_median, Lidx, L_median, Widx, W_median]
    
    # Run Nelder-Mead result simulation
    run_simulation(pars, ic_vec, time_vec, iRUN, run_dir, opt_alg="ABC")
