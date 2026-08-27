import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import os
from ripser import ripser
import scipy

import concurrent.futures
from scipy.integrate import ode
import glob
import imageio as io
from itertools import repeat

from Scripts.DorsognaNondim_Align import *
from Scripts.crocker import *


def run_simulation(pars, ic_vec, time_vec, opt_alg=None):
    SIGMA, ALPHA, BETA, C_idx, C, L_idx, L, W_idx, W = pars
    T0 = np.min(time_vec)
    TF = np.max(time_vec)
    DT = time_vec[1] - time_vec[0]

    par_dir = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)
    #Where to save the runs
    FIGURE_PATH = './'+par_dir+'/'
    
    if not os.path.isdir(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)

    if opt_alg is not None:
        pickle_path = os.path.join(FIGURE_PATH,'df_AABC.pkl')

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
    model.position_gif(par_dir,time_vec)
    os.rename(FIGURE_PATH+"/position.gif", FIGURE_PATH+"/AABC_med_simulation.gif")

def run_ABC_sim_aabc(C_idx,L_idx,W_idx,T0,TF,DT,in_num_agents):
    Cs = np.linspace(0.1,3.0,30)
    Ls = np.linspace(0.1,3.0,30)
    Ws = np.linspace(0.0,0.1,11)

    pars_idc = [(C_idx,L_idx,W_idx)]

    #Make time vector
    time_vec = np.arange(T0,TF+DT,DT)
    #Initial conditions
    rng = np.random.default_rng()

    num_agents = in_num_agents

    ic_vec = np.load('ic_vec.npy',allow_pickle=True)

    #Stochastic diffusivity parameter
    SIGMA = 0 #0.05
    #alpha
    ALPHA = 1.0
    BETA = 0.5

    C_true = Cs[C_idx-1]
    L_true = Ls[L_idx-1]
    W_true = Ws[W_idx]

    SAVE_PATH = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/'
    C_median, L_median, W_median = np.load(SAVE_PATH+'medians_aabc.npy')
    
    pars = [SIGMA, ALPHA, BETA, C_idx, C_median, L_idx, L_median, W_idx, W_median]
    
    # Run Nelder-Mead result simulation
    run_simulation(pars, ic_vec, time_vec, opt_alg="ABC")
    


    
