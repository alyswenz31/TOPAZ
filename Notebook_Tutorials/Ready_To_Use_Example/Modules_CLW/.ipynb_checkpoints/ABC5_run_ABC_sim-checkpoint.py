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
        pickle_path = os.path.join(FIGURE_PATH,'df_ABC.pkl')

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
    os.rename(FIGURE_PATH+"/position.gif", FIGURE_PATH+"/ABC_med_simulation.gif")

def run_ABC_sim(C_idx,L_idx,W_idx,T0,TF,DT,in_num_agents):
    Cs = np.linspace(0.1,3.0,30)
    Ls = np.linspace(0.1,3.0,30)
    Ls = np.linspace(0.0,0.1,11)

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
    W_true = Ls[W_idx]

    # Get ABC results:
    loss_path = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/sample_losses_angles.npy'
    sample_losses = np.load(loss_path,allow_pickle=True).item()
    
    sample_idx = []
    C_vals = []
    L_vals = []
    W_vals = []
    losses = []

     # max_Sample = len(sample_losses)
    samples_idcs = list(sample_losses)
    for iSample in samples_idcs:
        iSample = int(iSample)
        sample_idx.append(iSample)
        losses.append(sample_losses[str(iSample)]['loss'])
        C_vals.append(sample_losses[str(iSample)]['sampled_pars'][3])
        L_vals.append(sample_losses[str(iSample)]['sampled_pars'][4])
        W_vals.append(sample_losses[str(iSample)]['sampled_pars'][5])
    losses = np.array(losses)
    C_vals = np.array(C_vals)
    L_vals = np.array(L_vals)
    W_vals = np.array(W_vals)

    nan_idc = np.argwhere(np.isnan(losses))

    losses = np.delete(losses, nan_idc, axis=0)
    C_vals = np.delete(C_vals, nan_idc, axis=0)
    L_vals = np.delete(L_vals, nan_idc, axis=0)
    W_vals = np.delete(W_vals, nan_idc, axis=0)
    
    min_loss_idx = np.argmin(losses)
    C_min = C_vals[min_loss_idx]
    L_min = L_vals[min_loss_idx]
    W_min = W_vals[min_loss_idx]

    distance_threshold = np.percentile(losses, 1, axis=0)
    
    # Plot NM and ABC results
    belowTH_idc = np.where(losses < distance_threshold)[0]
    C_median = np.median(C_vals[belowTH_idc])
    L_median = np.median(L_vals[belowTH_idc])
    W_median = np.median(W_vals[belowTH_idc])

    SAVE_PATH = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/'
    np.save(SAVE_PATH+'medians.npy',[C_median,L_median,W_median])
    
    pars = [SIGMA, ALPHA, BETA, C_idx, C_median, L_idx, L_median, W_idx, W_median]
    
    # Run Nelder-Mead result simulation
    run_simulation(pars, ic_vec, time_vec, opt_alg="ABC")
    


    
