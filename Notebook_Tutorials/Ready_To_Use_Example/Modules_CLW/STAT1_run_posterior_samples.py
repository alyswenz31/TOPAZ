import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from itertools import permutations
from math import factorial
import scipy.optimize as opt
from functools import partial
import time
import matplotlib.pyplot as plt
import sys
import pandas as pd
from scipy.integrate import ode
import matplotlib.pyplot as plt
import os
import glob
import imageio as io
from itertools import repeat
from ripser import ripser
import scipy
import matplotlib as mpl
from functools import partial
import concurrent.futures
from Scripts.DorsognaNondim_Align import *


def run_simulation(pars, ic_vec, time_vec, num_sample, iSample):
    SIGMA, ALPHA, BETA, C, L, W = pars
    T0 = np.min(time_vec)
    TF = np.max(time_vec)
    DT = time_vec[1] - time_vec[0]
    
    FIGURE_PATH = './posterior_samples/'

    if not os.path.isdir(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)

    if not os.path.isdir(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)
    
    save_dir = os.path.join(FIGURE_PATH,'run_{0}'.format(iSample+1))
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    pickle_path = os.path.join(save_dir,'df.pkl')
    if not os.path.isfile(pickle_path):
        
        pars_path = os.path.join(save_dir,'pars.npy')
        np.save(pars_path,pars)

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
        model.position_gif(save_dir,time_vec)

def simulation_wrapper(args):
    
    SIGMA, ALPHA, BETA, C_val, L_val, W_val, ic_vec, time_vec, num_sample, iSample = args
    # print(iSample)
    pars = [SIGMA, ALPHA, BETA, C_val, L_val, W_val]

    run_simulation(pars, ic_vec, time_vec, num_sample, iSample)

def run_posterior_samples(C,L,W,T0,TF,DT,in_num_agents,num_post_samples):
    ###ARGS
    #Make time vector
    time_vec = np.arange(T0,TF+DT,DT)
    #Initial conditions
    rng = np.random.default_rng()

    #Number of datasets to make
    NUM_SAMPLE = num_post_samples

    num_agents = in_num_agents
    
    ic_vec = np.load('ic_vec.npy',allow_pickle=True)
    
    #Stochastic diffusivity parameter
    SIGMA = 0 #0.05
    #alpha
    ALPHA = 1.0
    BETA = 0.5
    
    # Suppose accept_prob is a 3D numpy array with values between 0 and 1
    # e.g. shape (30, 30, 10) corresponding to the grid of C, L, W values
    posteriors = np.load('./Chosen_C_'+str(C).zfill(2)+'_L_'+str(L).zfill(2)+'_W_'+str(W).zfill(2)+'/posteriors.npy', allow_pickle=True)
    # shrinks posterior to get rid of some empty space to make it faster
    # customize to each posterior
    accept_prob = posteriors[6:,:8,3:] 
    
    # Define the ranges and corresponding grid edges
    C_range = (0.7, 3.0)
    L_range = (0.1, 0.8)
    W_range = (0.03, 0.1)
    
    # Precompute grid edges for mapping values to indices
    C_edges = np.linspace(*C_range, accept_prob.shape[0])
    L_edges = np.linspace(*L_range, accept_prob.shape[1])
    W_edges = np.linspace(*W_range, accept_prob.shape[2])
    
    list_tuples = []
    while len(list_tuples) < NUM_SAMPLE:
        # Draw random C, L, W
        C = np.random.uniform(*C_range)
        L = np.random.uniform(*L_range)
        W = np.random.uniform(*W_range)
    
        # Find corresponding indices in the 3D array
        C_idx = np.searchsorted(C_edges, C) - 1
        L_idx = np.searchsorted(L_edges, L) - 1
        W_idx = np.searchsorted(W_edges, W) - 1
    
        # Clip to valid range
        C_idx = np.clip(C_idx, 0, accept_prob.shape[0]-1)
        L_idx = np.clip(L_idx, 0, accept_prob.shape[1]-1)
        W_idx = np.clip(W_idx, 0, accept_prob.shape[2]-1)
    
        # print([C,L,W,C_idx,L_idx,W_idx])
    
        # Acceptance probability from array
        p_accept = accept_prob[C_idx, L_idx, W_idx]
        # print('accept_prob: ')
        # print(p_accept)
    
        # Accept/reject step
        if np.random.rand() <= p_accept:
            # print('accepted')
            iSample = len(list_tuples)
            list_tuples.append((SIGMA, ALPHA, BETA, C, L, W, ic_vec, time_vec, NUM_SAMPLE, iSample))
            # print('length')
            # print(len(list_tuples))
    
    for i in range(len(list_tuples)):   
        simulation_wrapper(list_tuples[i])
