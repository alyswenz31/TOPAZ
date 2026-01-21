import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
import numpy as np
import pandas as pd
import concurrent.futures
from scipy.integrate import ode
import matplotlib.pyplot as plt
import os
import glob
import imageio as io
from itertools import repeat
#from Scripts.DorsognaNondim_Align_w05 import *
from Scripts.DorsognaNondim_Align import *



def run_simulation(pars, ic_vec, time_vec):
    SIGMA, ALPHA, BETA, C_idx, C, L_idx, L, W_idx, W = pars
    T0 = np.min(time_vec)
    TF = np.max(time_vec)
    DT = time_vec[1] - time_vec[0]
    
    FIGURE_PATH = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)

    if not os.path.isdir(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)

    pickle_path = os.path.join(FIGURE_PATH,'df.pkl')
    if not os.path.isfile(pickle_path):

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
        model.position_gif(FIGURE_PATH,time_vec)
        os.rename(FIGURE_PATH+"/position.gif", FIGURE_PATH+"/TDA_simulation.gif")

def simulation_wrapper(args):

    sim_values = args
    SIGMA, ALPHA, BETA, C_idx, C_val, L_idx, L_val, W_idx, W_val, ic_vec, time_vec = sim_values
    pars = [SIGMA, ALPHA, BETA, C_idx, C_val, L_idx, L_val, W_idx, W_val]
    
    run_simulation(pars, ic_vec, time_vec)

def run_CLW_sim(C_idx,L_idx,W_idx,T0,TF,DT,in_num_agents): 
    ###ARGS
    #Make time vector
    time_vec = np.arange(T0,TF+DT,DT)
    #Initial conditions
    rng = np.random.default_rng()

    ic_vec = np.load('ic_vec.npy',allow_pickle=True)

    #Stochastic diffusivity parameter
    SIGMA = 0 
    ALPHA = 1.0
    BETA = 0.5

    num_agents = in_num_agents

    Cs = np.linspace(0.1,3.0,30)
    Ls = np.linspace(0.1,3.0,30)
    Ws = np.linspace(0, 0.1, 11)

    list_tuples = []
    C = Cs[C_idx-1]
    L = Cs[L_idx-1]
    W = Cs[W_idx]
    #list_tuples.append((SIGMA, ALPHA, BETA, C_idx, C, L_idx, L, W_idx, W, ic_vec, time_vec))
    list_tuples=(SIGMA, ALPHA, BETA, C_idx, C, L_idx, L, W_idx, W, ic_vec, time_vec)
    #print(list_tuples.size)
    
    # len_grid = len(list_tuples)
    # list_tuples2 = list(zip(tuple(list_tuples), tuple(repeat(r,len_grid))))

    simulation_wrapper(list_tuples)



