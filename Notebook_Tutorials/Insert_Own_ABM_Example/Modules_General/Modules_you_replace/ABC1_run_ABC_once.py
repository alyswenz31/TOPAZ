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
    
    FIGURE_PATH = './single_ABC_sample'

    if not os.path.isdir(FIGURE_PATH):
        os.makedirs(FIGURE_PATH)

    pickle_path = os.path.join(FIGURE_PATH,'df.pkl')
    if not os.path.isfile(pickle_path):

        pars_path = os.path.join(FIGURE_PATH,'pars.npy')
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
        model.position_gif(FIGURE_PATH,time_vec)
        os.rename(FIGURE_PATH+"/position.gif", FIGURE_PATH+"/random_ABC_simulation.gif")

def simulation_wrapper(args):
    
    SIGMA, ALPHA, BETA, C_val, L_val, W_val, ic_vec, time_vec, num_sample, iSample = args
    # print(iSample)
    pars = [SIGMA, ALPHA, BETA, C_val, L_val, W_val]

    run_simulation(pars, ic_vec, time_vec, num_sample, iSample)

def run_ABC_once(T0,TF,DT,in_num_agents):
    ###ARGS
    #Make time vector
    time_vec = np.arange(T0,TF+DT,DT)
    #Initial conditions
    rng = np.random.default_rng()

    #Number of datasets to make
    NUM_SAMPLE = 1

    num_agents = in_num_agents
    
    ic_vec = np.load('ic_vec.npy',allow_pickle=True)
    
    #Stochastic diffusivity parameter
    SIGMA = 0 #0.05
    #alpha
    ALPHA = 1.0
    BETA = 0.5
    
    list_tuples = []
    for iSample in range(NUM_SAMPLE):
        if iSample>= 0:
            C = np.random.uniform(low=0.1,high=3.0)
            L = np.random.uniform(low=0.1,high=3.0)
            W = np.random.uniform(low=0.0,high=0.1)
            list_tuples = (SIGMA, ALPHA, BETA, C, L, W, ic_vec, time_vec, NUM_SAMPLE, iSample)
    
    simulation_wrapper(list_tuples)
