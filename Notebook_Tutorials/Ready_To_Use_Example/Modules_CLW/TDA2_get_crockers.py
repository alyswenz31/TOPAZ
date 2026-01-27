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
from Scripts.filtering_df import *
from Scripts.crocker import *

def run_save_crocker(args):
    
    C_idx, L_idx, W_idx, DATA_COLS, true_FRAME_LIST, pred_FRAME_LIST, PROX_VEC = args
    DF_PATH = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/df.pkl'
    
    #read in the dataframe and filter positions
    filt_df = pd.read_pickle(DF_PATH)

    filt_df, _ = filtering_df(filt_df, pred_FRAME_LIST, track_len=10, max_frame=128, min_speed=0)
    filt_df.x = filt_df.x/25
    filt_df.y = filt_df.y/25 #0.4, ~
    filt_df.vx = filt_df.vx/25
    filt_df.vy = filt_df.vy/25
    crocker = compute_crocker_custom(filt_df,true_FRAME_LIST,PROX_VEC,
                              data_cols=DATA_COLS,betti=[0,1])

    SAVE_PATH = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/crocker_angles.npy'
    np.save(SAVE_PATH,crocker)

    PLOT_PATH = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/crocker_plot.png'

    Cs = np.linspace(0.1,3.0,30)
    Ls = np.linspace(0.1,3.0,30)
    Ws = np.linspace(0.0,0.1,11)
    Cval = Cs[C_idx-1]
    Lval = Ls[L_idx-1]
    Wval = Ws[W_idx]

    plot_crocker_highres_split(crocker,PROX_VEC,[50,150,250,350,450],crocker,PROX_VEC,[50,150,250,350,450],[Cval,Lval,Wval],save_path=PLOT_PATH)


def get_crockers(C_idx,L_idx,W_idx):
    #VANILLA CROCKER
    #Which DataFrame columns to use as dimensions
    DATA_COLS = ('x','y','angle')
    # Frame 10: get initial conditions (x,y,vx,vy)
    # Frame 20:
    #List of frame values to use, must be aligned for direct comparison
    true_FRAME_LIST = range(20,120,1)
    pred_FRAME_LIST = range(10,120,1) #starts at 10 because of angle computation
    #compute the data for the crocker plot
    PROX_VEC = 10**(np.linspace(-2,2,200)) #for position/entire crocker
    
    list_tuples = []
    list_tuples=(C_idx, L_idx, W_idx, DATA_COLS, true_FRAME_LIST, pred_FRAME_LIST, PROX_VEC)
    
    
    run_save_crocker(list_tuples)
    
