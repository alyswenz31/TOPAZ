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
    
    DATA_COLS, true_FRAME_LIST, pred_FRAME_LIST, PROX_VEC = args
    DF_PATH = './single_ABC_sample/df.pkl'
    unscale_num = 25
#    if os.path.isfile(DF_PATH):
    #read in the dataframe and filter positions
    filt_df = pd.read_pickle(DF_PATH)

    filt_df, _ = filtering_df(filt_df, pred_FRAME_LIST, track_len=10, max_frame=500, min_speed=0)
    filt_df.x = filt_df.x/unscale_num
    filt_df.y = filt_df.y/unscale_num
    filt_df.vx = filt_df.vx/unscale_num
    filt_df.vy = filt_df.vy/unscale_num
#    print(max(filt_df.x))
    crocker = compute_crocker_custom(filt_df,true_FRAME_LIST,PROX_VEC,
                              data_cols=DATA_COLS,betti=[0,1])

    SAVE_PATH = './single_ABC_sample/crocker_angles.npy'
    np.save(SAVE_PATH,crocker)

def run_ABC_crocker_once():
    #Which DataFrame columns to use as dimensions
    DATA_COLS = ('x','y','angle')
    
    #List of frame values to use, must be aligned for direct comparison
    true_FRAME_LIST = range(20,120,1)
    pred_FRAME_LIST = range(10,120,1) #starts at 10 because of angle computation
    #compute the data for the crocker plot
    PROX_VEC = 10**(np.linspace(-2,2,200)) #for position/entire crocker
    
    list_tuples = []
    list_tuples = (DATA_COLS, true_FRAME_LIST, pred_FRAME_LIST, PROX_VEC)
    
    run_save_crocker(list_tuples)
 