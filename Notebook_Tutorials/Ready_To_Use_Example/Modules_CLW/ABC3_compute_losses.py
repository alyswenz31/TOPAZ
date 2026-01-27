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


def compute_crocker_error(true_metric, pred_metric):

    if len(true_metric.shape) > 2:
    
        max_B0 = np.max(true_metric[:,:,0])
        true_B0 = true_metric[:,:,0]/max_B0
        pred_B0 = pred_metric[:,:,0]/max_B0
        max_B1 = np.max(true_metric[:,:,1])
        true_B1 = true_metric[:,:,1]/max_B1
        pred_B1 = pred_metric[:,:,1]/max_B1
        loss = np.sum(np.abs(true_B0-pred_B0)) + np.sum(np.abs(true_B1-pred_B1))
    else:
        loss = np.sum((np.log10(true_metric)-np.log10(pred_metric))**2/np.max(np.log10(true_metric))**2)

    return loss


def run_compute_distance(args):
    
    pars_idx, true_FRAME_LIST, pred_FRAME_LIST, betti_numbers, chosen_NUM_SAMPLE, max_NUM_SAMPLE, sample_path = args
    C_idx, L_idx, W_idx = pars_idx
    true_path = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/crocker_angles.npy'
    save_path = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/sample_losses_angles.npy'
    true_crocker = np.load(true_path, allow_pickle=True)#.item()

    losses = {}

    for iSample in range(chosen_NUM_SAMPLE):
#        print(iSample)
        pars_path = sample_path+'/run_'+str(iSample+1)+'/pars.npy'
        pred_path = sample_path+'/run_'+str(iSample+1)+'/crocker_angles.npy'
            
        if os.path.isfile(pred_path):
            par_values = np.load(pars_path, allow_pickle=True)
            pred_crocker = np.load(pred_path, allow_pickle=True)
            
            loss = compute_crocker_error(true_crocker,pred_crocker)
            
            losses[str(iSample+1)] = {}
            losses[str(iSample+1)]['sampled_pars'] = par_values
            losses[str(iSample+1)]['loss'] = loss
        
    np.save(save_path,losses)

def compute_losses(C,L,W,sample_path,num_samples):

    #Cidx = 15
    #Lidx = 2
    betti_numbers = [0, 1]
    #VANILLA CROCKER
    #Which DataFrame columns to use as dimensions
    DATA_COLS = ('x','y','angle')
    
    #List of frame values to use, must be aligned for direct comparison
    true_FRAME_LIST = range(20,120,1)
    pred_FRAME_LIST = range(10,120,1) #starts at 10 because of angle computation
    #compute the data for the crocker plot
    PROX_VEC = 10**(np.linspace(-2,2,200)) #for position/entire crocker
    
    max_NUM_SAMPLE = num_samples
    chosen_NUM_SAMPLE = num_samples
    
    list_tuples = []
    list_tuples = ([C,L,W], true_FRAME_LIST, pred_FRAME_LIST, betti_numbers, chosen_NUM_SAMPLE, max_NUM_SAMPLE,sample_path)
    
    run_compute_distance(list_tuples)
    
