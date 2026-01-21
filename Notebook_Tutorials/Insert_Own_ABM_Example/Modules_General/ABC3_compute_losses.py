import warnings
warnings.simplefilter(action='ignore', category=FuturePar3arning) 

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
    
    NUM_SAMPLE, tda_crocker_angles_path, abc_crocker_angles_and_pars_path, sample_losses_angles_path = args
    true_path = tda_crocker_angles_path
    save_path = sample_losses_angles_path
    true_crocker = np.load(true_path, allow_pickle=True)#.item()

    losses = {}

    for iSample in range(chosen_NUM_SAMPLE):
#        print(iSample)
        pars_path = abc_crocker_angles_and_pars_path+'/run_'+str(iSample+1)+'/pars.npy'
        pred_path = abc_crocker_angles_and_pars_path+'/run_'+str(iSample+1)+'/crocker_angles.npy'
            
        if os.path.isfile(pred_path):
            par_values = np.load(pars_path, allow_pickle=True)
            pred_crocker = np.load(pred_path, allow_pickle=True)
            
            loss = compute_crocker_error(true_crocker,pred_crocker)
            
            losses[str(iSample+1)] = {}
            losses[str(iSample+1)]['sampled_pars'] = par_values
            losses[str(iSample+1)]['loss'] = loss
        
    np.save(save_path,losses)

def compute_losses(num_samples, tda_crocker_angles_path, abc_crocker_angles_and_pars_path, sample_losses_angles_path):

    #VANILLA CROCKER
    #Which DataFrame columns to use as dimensions
    DATA_COLS = ('x','y','angle')
    
    #compute the data for the crocker plot
    PROX_VEC = 10**(np.linspace(-2,2,200)) #for position/entire crocker

    NUM_SAMPLE = num_samples
    
    list_tuples = []
    list_tuples = (NUM_SAMPLE, tda_crocker_angles_path, abc_crocker_angles_and_pars_path, sample_losses_angles_path)
    
    run_compute_distance(list_tuples)
    
