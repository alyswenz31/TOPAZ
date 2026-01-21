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
#from Scripts.crocker import *

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
    
    pars_idx, true_FRAME_LIST, pred_FRAME_LIST, betti_numbers, chosen_NUM_SAMPLE, max_NUM_SAMPLE, chosen_NUM_SAMPLE_AABC = args
    Cidx, Lidx, Widx = pars_idx
    if 'angle' in DATA_COLS:
        true_path = './Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/crocker_angles.npy'
        save_path = './Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/sample_losses_angles.npy'
        print('angle')
    if 'vx' in DATA_COLS:
        true_path = './Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/crocker_velocities.npy'
        save_path = './Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/sample_losses_velocities.npy'
        print('vx')
    true_crocker = np.load(true_path, allow_pickle=True)#.item()

    losses = {}
#    samples = []
#    losses  = []
    
    for iSample in range(chosen_NUM_SAMPLE):
#        print(iSample)
        pars_path = './Simulated_Grid/ODE_Align/sample_'+str(max_NUM_SAMPLE)+'/run_'+str(iSample+1)+'/pars.npy'
        if 'angle' in DATA_COLS:
            pred_path = './Simulated_Grid/ODE_Align/sample_'+str(max_NUM_SAMPLE)+'/run_'+str(iSample+1)+'/crocker_angles.npy'
        if 'vx' in DATA_COLS:
            pred_path = './Simulated_Grid/ODE_Align/sample_'+str(max_NUM_SAMPLE)+'/run_'+str(iSample+1)+'/crocker_velocities.npy'
            
        if os.path.isfile(pred_path):
            par_values = np.load(pars_path, allow_pickle=True)
            pred_crocker = np.load(pred_path, allow_pickle=True)
            
            loss = compute_crocker_error(true_crocker,pred_crocker)
            
            losses[str(iSample+1)] = {}
            losses[str(iSample+1)]['sampled_pars'] = par_values
            losses[str(iSample+1)]['loss'] = loss


    ## Update to include all AABC samples you want to use for ABC calculations (Ex. 50000, 50001, etc)
    sample_aabc_folders = [50000]   # list of ## values
    max_runs = None                       # set to integer to limit runs, e.g. 100

    # NEW: global starting index
    next_index = chosen_NUM_SAMPLE + 1  

    for chosen_NUM_SAMPLE_AABC in sample_aabc_folders:

        folder_path = f'./Simulated_Grid/ODE_Align/sample_aabc_{chosen_NUM_SAMPLE_AABC}'
        
        # find all runs dynamically
        run_dirs = sorted([
            d for d in os.listdir(folder_path)
            if d.startswith("run_")
        ])

        # determine how many runs to process
        if max_runs is None:
            num_runs = len(run_dirs)
        else:
            num_runs = min(len(run_dirs), max_runs)

        # print(f"Processing sample_aabc_{chosen_NUM_SAMPLE_AABC} with {num_runs} runs")

        for iSample in range(num_runs):

            run_name = f"run_{iSample+1}"
            pars_path = f'{folder_path}/{run_name}/theta_star.npy'
            pred_path = f'{folder_path}/{run_name}/crocker_angles.npy'

            if os.path.isfile(pred_path):

                par_values = np.load(pars_path, allow_pickle=True)
                pred_crocker = np.load(pred_path, allow_pickle=True)

                # Expand pars from length 3 → length 6
                par_values_extended = np.zeros(6)
                par_values_extended[3:6] = par_values
                par_values = par_values_extended

                loss = compute_crocker_error(true_crocker, pred_crocker)

                # ------ NEW INDEXING SYSTEM ------
                losses[str(next_index)] = {
                    'sampled_pars': par_values,
                    'loss': loss
                }
                next_index += 1
                # --------------------------------


        
    np.save(save_path,losses)
    
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
if 'angle' in DATA_COLS:
    PROX_VEC = 10**(np.linspace(-2,2,200)) #for position/entire crocker
if 'vx' in DATA_COLS:
    PROX_VEC = 10**(np.linspace(-2,2,200)) #for position/entire crocker

# Make sure to set these values appropriately according to how many ABC samples you have
max_NUM_SAMPLE = 2080 # Update to match chosen grid size from ABC_S01_samples.py
chosen_NUM_SAMPLE = 2080 # Update to match chosen grid size from ABC_S01_samples.py
chosen_NUM_SAMPLE_AABC = 10000 #Can be ignored

# This code is designed for the C=0.7, L=2.5 grid
# You can modify the pars_idc list to include other parameter combinations as needed
pars_idc = [(6,24,0),(6,24,5)]

list_tuples = []
for idx in range(len(pars_idc)):
    list_tuples.append((pars_idc[idx], true_FRAME_LIST, pred_FRAME_LIST, betti_numbers, chosen_NUM_SAMPLE, max_NUM_SAMPLE, chosen_NUM_SAMPLE_AABC))

# run_compute_distance(list_tuples[0])
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
   results = executor.map(run_compute_distance, list_tuples)
