import warnings
warnings.simplefilter(action='ignore', category=FuturePar3arning)

import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
import glob
import imageio as io
import scipy
import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import axes3d


def compute_medians_and_densities(Par1_idx,Par2_idx,Par3_idx,sample_losses_angles_path,abc_posterior_densities_path,median_path):
    Par1s = np.linspace(,,) #update
    Par2s = np.linspace(,,) #update
    Par2s = np.linspace(,,) #update

    Par1_true = Par1s[Par1_idx]
    Par2_true = Par2s[Par2_idx]
    Par3_true = Par3s[Par3_idx]

    # Get ABC results:
    loss_path = sample_losses_angles_path
    sample_losses = np.load(loss_path,allow_pickle=True).item()

    sample_idx = []
    Par1_vals = []
    Par2_vals = []
    Par3_vals = []
    losses = []

    # max_Sample = len(sample_losses)
    samples_idcs = list(sample_losses)
    for iSample in samples_idcs:
        iSample = int(iSample)
        sample_idx.append(iSample)
        losses.append(sample_losses[str(iSample)]['loss'])
        Par1_vals.append(sample_losses[str(iSample)]['sampled_pars'][3])
        Par2_vals.append(sample_losses[str(iSample)]['sampled_pars'][4])
        Par3_vals.append(sample_losses[str(iSample)]['sampled_pars'][5])
    losses = np.array(losses)
    Par1_vals = np.array(Par1_vals)
    Par2_vals = np.array(Par2_vals)
    Par3_vals = np.array(Par3_vals)

    nan_idc = np.argwhere(np.isnan(losses))

    losses = np.delete(losses, nan_idc, axis=0)
    Par1_vals = np.delete(Par1_vals, nan_idc, axis=0)
    Par2_vals = np.delete(Par2_vals, nan_idc, axis=0)
    Par3_vals = np.delete(Par3_vals, nan_idc, axis=0)
    
    min_loss_idx = np.argmin(losses)
    Par1_min = Par1_vals[min_loss_idx]
    Par2_min = Par2_vals[min_loss_idx]
    Par3_min = Par3_vals[min_loss_idx]

    distance_threshold = np.percentile(losses, 1, axis=0) #this chooses the percent allow. could be 5 or 10% cuz 1 % is kinda harsh
    
    # Plot NM and ABC results
    belowTH_idc = np.where(losses < distance_threshold)[0]
    Par1_median = np.median(Par1_vals[belowTH_idc])
    Par2_median = np.median(Par2_vals[belowTH_idc])
    Par3_median = np.median(Par3_vals[belowTH_idc])

    #save medians
    np.save(median_path,[Par1_median,Par2_median,Par3_median])
    
    min_Par1 = 0.1 #update
    max_Par1 = 3.0 #update
    
    min_Par2 = 0.1 #update
    max_Par2 = 3.0 #update

    min_Par3 = 0.0 #update
    max_Par3 = 0.1 #update

    Par1s_plot = np.linspace(min_Par1,max_Par1,int(np.ceil((max_Par1/3.0)*11))) #update
    Par2s_plot = np.linspace(min_Par2,max_Par2,int(np.ceil((max_Par2/3.0)*11))) #update
    Par3s_plot = np.linspace(min_Par3,max_Par3,int(np.ceil((max_Par3/0.1)*11))) #update

    sample_count_map = np.zeros((len(Par1s_plot),len(Par2s_plot),len(Par3s_plot)))
    for belowTH_idx in belowTH_idc:
        Par1idx_belowTH = np.argmin(np.abs(Par1s_plot-Par1_vals[belowTH_idx]))
        Par2idx_belowTH = np.argmin(np.abs(Par2s_plot-Par2_vals[belowTH_idx]))
        Par3idx_belowTH = np.argmin(np.abs(Par3s_plot-Par3_vals[belowTH_idx]))
        sample_count_map[Par1idx_belowTH,Par2idx_belowTH,Par3idx_belowTH] += 1

    Par1_mesh_plot, Par2_mesh_plot = np.meshgrid(Par1s_plot, Par2s_plot, indexing='ij')

    for each_Par3 in range(len(Par3s)):
        fig, ax = plt.subplots(1,1,figsize=(6,6),dpi=400)
        plt.contourf(Par1_mesh_plot, Par2_mesh_plot, sample_count_map[:,:,each_Par3])
        if Par3s[each_Par3]-0.005 <= Par3_true < Par3s[each_Par3]+0.005:
            plt.scatter(Par1_true,Par2_true,
                        c="Par3",
                        linewidths = 0.5,
                        marker = '*',
                        edgecolor = "k",
                        s = 500,
                        label='True')
        if Par3s[each_Par3]-0.005 <= Par3_median < Par3s[each_Par3]+0.005:
            plt.scatter(Par1_median,Par2_median,
                        c="k",
                        linewidths = 0.5,
                        marker = 'o',
                        edgecolor = "k",
                        s = 180,
                        label='ABC-Median')
        ax.set_aspect('equal', adjustable='box')
        title_str = 'True: Par1 = ' + str(round(Par1_true,3)) + ', Par2 = ' + str(round(Par2_true,3)) + ', Par3 = ' + str(round(Par3_true,3)) + ' \n Median: Par1 = ' + str(round(Par1_median,3)) + ', Par2 = ' + str(round(Par2_median,3)) + ', Par3 = ' + str(round(Par3_median,3)) + '\n Par3 Plot Val = ' + str(Par3s[each_Par3])
        plt.title(title_str, fontsize=18) 
        plt.xlabel("Par1", fontsize=20)
        plt.ylabel("Par2", fontsize=20)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_xticks([0.1, 1.0, 2.0, 3.0]) #update
        ax.set_yticks([0.1, 1.0, 2.0, 3.0]) #update

        fig.tight_layout()
        fig.savefig(abc_posterior_densities_path+'/posterior_density_slice_at_Par3_'+str(each_Par3).zfill(2)+'.png',bbox_inches='tight')
 
        plt.close()   

    return([Par1_median,Par2_median,Par3_median])



