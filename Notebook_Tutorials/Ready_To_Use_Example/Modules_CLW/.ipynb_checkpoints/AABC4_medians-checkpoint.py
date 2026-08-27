import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
from Scripts.aabc_utils import unpack_loss_records


def compute_medians_and_densities_aabc(C_idx,L_idx,W_idx,parameter_bounds):
    Cs = np.linspace(0.1,3.0,30)
    Ls = np.linspace(0.1,3.0,30)
    Ws = np.linspace(0, 0.1, 11)

    C_true = Cs[C_idx-1]
    L_true = Ls[L_idx-1]
    W_true = Ws[W_idx]

    # Get ABC results:
    loss_path = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/sample_losses_angles_aabc.npy'
    sample_losses = np.load(loss_path,allow_pickle=True).item()

    parameters, losses = unpack_loss_records(sample_losses)
    C_vals, L_vals, W_vals = parameters.T
    
    min_loss_idx = np.argmin(losses)
    C_min = C_vals[min_loss_idx]
    L_min = L_vals[min_loss_idx]
    W_min = W_vals[min_loss_idx]

    number_to_accept = max(1, int(np.ceil(0.01 * len(losses))))
    belowTH_idc = np.argsort(losses, kind='stable')[:number_to_accept]
    C_median = np.median(C_vals[belowTH_idc])
    L_median = np.median(L_vals[belowTH_idc])
    W_median = np.median(W_vals[belowTH_idc])

    [C_lower_val, C_upper_val] = parameter_bounds[0]
    [L_lower_val, L_upper_val] = parameter_bounds[1]
    [W_lower_val, W_upper_val] = parameter_bounds[2]
    
    min_C = C_lower_val
    max_C = C_upper_val
    
    min_L = L_lower_val
    max_L = L_upper_val

    min_W = W_lower_val
    max_W = W_upper_val

    Cs_plot = np.linspace(min_C,max_C,11)
    Ls_plot = np.linspace(min_L,max_L,11)
    Ws_plot = np.linspace(min_W,max_W,int(np.ceil((max_W/0.1)*11)))

    sample_count_map = np.zeros((len(Cs_plot),len(Ls_plot),len(Ws_plot)))
    for belowTH_idx in belowTH_idc:
        Cidx_belowTH = np.argmin(np.abs(Cs_plot-C_vals[belowTH_idx]))
        Lidx_belowTH = np.argmin(np.abs(Ls_plot-L_vals[belowTH_idx]))
        Widx_belowTH = np.argmin(np.abs(Ws_plot-W_vals[belowTH_idx]))
        sample_count_map[Cidx_belowTH,Lidx_belowTH,Widx_belowTH] += 1

    C_mesh_plot, L_mesh_plot = np.meshgrid(Cs_plot, Ls_plot, indexing='ij')

    for each_w in range(len(Ws)):
        fig, ax = plt.subplots(1,1,figsize=(6,6),dpi=400)
        plt.contourf(C_mesh_plot, L_mesh_plot, sample_count_map[:,:,each_w])
        if Ws[each_w]-0.005 <= W_true < Ws[each_w]+0.005:
            plt.scatter(C_true,L_true,
                        c="w",
                        linewidths = 0.5,
                        marker = '*',
                        edgecolor = "k",
                        s = 500,
                        label='True')
        if Ws[each_w]-0.005 <= W_median < Ws[each_w]+0.005:
            plt.scatter(C_median,L_median,
                        c="k",
                        linewidths = 0.5,
                        marker = 'o',
                        edgecolor = "k",
                        s = 180,
                        label='AABC-Median')
        ax.set_aspect('equal', adjustable='box')
        title_str = 'True: C = ' + str(round(C_true,3)) + ', L = ' + str(round(L_true,3)) + ', W = ' + str(round(W_true,3)) + ' \n Median: C = ' + str(round(C_median,3)) + ', L = ' + str(round(L_median,3)) + ', W = ' + str(round(W_median,3)) + '\n W Plot Val = ' + str(Ws[each_w])
        plt.title(title_str, fontsize=18)#, 
        plt.xlabel("C", fontsize=20)
        plt.ylabel("L", fontsize=20)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_xticks([0.1, 1.0, 2.0, 3.0])
        ax.set_yticks([0.1, 1.0, 2.0, 3.0])
        
        save_dir = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/AABC_posterior_densities'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        fig.tight_layout()
        fig.savefig(save_dir+'/posterior_density_slice_at_w'+str(each_w).zfill(2)+'.png',bbox_inches='tight')
 
        plt.close()   

    medians = np.array([C_median,L_median,W_median])
    np.save(os.path.dirname(loss_path) + '/medians_aabc.npy', medians)
    return medians

