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

Cs = np.linspace(0.1,3.0,30)
Ls = np.linspace(0.1,3.0,30)
Ws = np.linspace(0, 0.1, 11)

# C_mesh, L_mesh, W_mesh = np.meshgrid(Cs, Ls, indexing='ij')
# pars_idc = [(6,2),(3,9),(7,1),(3,2),(2,2),(1,5),(5,3),(8,9),(7,5)]
# pars_idc = [(18,4),(7,25),(20,1),(9,6),(5,5),(2,15),(15,7),(25,25),(20,15)]
pars_idc = [(17,3,0),(6,24,0),(19,0,0),(8,5,0),(4,4,0),(1,14,0),(14,6,0),(24,24,0),(19,14,0),(17,3,5),(6,24,5),(19,0,5),(8,5,5),(4,4,5),(1,14,5),(14,6,5),(24,24,5),(19,14,5)]
# pars_idc = [(1,2,0), (1,2,4), (6,1,0), (6,1,4), (3,9,0), (3,9,4)]

for pars_idx in pars_idc:
    Cidx, Lidx, Widx = pars_idx
    C_true = Cs[Cidx]
    L_true = Ls[Lidx]
    W_true = Ws[Widx]
    
    # Get ABC results:
    loss_path = './Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/sample_losses_angles.npy'
    sample_losses = np.load(loss_path,allow_pickle=True).item()
    
    
    
    sample_idx = []
    C_vals = []
    L_vals = []
    W_vals = []
    losses = []
    
    # max_Sample = len(sample_losses)
    samples_idcs = list(sample_losses)
    for iSample in samples_idcs:
        iSample = int(iSample)
        sample_idx.append(iSample)
        losses.append(sample_losses[str(iSample)]['loss'])
        C_vals.append(sample_losses[str(iSample)]['sampled_pars'][3])
        L_vals.append(sample_losses[str(iSample)]['sampled_pars'][4])
        W_vals.append(sample_losses[str(iSample)]['sampled_pars'][5])
    losses = np.array(losses)
    C_vals = np.array(C_vals)
    L_vals = np.array(L_vals)
    W_vals = np.array(W_vals)

    nan_idc = np.argwhere(np.isnan(losses))

    losses = np.delete(losses, nan_idc, axis=0)
    C_vals = np.delete(C_vals, nan_idc, axis=0)
    L_vals = np.delete(L_vals, nan_idc, axis=0)
    W_vals = np.delete(W_vals, nan_idc, axis=0)
    
    min_loss_idx = np.argmin(losses)
    C_min = C_vals[min_loss_idx]
    L_min = L_vals[min_loss_idx]
    W_min = W_vals[min_loss_idx]

    distance_threshold = np.percentile(losses, 1, axis=0) #this chooses the percent allow. could be 5 or 10% cuz 1 % is kinda harsh
    
    # Plot NM and ABC results
    belowTH_idc = np.where(losses < distance_threshold)[0]
    C_median = np.median(C_vals[belowTH_idc])
    L_median = np.median(L_vals[belowTH_idc])
    W_median = np.median(W_vals[belowTH_idc])
    
    min_C = 0.1
    max_C = 3.0
    
    min_L = 0.1
    max_L = 3.0

    min_W = 0.0
    max_W = 0.1

    Cs_plot = np.linspace(min_C,max_C,int(np.ceil((max_C/3.0)*11)))
    Ls_plot = np.linspace(min_L,max_L,int(np.ceil((max_L/3.0)*11)))
    Ws_plot = np.linspace(min_W,max_W,int(np.ceil((max_W/0.1)*11)))

    sample_count_map = np.zeros((len(Cs_plot),len(Ls_plot),len(Ws_plot)))
    for belowTH_idx in belowTH_idc:
        Cidx_belowTH = np.argmin(np.abs(Cs_plot-C_vals[belowTH_idx]))
        Lidx_belowTH = np.argmin(np.abs(Ls_plot-L_vals[belowTH_idx]))
        Widx_belowTH = np.argmin(np.abs(Ws_plot-W_vals[belowTH_idx]))
        sample_count_map[Cidx_belowTH,Lidx_belowTH,Widx_belowTH] += 1

############### 2D PLOTS  - OLD (star and dot on every plot) #################
    C_mesh_plot, L_mesh_plot = np.meshgrid(Cs_plot, Ls_plot, indexing='ij')
    
    # sample_count_map = np.zeros((len(Cs_plot),len(Ls_plot)))
    # for belowTH_idx in belowTH_idc:
    #     Cidx_belowTH = np.argmin(np.abs(Cs_plot-C_vals[belowTH_idx]))
    #     Lidx_belowTH = np.argmin(np.abs(Ls_plot-L_vals[belowTH_idx]))
    #     sample_count_map[Cidx_belowTH,Lidx_belowTH] += 1

    for each_w in range(len(Ws)):
        fig, ax = plt.subplots(1,1,figsize=(6,6),dpi=400)
        plt.contourf(C_mesh_plot, L_mesh_plot, sample_count_map[:,:,each_w])
        plt.scatter(C_true,L_true,
                    c="w",
                    linewidths = 0.5,
                    marker = '*',
                    edgecolor = "k",
                    s = 500,
                    label='True')
        plt.scatter(C_median,L_median,
                    c="k",
                    linewidths = 0.5,
                    marker = 'o',
                    edgecolor = "k",
                    s = 180,
                    label='ABC-Median')
        ax.set_aspect('equal', adjustable='box')
    #     ax.set_title(f' Posterior Density \n True C = {C_true:.02}, True L = {L_true:.02} \n ABC-Threshold = {distance_threshold} ',fontsize=16)
    #     plt.title(f'ABC-Posterior Density\n', fontsize=25)
        title_str = 'True: C = ' + str(round(C_true,3)) + ', L = ' + str(round(L_true,3)) + ', W = ' + str(round(W_true,3)) + ' \n Median: C = ' + str(round(C_median,3)) + ', L = ' + str(round(L_median,3)) + ', W = ' + str(round(W_median,3)) + '\n W Plot Val = ' + str(Ws[each_w])
        plt.title(title_str, fontsize=18)#, horizontalalignment='center', x=0.535, y=0.85)
    #     plt.suptitle(f'(ABC-Threshold = {distance_threshold})\n', fontsize=15,  y=0.5)
        plt.xlabel("C", fontsize=20)
        plt.ylabel("L", fontsize=20)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        if (Cidx == 25 and Lidx == 15) or (Cidx == 25 and Lidx == 25):
            pass
        else:
            ax.set_xticks([0.1, 1.0, 2.0, 3.0])
            ax.set_yticks([0.1, 1.0, 2.0, 3.0])
        
        save_dir = './Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/ABC_angle_plot'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        fig.tight_layout()
        fig.savefig(save_dir+'/ABC_'+'Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'_angles_at_w'+str(each_w).zfill(2)+'_OGw05.pdf',bbox_inches='tight')
 
        plt.close()   
    

############### 2D PLOTS  - NEW (star and dot on one plot) #################
    C_mesh_plot, L_mesh_plot = np.meshgrid(Cs_plot, Ls_plot, indexing='ij')
    
    # sample_count_map = np.zeros((len(Cs_plot),len(Ls_plot)))
    # for belowTH_idx in belowTH_idc:
    #     Cidx_belowTH = np.argmin(np.abs(Cs_plot-C_vals[belowTH_idx]))
    #     Lidx_belowTH = np.argmin(np.abs(Ls_plot-L_vals[belowTH_idx]))
    #     sample_count_map[Cidx_belowTH,Lidx_belowTH] += 1

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
                        label='ABC-Median')
        ax.set_aspect('equal', adjustable='box')
    #     ax.set_title(f' Posterior Density \n True C = {C_true:.02}, True L = {L_true:.02} \n ABC-Threshold = {distance_threshold} ',fontsize=16)
    #     plt.title(f'ABC-Posterior Density\n', fontsize=25)
        title_str = 'True: C = ' + str(round(C_true,3)) + ', L = ' + str(round(L_true,3)) + ', W = ' + str(round(W_true,3)) + ' \n Median: C = ' + str(round(C_median,3)) + ', L = ' + str(round(L_median,3)) + ', W = ' + str(round(W_median,3)) + '\n W Plot Val = ' + str(Ws[each_w])
        plt.title(title_str, fontsize=18)#, horizontalalignment='center', x=0.535, y=0.85)
    #     plt.suptitle(f'(ABC-Threshold = {distance_threshold})\n', fontsize=15,  y=0.5)
        plt.xlabel("C", fontsize=20)
        plt.ylabel("L", fontsize=20)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        if (Cidx == 25 and Lidx == 15) or (Cidx == 25 and Lidx == 25):
            pass
        else:
            ax.set_xticks([0.1, 1.0, 2.0, 3.0])
            ax.set_yticks([0.1, 1.0, 2.0, 3.0])
        
        save_dir = './Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/ABC_angle_plot_updated'
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        fig.tight_layout()
        fig.savefig(save_dir+'/ABC_'+'Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'_angles_at_w'+str(each_w).zfill(2)+'_OGw05.pdf',bbox_inches='tight')
 
        plt.close()   


    


    ############ 3D PLOTS ##############
    
    C_mesh_plot, L_mesh_plot, W_mesh_plot= np.meshgrid(Cs_plot, Ls_plot, Ws_plot, indexing='ij')
    
    # fig, ax = plt.subplots(1,1,figsize=(6,6),dpi=400,projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(C_mesh_plot, L_mesh_plot, W_mesh_plot, c=sample_count_map, cmap='viridis')
    # plt.contourf([C_mesh_plot, L_mesh_plot], W_mesh_plot, sample_count_map)
    ax.scatter(C_true,L_true,W_true,
                c="w",
                linewidths = 0.5,
                marker = '*',
                edgecolor = "k",
                s = 500,
                label='True')
    ax.scatter(C_median,L_median,W_median,
                c="k",
                linewidths = 0.5,
                marker = 'o',
                edgecolor = "k",
                s = 180,
                label='ABC-Median')

########## NEED TO CHANGE? ###########
    # if (Cidx == 20 and Lidx == 1) or (Cidx == 5 and Lidx == 1) or (Cidx == 15 and Lidx == 2):
    #     ax.legend(fontsize=15)
    ax.set_aspect('equal', adjustable='box')
#     ax.set_title(f' Posterior Density \n True C = {C_true:.02}, True L = {L_true:.02} \n ABC-Threshold = {distance_threshold} ',fontsize=16)
#     plt.title(f'ABC-Posterior Density\n', fontsize=25)
    # title_str = 'True C = {0:.02}, True L = {1:.02}, True W = {0:.00}'.format(C_true, L_true, W_true)
    title_str = 'True: C = ' + str(round(C_true,3)) + ', L = ' + str(round(L_true,3)) + ', W = ' + str(round(W_true,3)) + ' \n Median: C = ' + str(round(C_median,3)) + ', L = ' + str(round(L_median,3)) + ', W = ' + str(round(W_median,3))
    ax.set_title(title_str, fontsize=18)#, horizontalalignment='center', x=0.535, y=0.85)
#     plt.suptitle(f'(ABC-Threshold = {distance_threshold})\n', fontsize=15,  y=0.5)
    ax.set_xlabel("C", fontsize=20)
    ax.set_ylabel("L", fontsize=20)
    ax.set_zlabel("W", fontsize=20)
    # plt.zlabel("W", fontsize=20)
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='z', labelsize=15)
    if (Cidx == 25 and Lidx == 15) or (Cidx == 25 and Lidx == 25):
        pass
    else:
        ax.set_xticks([0.1, 1.0, 2.0, 3.0])
        ax.set_yticks([0.1, 1.0, 2.0, 3.0])
    
    fig.tight_layout()
    fig.savefig('./Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/ABC_'+'Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'_angles_OGw05.pdf',bbox_inches='tight')
