import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy import io
from sklearn.manifold import TSNE
matplotlib.rcParams.update({'font.size': 5})
import os
import csv
import pdb

############################### Beginning Set Up ###############################

Cs = np.linspace(0.1,3.0,30)
Ls = np.linspace(0.1,3.0,30)
Ws = np.linspace(0, 0.1, 11)

list_tuples = []
for C_idx, C in enumerate(Cs):
	for L_idx, L, in enumerate(Ls):
		list_tuples.append((C_idx, L_idx))

C_idx = 0
L_idx = 1
count = 1



############################### Step 1: Get Crockers and Reshape Them ###############################

if os.path.exists('b01_save.npy'):
    print("File b01_save.npy exists. Skipping step 1.")
else:
    for W_idx, W in enumerate(Ws):
      
        # print("Analyzing W = {}".format(W_idx))
        for C_idx, C in enumerate(Cs):
            for L_idx, L in enumerate(Ls):
                
                
                print("Analyzing C = {}\tL = {}\tW = {}".format(C_idx, L_idx, W_idx))
    
                CROCKER_PATH = './Simulated_Grid/ODE_Align/Cidx_'+str(C_idx).zfill(2)+'_Lidx_'+str(L_idx).zfill(2)+'_Widx_'+str(W_idx).zfill(2)+'/run_1/crocker_angles.npy'
                crocker = np.load(CROCKER_PATH, allow_pickle=True)
                crocker = np.asarray(crocker, dtype='float64')
    
                # T-SNE on the last betti numbers in time
      
                b01 = crocker[:,:,:]
                b01[:,:,0] = b01[:,:,0]/b01[:,:,0].max()
                b01[:,:,1] = b01[:,:,1]/b01[:,:,1].max()
               
                if count ==1:
                    b01_save =  b01.reshape((-1,1))
                else:
                    b01_save = np.concatenate((b01_save, b01.reshape((-1,1))),axis=1)
    
                count += 1
                
    np.save('b01_save.npy',b01_save)



############################### Step 2: Compute t-SNEs ###############################

irun = 0
if os.path.exists('tsne_results_full'+str(irun+1)+'_angles_SD.npy'):
    print("File tsne_results_full1_angles_SD.npy exists. Skipping step 2.")
else:
    num_run = 1
    for irun in range(num_run):
        b01_save = np.load('b01_save.npy', allow_pickle=True)
        # print('b01_save.T size: ' + str((b01_save.T).shape))
        # print(irun,'t-sneeeing')
        #b0_end_embedded  = TSNE(n_components=3,verbose=2).fit_transform(b0_end_mat)
        #b1_end_embedded  = TSNE(n_components=3,verbose=2).fit_transform(b1_end_mat)
        b01_end_embedded = TSNE(n_components=3,verbose=2).fit_transform(b01_save.T)
        # print('b01_end_embedded size: ' + str((b01_end_embedded).shape))
        
        results = {}
        # RGB
        results['RGB'] = {}
        #results['RGB']['b0']  = b0_end_embedded
        #results['RGB']['b1']  = b1_end_embedded
        results['RGB']['b01'] = b01_end_embedded
    
            # np.save('./Simulated_Grid/TSNE_Align/tsne_results_full'+str(irun+1)+'_Widx_'+str(W_idx+1).zfill(2)+'_angles_SD.npy',results)
    np.save('tsne_results_full'+str(irun+1)+'_angles_SD.npy',results)

# print("Job done!")



############################### Normalize t-SNE function  ###############################

def normalize_tsne(tsne_array, percentile):
    """Normalize a single t-SNE array using the given percentiles and clip between 0 and 1."""
    for idx in range(tsne_array.shape[2]):
        tsne_array[:,:,idx] = (tsne_array[:,:,idx] - percentile[0, idx]) / (percentile[1, idx] - percentile[0, idx])
    return np.clip(tsne_array, 0, 1)



############################### Filter t-SNE function ###############################
    
def filter_tsne(irun=0):
    
    npy_path = 'tsne_results_full'+str(irun+1)+'_angles_SD.npy'
    tsne_results = np.load(npy_path, allow_pickle=True).item()
    # print('tsne_results size: ' + str(len(tsne_results)))
    # pdb.set_trace()

    # tsne_results

    #b0_tsne  = tsne_results['RGB']['b0']
    #b1_tsne  = tsne_results['RGB']['b1']
    b01_tsne = tsne_results['RGB']['b01']
    # print('in filter_tsne b01_tsne size: ' + str(b01_tsne.shape))
    #3x(900*11)
    #need to reshape into 
    # b0_tsne = b0_tsne.reshape((30,30,-1))
    # b1_tsne = b1_tsne.reshape((30,3q0,-1))
    # b01_tsne = b01_tsne.reshape((30,30,-1))

# Assuming b0_tsne, b1_tsne, and b01_tsne are already defined as numpy arrays
    # reshaped_dicts = {'b0_tsne': {}, 'b1_tsne': {}, 'b01_tsne': {}}
    reshaped_dicts = {'b01_tsne': {}}
    
    # Loop through the 11 groups
    for i in range(11):
        start_col = i * 900
        end_col = (i + 1) * 900
        
        # Reshape for b0_tsne
        # reshaped_dicts['b0_tsne'][i + 1] = b0_tsne[:, start_col:end_col].reshape(30, 30, 3)
        
        # Reshape for b1_tsne
        # reshaped_dicts['b1_tsne'][i + 1] = b1_tsne[:, start_col:end_col].reshape(30, 30, 3)
        
        # Reshape for b01_tsne
#suggestion to try and figure out if reshaping is correct or not:        
        # b01_tsne[900:1800, 0] = np.zeros(900)
        # b01_tsne[900:1800, 1] = np.zeros(900)
        # b01_tsne[900:1800, 2] = np.ones(900)
        reshaped_dicts['b01_tsne'][i] = b01_tsne[start_col:end_col, :].reshape(30, 30, 3)
    


    # Loop through each group in reshaped_dicts and apply normalization and clipping
    for group_id in range(0, 11):  # Looping over groups 1 to 11
        # Extract the arrays for b0_tsne, b1_tsne, and b01_tsne for the current group
        # b0_array = reshaped_dicts['b0_tsne'][group_id]
        # b1_array = reshaped_dicts['b1_tsne'][group_id]
        b01_array = reshaped_dicts['b01_tsne'][group_id]
        
        # Calculate the percentiles for the current group
        # b0_percentile  = np.percentile(b0_array, [1, 99], axis=[0, 1])
        # b1_percentile  = np.percentile(b1_array, [1, 99], axis=[0, 1])
        b01_percentile = np.percentile(b01_array, [1, 99], axis=[0, 1])
        
        # Normalize and clip each array for the current group
        # reshaped_dicts['b0_tsne'][group_id] = normalize_tsne(b0_array, b0_percentile)
        # reshaped_dicts['b1_tsne'][group_id] = normalize_tsne(b1_array, b1_percentile)         
       
        reshaped_dicts['b01_tsne'][group_id] = normalize_tsne(b01_array, b01_percentile) #NEED TO ADD BACK IN

    # b0_percentile  = np.percentile(b0_tsne, [1,99], axis=[0,1]) #np.min(b0_tsne[:,:,idx])
    # b1_percentile  = np.percentile(b1_tsne, [1,99], axis=[0,1]) #np.min(b0_tsne[:,:,idx])
    # b01_percentile = np.percentile(b01_tsne, [1,99], axis=[0,1]) #np.min(b0_tsne[:,:,idx])

    # for idx in range(b0_tsne.shape[2]):

    #     b0_tsne[:,:,idx]  = (b0_tsne[:,:,idx] - b0_percentile[0,idx])/(b0_percentile[1,idx] - b0_percentile[0,idx])
    #     b1_tsne[:,:,idx]  = (b1_tsne[:,:,idx] - b1_percentile[0,idx])/(b1_percentile[1,idx] - b1_percentile[0,idx])
    #     b01_tsne[:,:,idx] = (b01_tsne[:,:,idx] - b01_percentile[0,idx])/(b01_percentile[1,idx] - b01_percentile[0,idx])

    # b0_tsne  = np.clip(b0_tsne, 0, 1)
    # b1_tsne  = np.clip(b1_tsne, 0, 1)
    # b01_tsne = np.clip(b01_tsne, 0, 1)
    
    return reshaped_dicts




############################### Put it all together to get rainbow plots  ###############################

if os.path.exists('Widx_Total_tsne_crocker_heatmap_angles_SD_updated.pdf'):
    print("2D Rainbow plots exists. Skipping this step.")
else:
    
    w_vals = list(np.linspace(0, 0.1, 11))
    
    num_rows = len(w_vals)
    print('num_rows size: ' + str(num_rows))
    num_cols = 3
    num_subplots = num_rows * num_cols
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8, 20), dpi=400, constrained_layout = True)
    tsne_dictionary = filter_tsne(0)
    
    for w_idx, w in enumerate(w_vals):
        
        # b0_tsne = tsne_dictionary['b0_tsne'][w_idx]
        # b1_tsne = tsne_dictionary['b1_tsne'][w_idx]
        b01_tsne = tsne_dictionary['b01_tsne'][w_idx]
        for j, ax in enumerate(axs[w_idx,:]): # type: ignore
    
            j=2
        
            if j == 0:
                ax.imshow(np.swapaxes(b0_tsne, 0, 1))
                ax.invert_yaxis()
                ax.set_xlabel('C')
                ax.set_ylabel('L')
                plt.setp(ax, xticks=[0, 9, 19, 29], xticklabels=[0.1, 1.0, 2.0, 3.0],
                            yticks=[0, 9, 19, 29], yticklabels=[0.1, 1.0, 2.0, 3.0])
                if w_idx == 0:
                    ax.set_title('t-SNE: Betti-0')
        
            elif j == 1:
                ax.imshow(np.swapaxes(b1_tsne, 0, 1))
                ax.invert_yaxis()
                ax.set_xlabel('C')
                plt.setp(ax, xticks=[0, 9, 19, 29], xticklabels=[0.1, 1.0, 2.0, 3.0], 
                            yticks=[0, 9, 19, 29], yticklabels=[0.1, 1.0, 2.0, 3.0])
                if w_idx == 0:
                    ax.set_title("t-SNE: Betti-1")
        
            else:
                ax.imshow(np.swapaxes(b01_tsne, 0, 1))
                ax.invert_yaxis()
                ax.set_xlabel('C')
                ax.set_ylabel('L')
                w_title = 'W = ' + str(np.round(w, 2))
               # ax.set_ylabel(w_title, rotation='vertical', labelpad=20)
                #ax.yaxis.set_label_position('right')
                ax.set_title(w_title)
                plt.setp(ax, xticks=[0, 9, 19, 29], xticklabels=[0.1, 1.0, 2.0, 3.0], 
                        yticks=[0, 9, 19, 29], yticklabels=[0.1, 1.0, 2.0, 3.0])
                # if w_idx == 0:
                    # ax.set_title("t-SNE: Betti-0 & 1")

                
           
    # fig.savefig('./Simulated_Grid/TSNE_Align/Widx_Total_tsne_crocker_heatmap_angles_SD.pdf', bbox_inches='tight')
    fig.savefig('Widx_Total_tsne_crocker_heatmap_angles_SD_updated.pdf',bbox_inches='tight') # added _BW
    



############################### Code to get 3D rainbow dot plots with 9900 points #############################

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

if os.path.exists('tsne_crocker_3D_angles_SD_all9900.pdf'):
    print("3D Rainbow plots with 9900 exists. Skipping this step.")
else:
    
    w_vals = list(np.linspace(0, 0.1, 11))
    
    for w_idx, w in enumerate(w_vals):
        # b0_tsne, b1_tsne, b01_tsne = make_tsne(w_idx)
        tsne_dictionary = filter_tsne(0)
        # b01_tsne = tsne_dictionary['b01_tsne'][w_idx]
        if w_idx == 0:
            b01_tsne = tsne_dictionary['b01_tsne'][w_idx]
            # print ('b01_tsne important shape: '+ str(b01_tsne.shape))
        else:
            b01_tsne = np.append(b01_tsne,tsne_dictionary['b01_tsne'][w_idx])
            # print ('b01_tsne important shape: '+ str(b01_tsne.shape))
            
    
    # b0_tsne_color = np.squeeze(b0_tsne.reshape((-1,1,3)))
    # b1_tsne_color = np.squeeze(b1_tsne.reshape((-1,1,3)))
    b01_tsne_color = np.squeeze(b01_tsne.reshape((-1,1,3)))
    b01_tsne_color_normalized = (b01_tsne_color - b01_tsne_color.min()) / (b01_tsne_color.max() - b01_tsne_color.min())
    print('b01_tsne_color size: ' + str(b01_tsne_color.shape))
    
    # mat_tsne = {}
    # mat_tsne['norm_tsne'] = {}
    # # mat_tsne['norm_tsne']['b0_tsne'] = np.swapaxes(b0_tsne, 0, 1)
    # # mat_tsne['norm_tsne']['b1_tsne'] = np.swapaxes(b1_tsne, 0, 1)
    # mat_tsne['norm_tsne']['b01_tsne'] = np.swapaxes(b01_tsne, 0, 1)
    
    # file_path = './Simulated_Grid/TSNE_Align'
    tsne_results = np.load('tsne_results_full'+str(irun+1)+'_angles_SD.npy',allow_pickle=True).item()
    
    # b0_tsne  = tsne_results['RGB']['b0']
    # b1_tsne  = tsne_results['RGB']['b1']
    b01_tsne = tsne_results['RGB']['b01']
    
    # start_col = w_idx * 900
    # end_col = (w_idx + 1) * 900
    # b01_tsne = b01_tsne[start_col:end_col, :]
    print('b01_tsne size: ' + str(b01_tsne.shape))
    
    # mat_tsne['tsne'] = {}
    # # mat_tsne['tsne']['b0_tsne'] = np.swapaxes(b0_tsne, 0, 1)
    # # mat_tsne['tsne']['b1_tsne'] = np.swapaxes(b1_tsne, 0, 1)
    # mat_tsne['tsne']['b01_tsne'] = np.swapaxes(b01_tsne, 0, 1)
    
    # # scipy.io.savemat('Widx_'+str(w_idx).zfill(2)+'_tsne_mat_angles_SD.mat',mat_tsne)
    # scipy.io.savemat('tsne_mat_angles_SD.mat',mat_tsne)
    
    
    fig=plt.figure(figsize=(35, 10))
    fig.suptitle(f'Alignment, W = {np.round(w, 2)}', fontsize=40, y=1.05)
    # ax1=fig.add_subplot(1,3,1,projection='3d', adjustable="box")
    # ax2=fig.add_subplot(1,3,2,projection='3d', adjustable="box")
    # ax3=fig.add_subplot(1,3,3,projection='3d', adjustable="box")
    ax3=fig.add_subplot(1,3,1,projection='3d', adjustable="box")
    fig.subplots_adjust(top=0.9)
    # for idx in range(b0_tsne.shape[0]):
    for idx in range(b01_tsne.shape[0]):
        # print(b01_tsne_color[idx,:])
        # ax1.scatter(b0_tsne[idx,0], b0_tsne[idx,1], b0_tsne[idx,2], c=np.squeeze(b0_tsne_color.reshape((-1,1,3)))[idx,:][:,None].T)
        # ax2.scatter(b1_tsne[idx,0], b1_tsne[idx,1], b1_tsne[idx,2], c=np.squeeze(b1_tsne_color.reshape((-1,1,3)))[idx,:][:,None].T)
        ax3.scatter(b01_tsne[idx,0], b01_tsne[idx,1], b01_tsne[idx,2], c=b01_tsne_color_normalized[idx,:])
        # ax3.scatter(b01_tsne[idx,0], b01_tsne[idx,1], b01_tsne[idx,2], c=np.squeeze(b01_tsne_color.reshape((-1,1,3)))[idx,:][:,None].T)
            
        
    # ax1.set_title("t-SNE on Parameters: Betti-0", fontsize=32)
    # ax2.set_title("t-SNE on Parameters: Betti-1", fontsize=32)
    ax3.set_title("t-SNE on Parameters: Betti-0 and Betti-1", fontsize=32)
    # ax1.set_xlabel('t-SNE on Pars Dim. 1', fontsize=20)
    # ax1.set_ylabel('t-SNE on Pars Dim. 2', fontsize=20)
    # ax1.set_zlabel('t-SNE on Pars Dim. 3', fontsize=20)
    # ax2.set_xlabel('t-SNE on Pars Dim. 1', fontsize=20)
    # ax2.set_ylabel('t-SNE on Pars Dim. 2', fontsize=20)
    # ax2.set_zlabel('t-SNE on Pars Dim. 3', fontsize=20)
    ax3.set_xlabel('t-SNE on Pars Dim. 1', fontsize=20)
    ax3.set_ylabel('t-SNE on Pars Dim. 2', fontsize=20)
    ax3.set_zlabel('t-SNE on Pars Dim. 3', fontsize=20)
    # ax1.tick_params(axis='both', which='major', labelsize=12)
    # ax1.tick_params(axis='both', which='minor', labelsize=10)
    # ax2.tick_params(axis='both', which='major', labelsize=12)
    # ax2.tick_params(axis='both', which='minor', labelsize=10)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax3.tick_params(axis='both', which='minor', labelsize=10)
    
    fig.tight_layout(h_pad=5)
    fig.savefig('tsne_crocker_3D_angles_SD_all9900.pdf',bbox_inches='tight')
    print(f'Saved figure {w_idx}')
    

############################### Code to get 3D rainbow dot plots with 990 points #############################

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


w_vals = list(np.linspace(0, 0.1, 11))

for w_idx, w in enumerate(w_vals):
    tsne_dictionary = filter_tsne(0)
    if w_idx == 0:
        b01_tsne = tsne_dictionary['b01_tsne'][w_idx]
    else:
        b01_tsne = np.append(b01_tsne,tsne_dictionary['b01_tsne'][w_idx])
        

b01_tsne_color = np.squeeze(b01_tsne.reshape((-1,1,3)))
b01_tsne_color_normalized = (b01_tsne_color - b01_tsne_color.min()) / (b01_tsne_color.max() - b01_tsne_color.min())
# print('b01_tsne_color size: ' + str(b01_tsne_color.shape))


tsne_results = np.load('tsne_results_full'+str(irun+1)+'_angles_SD.npy',allow_pickle=True).item()

b01_tsne = tsne_results['RGB']['b01']


# print('b01_tsne size: ' + str(b01_tsne.shape))


ten_perc = np.random.choice(9900, size=990, replace=False)


fig=plt.figure(figsize=(35, 10))
# fig.suptitle(f'Alignment, W = {np.round(w, 2)}', fontsize=40, y=1.05)
ax3=fig.add_subplot(1,3,1,projection='3d', adjustable="box")
fig.subplots_adjust(top=0.9)

for i in range(ten_perc.shape[0]):
    idx = ten_perc[i]
    # ax3.scatter(b01_tsne[idx,0], b01_tsne[idx,1], b01_tsne[idx,2], c=b01_tsne_color_normalized[idx,:])
    ax3.scatter(b01_tsne[idx,0], b01_tsne[idx,1], b01_tsne[idx,2], c='gray')
    
# ax3.set_title("t-SNE on Parameters: Betti-0 and Betti-1", fontsize=32)
ax3.set_xlabel('t-SNE Dim. 1', fontsize=20, labelpad=10)
ax3.set_ylabel('t-SNE Dim. 2', fontsize=20, labelpad=10)
ax3.set_zlabel('t-SNE Dim. 3', fontsize=20, labelpad=10)
ax3.tick_params(axis='both', which='major', labelsize=12)
ax3.tick_params(axis='both', which='minor', labelsize=10)

# fig.tight_layout(h_pad=5)
# fig.savefig('tsne_crocker_3D_angles_SD_fixed.pdf',bbox_inches='tight')
fig.savefig('tsne_crocker_3D_angles_SD_gray2.pdf')
# fig.savefig('tsne_crocker_3D_angles_SD_fixed.pdf')
# print(f'Saved figure {w_idx}')

fig=plt.figure(figsize=(35, 10))
# fig.suptitle(f'Alignment, W = {np.round(w, 2)}', fontsize=40, y=1.05)
ax3=fig.add_subplot(1,3,1,projection='3d', adjustable="box")
fig.subplots_adjust(top=0.9)

for i in range(ten_perc.shape[0]):
    idx = ten_perc[i]
    ax3.scatter(b01_tsne[idx,0], b01_tsne[idx,1], b01_tsne[idx,2], c=b01_tsne_color_normalized[idx,:])
    # ax3.scatter(b01_tsne[idx,0], b01_tsne[idx,1], b01_tsne[idx,2], c='gray')
    
# ax3.set_title("t-SNE on Parameters: Betti-0 and Betti-1", fontsize=32)
ax3.set_xlabel('t-SNE Dim. 1', fontsize=20, labelpad=10)
ax3.set_ylabel('t-SNE Dim. 2', fontsize=20, labelpad=10)
ax3.set_zlabel('t-SNE Dim. 3', fontsize=20, labelpad=10)
ax3.tick_params(axis='both', which='major', labelsize=12)
ax3.tick_params(axis='both', which='minor', labelsize=10)

# fig.tight_layout(h_pad=5)
# fig.savefig('tsne_crocker_3D_angles_SD_fixed.pdf',bbox_inches='tight')
fig.savefig('tsne_crocker_3D_angles_SD_color2.pdf')
# fig.savefig('tsne_crocker_3D_angles_SD_fixed.pdf')
print(f'Saved figure {w_idx}')



