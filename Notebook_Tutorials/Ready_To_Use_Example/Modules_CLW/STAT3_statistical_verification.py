import sys 
import numpy as np
import pandas as pd 
import os
os.environ["NUMBA_DISABLE_CACHING"] = "1"
import numba
numba.config.DISABLE_CACHING = True
from sklearn.metrics import pairwise_distances
from skbio.stats.distance import DistanceMatrix
from skbio.stats.distance import permanova
from scipy.stats import ttest_ind
from scipy.spatial.distance import cdist
from hyppo.ksample import Energy
from hyppo.ksample import MMD


def statistical_verification(C,L,W):

    ### goal: load all of the results and put into dataframe ###
    
    #the dataframe 
    stats_results_df = pd.DataFrame()
    
    #ranges for C, L, W parameters 
    Cs = np.linspace(0.1,3.0,30)
    Ls = np.linspace(0.1,3.0,30)
    Ws = np.linspace(0.0,0.1,11) 

    # C, L, W values    
    C_true = Cs[C]
    L_true = Ls[L]
    W_true = Ws[W]
    
    #Flatten Model_AL crockers 

    Model_AL_crockers_path = './posterior_samples/'

    # Already created for this example 
    output_file = './Statistical_Verification_Files/all_AL_matrices.npy'

    if os.path.exists(output_file):
        all_AL_matrices = np.load(output_file)

    else:

        # List all subfolders that start with "run_"
        run_folders = sorted(
            [f for f in os.listdir(Model_AL_crockers_path) if f.startswith("run_")],
            key=lambda x: int(x.split("_")[1])  # ensures run_1, run_2, ... run_500 order
        )

        all_AL_matrices = []

        for folder in run_folders:
            file_path = os.path.join(Model_AL_crockers_path, folder, "crocker_angles.npy")
            
            # Load the matrix
            M = np.load(file_path)  # expected shape: (100, 200, 2)
            
            # Flatten the matrix (row-major order)
            M_flat = M.reshape(-1)
            
            # Store
            all_AL_matrices.append(M_flat)

        # Stack into one matrix of shape (num_runs, flattened_size)
        all_AL_matrices = np.vstack(all_AL_matrices)

        # print("Final AL shape:", all_AL_matrices.shape)

        # Save for later use
        np.save(output_file, all_AL_matrices)


    #Flatten Model_DO crockers 

    # Will need to differentiate if creating Model_DO in addition to Model_AL
    Model_DO_crockers_path = './posterior_samples/'

    # Already created for this example 
    output_file = './Statistical_Verification_Files/all_DO_matrices.npy'

    if os.path.exists(output_file):
        all_DO_matrices = np.load(output_file)

    else:

        # List all subfolders that start with "run_"
        run_folders = sorted(
            [f for f in os.listdir(Model_DO_crockers_path) if f.startswith("run_")],
            key=lambda x: int(x.split("_")[1])  # ensures run_1, run_2, ... run_500 order
        )

        all_DO_matrices = []

        for folder in run_folders:
            file_path = os.path.join(Model_DO_crockers_path, folder, "crocker_angles.npy")
            
            # Load the matrix
            M = np.load(file_path)  # expected shape: (100, 200, 2)
            
            # Flatten the matrix (row-major order)
            M_flat = M.reshape(-1)
            
            # Store
            all_DO_matrices.append(M_flat)

        # Stack into one matrix of shape (num_runs, flattened_size)
        all_DO_matrices = np.vstack(all_DO_matrices)

        # print("Final AL shape:", all_AL_matrices.shape)

        # Save for later use
        np.save(output_file, all_DO_matrices)


    # rename flattened matrices 
    groupAL = all_AL_matrices
    groupDO = all_DO_matrices


    # permanova statistical test 

    # Combine
    data = np.vstack([groupAL, groupDO])

    # Create labels for group membership
    labels = np.array(["AL"] * groupAL.shape[0] + ["DO"] * groupDO.shape[0])

    # Compute distance matrix (Euclidean works well for high-dim)
    D = pairwise_distances(data, metric='euclidean')

    # Convert to skbio distance matrix form
    dm = DistanceMatrix(D)

    # Run PERMANOVA
    result = permanova(dm, labels, permutations=999)
    # print(result)

    # Extract results 
    perma_stat = result['test statistic']
    perma_pval = result['p-value']   



    # Energy Distance Test

    energy_stat, energy_pval = Energy().test(groupAL, groupDO)
    # print(stat, pval)    

    
    # Maximum Mean Discrepancy (MMD)

    mmd_stat, mmd_pval = MMD().test(groupAL, groupDO)
    # print(stat, pval)


    # putting it all together to add to dataframe 
    new_row = pd.DataFrame({
        'C_true': [round(C_true, 3)],
        'L_true': [round(L_true, 3)],
        'W_true': [round(W_true, 3)],
        'perma_stat': [perma_stat],
        'perma_pval': [perma_pval],
        'perma_pval_r': [round(perma_pval, 3)],
        'energy_stat': [round(energy_stat, 3)],
        'energy_pval': [energy_pval],
        'energy_pval_r': [round(perma_pval, 3)],
        'mmd_stat': [round(mmd_stat, 3)],
        'mmd_pval': [mmd_pval],
        'mmd_pval_r': [round(perma_pval, 3)],
    })
    stats_results_df = pd.concat([stats_results_df, new_row], ignore_index=True)

    #dataframe column names
    stats_results_df.columns = ['C_true', 'L_true', 'W_true', 'PERMANOVA Stat', 'PERMANOVA p-value', 'Permanova p-value R', 'Energy Stat', 'Energy p-value', 'Energy p-value R', 'MMD Stat', 'MMD p-value', 'MMD p-value R']
    
    #save the dataframe 
    stats_results_df.to_csv('./Chosen_C_'+str(C).zfill(2)+'_L_'+str(L).zfill(2)+'_W_'+str(W).zfill(2)+'/stats_results.csv')

