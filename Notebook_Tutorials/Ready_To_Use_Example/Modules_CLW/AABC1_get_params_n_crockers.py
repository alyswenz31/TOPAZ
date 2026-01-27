import os 
import numpy as np
import re

# for loop over crocker files in a directory and get parameters from each crocker file 
CLoptions = [(6,24)] # Update to include all C and L combinations you want to process


def get_params_n_crockers_aabc(samples_path,C_idx,L_idx,W_idx):

    # get all crockers and parameters from each sample and run

    # create lists to hold all crockers and parameters
    all_crockers = []
    all_params = []

    # Loop through run_# folders
    for run_name in os.listdir(samples_path):
        run_path = os.path.join(samples_path, run_name)

        if not os.path.isdir(run_path):
            continue
        if not run_name.startswith("run_"):
            continue

        # get the crockers 
        # Path to the file we want
        crocker_path = os.path.join(run_path, "crocker_angles.npy")

        # Skip if missing
        if not os.path.isfile(crocker_path):
            continue

        # Load the file
        crockers = np.load(crocker_path, allow_pickle=True)

        # flatten the crockers 
        crocker_flat = crockers.reshape(-1) 

        # save the crockers in one large file 
        all_crockers.append(crocker_flat)

        
        # get the parameters 
        # Path to the file we want
        pars_path = os.path.join(run_path, "pars.npy")

        # Skip if missing
        if not os.path.isfile(pars_path):
            continue

        # Load the file
        [SIGMA, ALPHA, BETA, C_par, L_par, W_par] = np.load(pars_path, allow_pickle=True)
        params = [C_par, L_par, W_par]

        # save the parameters in one large file
        all_params.append(params)

    # save all crockers and parameters to one file
    all_crockers_array = np.vstack(all_crockers)
    all_params_array = np.vstack(all_params)


    BASE_DIR = './Chosen_C_'+str(C_idx).zfill(2)+'_L_'+str(L_idx).zfill(2)+'_W_'+str(W_idx).zfill(2)+'/'
    save_crocker_path = os.path.join(BASE_DIR, "all_crockers_flattened.npy")
    np.save(save_crocker_path, all_crockers_array)

    save_params_path = os.path.join(BASE_DIR, "all_params.npy")
    np.save(save_params_path, all_params_array)

    return [all_params_array, all_crockers_array]
