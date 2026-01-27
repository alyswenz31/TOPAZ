


    
    # get all crockers and parameters from each sample and run
    for CLvals in CLoptions:
        C_val, L_val = CLvals
        model_options = ('AL','DO')
        # model_options = ('AL')
        for model_option in model_options:
            if model_option == 'AL':
                BASE_DIR = './Smaller_Denser_Grid_' + str(C_val).zfill(2) + '_' + str(L_val).zfill(2) + '/Model_AL/Simulated_Grid/ODE_Align/'
                # Regex for folders named sample_#####
                sample_pattern = re.compile(r"sample_(\d{5})$")
            elif model_option == 'DO':
                BASE_DIR = './Smaller_Denser_Grid_' + str(C_val).zfill(2) + '_' + str(L_val).zfill(2) + '/Model_DO/Simulated_Grid/ODE_Align/'
                # Regex for folders named sample_#####
                sample_pattern = re.compile(r"sample_(\d{4})$")
    
            # Loop over all sample folders
            for sample_name in os.listdir(BASE_DIR):
                sample_path = os.path.join(BASE_DIR, sample_name)
    
                # Only keep folders matching sample_#####
                if not os.path.isdir(sample_path):
                    continue
                if not sample_pattern.match(sample_name):
                    continue
    
                print(f"Processing {sample_name}")
    
                # create lists to hold all crockers and parameters
                all_crockers = []
                all_params = []
    
                # Loop through run_# folders
                for run_name in os.listdir(sample_path):
                    run_path = os.path.join(sample_path, run_name)
    
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
    
                    # ----- DO SOMETHING WITH data -----
                    # save the crockers in one large file 
                    all_crockers.append(crocker_flat)
    
    
                    # get the parameters 
                    # Path to the file we want
                    pars_path = os.path.join(run_path, "pars.npy")
    
                    # Skip if missing
                    if not os.path.isfile(pars_path):
                        continue
    
                    # Load the file
                    [SIGMA, ALPHA, BETA, C, L, W] = np.load(pars_path, allow_pickle=True)
                    params = [C, L, W]
    
                    # ----- DO SOMETHING WITH data -----
                    # save the parameters in one large file
                    all_params.append(params)
    
                # save all crockers and parameters to one file
                all_crockers_array = np.vstack(all_crockers)
                all_params_array = np.vstack(all_params)
    
                save_crocker_path = os.path.join(BASE_DIR, "all_crockers_flattened.npy")
                np.save(save_crocker_path, all_crockers_array)
                print(f"Saved all crockers to {save_crocker_path}, shape = {all_crockers_array.shape}")
    
                save_params_path = os.path.join(BASE_DIR, "all_params.npy")
                np.save(save_params_path, all_params_array)
                print(f"Saved all parameters to {save_params_path}, shape = {all_params_array.shape}")