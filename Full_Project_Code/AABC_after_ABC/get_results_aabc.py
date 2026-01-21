import numpy as np
import pandas as pd 


### goal: load all of the results and put into dataframe ###

#the dataframe 
CLW_results_df = pd.DataFrame()

#ranges for C, L, W parameters 
Cs = np.linspace(0.1,3.0,30)
Ls = np.linspace(0.1,3.0,30)
Ws = np.linspace(0.0,0.1,11)

# chosen C, L, W indices to study 
pars_idc = [(17,3,0),(6,24,0),(19,0,0),(8,5,0),(4,4,0),(1,14,0),(14,6,0),(24,24,0),(19,14,0),(11,6,0),(17,3,5),(6,24,5),(19,0,5),(8,5,5),(4,4,5),(1,14,5),(14,6,5),(24,24,5),(19,14,5),(11,6,5)]

### load the results and put into dataframe for each chosen C,L,W grouping and each original w value ###

# for OG w values 0 and 0.05
OGw_options = [0,5]

# running over all combinations of pars_idc and OGw_options to load and save the results 
for OGw in OGw_options:
    for pars_idx in pars_idc:

        # C, L, W indices
        Cidx, Lidx, Widx = pars_idx

        if Cidx == 14:
            file_path = 'Smaller_Denser_Grid_'+str(11).zfill(2)+'_'+str(6).zfill(2)
        else: 
            file_path = 'Smaller_Denser_Grid_'+str(Cidx).zfill(2)+'_'+str(Lidx).zfill(2)

        if OGw == Widx: 
    
            #getting the OGw_value 
            if OGw == 0:
                OGw_value = 0
            else:
                OGw_value = 0.05
        
            # C, L, W values    
            C_true = Cs[Cidx]
            L_true = Ls[Lidx]
            W_true = Ws[Widx]
            
            
            # Median values 
            if OGw == 0:
            #for OG w = 0.00
                model_al_median_path = file_path+'/Model_AL/Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/medians.npy'
                model_do_median_path = file_path+'/Model_DO/Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/medians.npy'
    
            else:
            #for OG w = 0.05
                model_al_median_path = file_path+'/Model_AL/Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/medians.npy'
                model_do_median_path = file_path+'/Model_DO/Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/medians.npy'
            model_al_medians = np.load(model_al_median_path,allow_pickle=True)
            model_do_medians = np.load(model_do_median_path,allow_pickle=True)
            
            
            # Final BIC values
            if OGw == 0:
                #for OG w = 0
                model_al_bic_path = file_path+'/Model_AL/Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/bic_results.npy'
                model_do_bic_path = file_path+'/Model_DO/Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/bic_results.npy'
        
            else:
                #for OG w = 0.05
                model_al_bic_path = file_path+'/Model_AL/Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/bic_results.npy'
                model_do_bic_path = file_path+'/Model_DO/Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/bic_results.npy'
                
            model_al_bics = np.load(model_al_bic_path,allow_pickle=True)
            model_do_bics = np.load(model_do_bic_path,allow_pickle=True)
        
            # putting it all together to add to dataframe 
            new_row = pd.DataFrame({'C_true': [round(C_true,3)], 
                       'L_true': [round(L_true,3)], 
                       'W_true': [round(W_true,3)], 
                       'AL C_median': [round(model_al_medians[0],3)], 
                       'AL L_median': [round(model_al_medians[1],3)], 
                       'AL W_median': [round(model_al_medians[2],3)], 
                       'DO C_median': [round(model_do_medians[0],3)], 
                       'DO L_median': [round(model_do_medians[1],3)], 
                       'AL BIC val': [round(model_al_bics[0],5)], 
                       'DO BIC val': [round(model_do_bics[0],5)], 
                       'AL RSS val': [round(model_al_bics[1],5)], 
                       'DO RSS val': [round(model_do_bics[1],5)],
                       'AL BIC - DO BIC': [round((model_al_bics[0]-model_do_bics[0]),5)]})
            CLW_results_df = pd.concat([CLW_results_df, new_row], ignore_index=True)

#dataframe column names
CLW_results_df.columns = ['C_true', 'L_true', 'W_true', 'AL C_median', 'AL L_median', 'AL W_median', 'DO C_median', 'DO L_median', 'AL BIC val', 'DO BIC val', 'AL RSS val', 'DO RSS val', 'AL BIC - DO BIC']

#save the dataframe 
CLW_results_df.to_csv('CLW_results_aabc_fixed.csv')
