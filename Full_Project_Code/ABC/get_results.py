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
# pars_idc = [(1,2,0), (1,2,4), (6,1,0), (6,1,4), (3,9,0), (3,9,4)]
pars_idc = [(17,3,0),(6,24,0),(19,0,0),(8,5,0),(4,4,0),(1,14,0),(14,6,0),(24,24,0),(19,14,0),(17,3,5),(6,24,5),(19,0,5),(8,5,5),(4,4,5),(1,14,5),(14,6,5),(24,24,5),(19,14,5)]

### load the results and put into dataframe for each chosen C,L,W grouping and each original w value ###

# for OG w values 0 and 0.05
OGw_options = [0,5]

# running over all combinations of pars_idc and OGw_options to load and save the results 
for OGw in OGw_options:
    for pars_idx in pars_idc:

        # C, L, W indices
        Cidx, Lidx, Widx = pars_idx

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
                change_median_path = './ChangeW_00/Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/medians_OGw00.npy'
                no_change_median_path = './NoChangeW_00/Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/medians_OGw00.npy'
    
            else:
            #for OG w = 0.05
                change_median_path = './ChangeW_05/Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/medians_OGw05.npy'
                no_change_median_path = './NoChangeW_05/Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/medians_OGw05.npy'
            change_medians = np.load(change_median_path,allow_pickle=True)
            no_change_medians = np.load(no_change_median_path,allow_pickle=True)
            
            
            # Final BIC values
            if OGw == 0:
                #for OG w = 0
                change_bic_path = './ChangeW_00/Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/bic_results_final_OGw00.npy'
                no_change_bic_path = './NoChangeW_00/Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/bic_result_finals_OGw00.npy'
        
            else:
                #for OG w = 0.05
                change_bic_path = './ChangeW_05/Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/bic_results_final_OGw05.npy'
                no_change_bic_path = './NoChangeW_05/Simulated_Grid/ODE_Align/Cidx_'+str(Cidx).zfill(2)+'_Lidx_'+str(Lidx).zfill(2)+'_Widx_'+str(Widx).zfill(2)+'/run_1/bic_results_final_OGw05.npy'
                
            change_bics = np.load(change_bic_path,allow_pickle=True)
            no_change_bics = np.load(no_change_bic_path,allow_pickle=True)
        
            # putting it all together to add to dataframe 
            new_row = pd.DataFrame({'OG w value': [OGw_value], 
                       'C_true': round(C_true,3), 
                       'L_true': round(L_true,3), 
                       'W_true': round(W_true,3), 
                       'M1 C_median': round(change_medians[0],3), 
                       'M1 L_median': round(change_medians[1],3), 
                       'M1 W_median': round(change_medians[2],3), 
                       'M2 C_median': round(no_change_medians[0],3), 
                       'M2 L_median': round(no_change_medians[1],3), 
                       'M1 BIC val': round(change_bics[0],5), 
                       'M2 BIC val': round(no_change_bics[0],5), 
                       'M1 RSS val': round(change_bics[1],5), 
                       'M2 RSS val': round(no_change_bics[1],5)})
            CLW_results_df = pd.concat([CLW_results_df, new_row], ignore_index=True)

#dataframe column names
CLW_results_df.columns = ['OG w value', 'C_true', 'L_true', 'W_true', 'M1 C_median', 'M1 L_median', 'M1 W_median', 'M2 C_median', 'M2 L_median', 'M1 BIC val', 'M2 BIC val', 'M1 RSS val', 'M2 RSS val']

#save the dataframe 
CLW_results_df.to_csv('CLW_results_final.csv')
