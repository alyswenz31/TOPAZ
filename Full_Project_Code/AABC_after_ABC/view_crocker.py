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
from scipy.integrate import ode
import glob
import imageio as io
from itertools import repeat

from Scripts.DorsognaNondim_Align import *
from Scripts.crocker import *

for iSample in range(10):

    # Adjust folder and sample number as needed for your specific case
    print(f'Processing sample {iSample+1}/10')
    if not os.path.exists(f'./Smaller_Denser_Grid_06_24/Model_AL/Simulated_Grid/ODE_Align/sample_20800/run_{iSample+1}/'):
        print('Path does not exist, skipping...')
        pass
    else:  
        if os.path.exists(f'./Smaller_Denser_Grid_06_24/Model_AL/Simulated_Grid/ODE_Align/sample_20800/run_{iSample+1}/crocker.pdf'):
            print('Crocker plot already exists, skipping...')
            pass
        elif not os.path.exists(f'./Smaller_Denser_Grid_06_24/Model_AL/Simulated_Grid/ODE_Align/sample_20800/run_{iSample+1}/crocker_angles.npy'):
            print('Crocker data does not exist, skipping...')
            pass
        else:
            print('Generating crocker plot...')
            true_path = f'./Smaller_Denser_Grid_06_24/Model_AL/Simulated_Grid/ODE_Align/sample_20800/run_{iSample+1}/crocker_angles.npy'
            true_crocker = np.load(true_path, allow_pickle=True)
            PROX_VEC = 10**(np.linspace(-2,2,200)) 
            # [Cval,Lval,Wval] = np.load(f'./Smaller_Denser_Grid_17_03_Possible_Error/Model_AL/Simulated_Grid/ODE_Align/sample_aabc_22880/run_{iSample+1}/theta_star.npy', allow_pickle=True)
            pars_vals = np.load(f'./Smaller_Denser_Grid_06_24/Model_AL/Simulated_Grid/ODE_Align/sample_20800/run_{iSample+1}/pars.npy', allow_pickle=True)
            # [Cval,Lval,Wval] = pars_vals[3,4,5]
            [Cval,Lval,Wval] = [0,0,0]

            plot_crocker_highres_split(true_crocker,
                                        PROX_VEC,
                                        [50,150,250,350,450],
                                        true_crocker,
                                        PROX_VEC,
                                        [50,150,250,350,450],
                                        [Cval,Lval,Wval],
                                            save_path=f'./Smaller_Denser_Grid_06_24/Model_AL/Simulated_Grid/ODE_Align/sample_20800/run_{iSample+1}/crocker.pdf')