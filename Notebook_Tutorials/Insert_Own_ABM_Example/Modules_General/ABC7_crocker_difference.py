import numpy as np

def compute_crocker_difference(tda_crocker_angles_path,median_crocker_angles_path,crocker_difference_path):

    # load TDA crocker
    tda_crocker = np.load(tda_crocker_angles_path,allow_pickle=True)

    # load ABC median crocker
    medians_crocker = np.load(median_crocker_angles_path,allow_pickle=True)

    # get the difference in the true (TDA) and simulated (median value) crocker plots 
    crocker_diff = tda_crocker - medians_crocker
    np.save(crocker_difference_path, crocker_diff)
