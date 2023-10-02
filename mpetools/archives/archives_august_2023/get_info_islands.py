"""
This module allows us to retrieve the currently available data about a given island and print it.
It opens the file containing the dictionary `info_{island}_{country}.data`.
If no information is available yet, the code suggests to run `pre_timeseries_steps.py`.

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import pickle
from mpetools import pre_timeseries_steps
import os
import pandas as pd

def retrieve_info_island(island, country, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands'), run_pre_timeseries_steps=True, verbose=True):

    if verbose: 
        print('\n-------------------------------------------------------------------')
        print('RETRIEVING ISLAND INFORMATION')
        print('Island:', ', '.join([island, country]))
        print('-------------------------------------------------------------------\n')

    # If the path in which the data will be stored doesn't exist, we create it
    if not os.path.exists(island_info_path): 
        os.makedirs(island_info_path)

    # Check what information is already available
    file_island_info = os.path.join(island_info_path, 'info_{}_{}.data'.format(island, country))

    if os.path.isfile(file_island_info):
        # Load the .data file with pickle
        with open(file_island_info, 'rb') as f:
            island_info = pd.read_pickle(f)

        if verbose:
            print('~ The following information is available: ~\n')
            for info in island_info.keys(): 
                print(info)
                if type(island_info[info]) == dict:
                    for info_sd in island_info[info].keys(): 
                        print('              ', info_sd)
    
    # No file exists
    else:
        if run_pre_timeseries_steps:
            if verbose: 
                print('~ No file exists. Will run `pre_timeseries_steps.py`. ~\n')

            island_info = pre_timeseries_steps.PreTimeSeries(island, country).main()

        else:
            if verbose: 
                print('~ No file exists. ~\n')
    
    return island_info