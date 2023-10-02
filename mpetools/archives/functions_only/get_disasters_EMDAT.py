"""
This module allows to retrieve disaster data from EM-DAT.
Excel files have to be downloaded from https://public.emdat.be/data
TODO: add other data sources

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import numpy as np
import pandas as pd
import pickle
import os
from mpetools import get_info_islands

def retrieveDisasters(island_info):

    print('~ Retrieving disaster data from EM-DAT database. ~')

    # Define path for data
    path_disasters = os.getcwd() + '\\data\\disasters'

    # Retrieve data from Excel file
    disasters_data = pd.read_excel(path_disasters+'\\disasters_database_1975_2023.xlsx')

    # Get DataFrame for given island
    disasters_data_island = disasters_data[disasters_data.Country == island_info['general_info']['country']]

    # Save information in dictionary
    island_info['Disasters']['DataFrame'] = disasters_data_island

    return island_info

def getDisasters(island, country, verbose_init=True, island_info_path=os.getcwd()+'\\data\\info_islands'):

    # Retrieve the dictionary with currently available information about the island.
    island_info = get_info_islands.retrieveInfoIslands(island, country, verbose=verbose_init)

    print('\n-------------------------------------------------------------------')
    print('RETRIEVING DISASTERS (EM-DAT) DATA')
    print('Island:', ', '.join([island, country]))
    print('-------------------------------------------------------------------\n')

    # If Disasters data have NOT already been generated
    if not 'Disasters' in island_info.keys():

        # Create key/dict for Disasters data
        island_info['Disasters'] = {}

        # Since `Disasters` data is available for a whole country (and not a specific island), check if data has already been extracted for that country
        if np.shape(np.argwhere(np.array([country in listt for listt in os.listdir(island_info_path)])))[0] > 1:

            # Retrieve dictionary for another island of that country
            array_listdir = np.array(os.listdir(island_info_path))
            array_listdir = np.delete(array_listdir, np.argwhere(array_listdir == 'info_{}_{}.data'.format(island_info['general_info']['island'], island_info['general_info']['country'])))
            idx_listdir = np.argwhere(np.array([country in listt for listt in array_listdir]))
            fw = open(island_info_path + '\\{}'.format(array_listdir[idx_listdir[0]][0]), 'rb')
            island_info_other_island = pickle.load(fw)    
            fw.close() 

            # Fill information with other island
            if 'Disasters' in list(island_info_other_island.keys()):

                island_info['Disasters']['DataFrame'] = island_info_other_island['Disasters']['DataFrame']
            
            else:

                # Run all functions
                island_info = retrieveDisasters(island_info)                

        else:

            # Run all functions
            island_info = retrieveDisasters(island_info)
    
    # If Disasters data have already been generated
    else:

        print('~ Information already available. Returning data. ~')

    # Save dictionary
    fw = open(island_info_path + '\\info_{}_{}.data'.format(island_info['general_info']['island'], island_info['general_info']['country']), 'wb')
    pickle.dump(island_info, fw)
    fw.close()

    return island_info