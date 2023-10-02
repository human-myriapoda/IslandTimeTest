"""
This module allows to retrieve data from the Pacific Sea Level and Geodetic Monitoring Project (Monthly Sea Level and Meteorological Statistics).
Available information: Monthly sea level, barometric pressure, water temperature and air temperature.
Source: http://www.bom.gov.au/oceanography/projects/spslcmp/data/monthly.shtml
TODO: skip if data is not available, better variable names

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import pandas as pd
import pickle
import os
from mpetools import get_info_islands
import numpy as np

def retrievePSLGM(island_info, path_to_PSLGM=os.getcwd()+'\\data\\PSLGM'):

    for info in ['sea_level', 'barometric_pressure', 'water_temperature', 'air_temperature']:

        try:
            # Read data from file
            data_PSLGM = np.array(open(path_to_PSLGM + '\\{}_{}_{}.txt'.format(island_info['general_info']['island'], \
                                                                    island_info['general_info']['country'], \
                                                                    info), 'r').readlines())
        except: 

            print('No PSLGM information for this island.')
            
            return island_info

        print("~ Retrieving {}. ~".format(info.replace('_', ' ').capitalize()))

        # Select rows corresponding to the data
        data_cleaned = np.array(data_PSLGM[np.argwhere(np.char.startswith(data_PSLGM, '      Mth'))[0][0] + 1 : \
                                    np.argwhere(np.char.startswith(data_PSLGM, '      Totals'))[0][0]])

        for row in range(len(data_cleaned)):

            row_cleaned = np.array(data_cleaned[row].replace('\n', '').split(' '))
            row_fully_cleaned = row_cleaned[row_cleaned != ''].astype(float)

            # Expected length is 8 (otherwise there is missing data)
            if len(row_fully_cleaned) != 8: continue

            # First iteration
            if row == 0: full_data = row_fully_cleaned
            
            # Stack with existing data
            else: full_data = np.row_stack((full_data, row_fully_cleaned))
    
        # Create pandas.DataFrame
        df_PSLGM = pd.DataFrame(full_data, columns=['Month', 'Year', 'Gaps', 'Good', 'Minimum', 'Maximum', 'Mean', 'StD'])

        # Save information in dictionary
        island_info['PSLGM'][info] = df_PSLGM
        
    return island_info

def getPSLGM(island, country, verbose_init=True, island_info_path=os.getcwd()+'\\data\\info_islands'):

    # Retrieve the dictionary with currently available information about the island.
    island_info = get_info_islands.retrieveInfoIslands(island, country, verbose=verbose_init)

    print('\n-------------------------------------------------------------------')
    print('RETRIEVING TIME SERIES FOR SEA LEVELS, BAROMETRIC PRESSURES, WATER TEMPERATURES AND AIR TEMPERTATURES (PSLGM) DATA')
    print('Island:', ', '.join([island, country]))
    print('-------------------------------------------------------------------\n')

    # If PSLGM data have NOT already been generated
    if not 'PSLGM' in island_info.keys():

        # Create key/dict for OpenStreetMap data
        island_info['PSLGM'] = {}

        # Run all functions
        island_info = retrievePSLGM(island_info)
    
    # If PSLGM data have already been generated
    else:

        print('~ Information already available. Returning data. ~')

    # Save dictionary
    fw = open(island_info_path + '\\info_{}_{}.data'.format(island_info['general_info']['island'], island_info['general_info']['country']), 'wb')
    pickle.dump(island_info, fw)
    fw.close()

    return island_info