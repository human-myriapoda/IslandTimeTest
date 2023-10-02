"""
This module allows to retrieve socioeconomics and environmental data from World Health Organization (WHO)
NOTE: Inspired by https://towardsdatascience.com/analyze-data-from-the-world-health-organization-global-health-observatory-723418d3642b
TODO: average data from the same year, split PM2.5 (different categories)

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import numpy as np
from mpetools import OLD_timeseries_socioeconomics_WorldBank, get_info_islands
import pandas as pd
import os
import requests
import pickle
import json
from tqdm import tqdm

def getIndicators(path_WHO=os.getcwd()+'\\data\\WHO'):

    # Read DataFrame of indicators
    if os.path.exists(path_WHO + '\\indicator_df.xlsx'):

        # Load Excel file
        df_indicator = pd.read_excel(path_WHO + '\\indicator_df.xlsx')
    
    # If DataFrame does not exist, generate it from .json file
    else:
        
        # Load json file
        indicator_json = json.load(open(os.getcwd() + '\\data\\WHO\\Indicator.json'))['value']

        # Create array for codes and descriptions
        codes = np.array([indicator_json[code]['IndicatorCode'] for code in range(len(indicator_json))])
        dess = np.array([indicator_json[name]['IndicatorName'] for name in range(len(indicator_json))])

        # Create combined array and DataFrame
        arr_indicator = np.column_stack((codes, dess))
        df_indicator = pd.DataFrame(arr_indicator, columns=['Code', 'Name'])

        # Save DataFrame
        df_indicator.to_excel(os.getcwd()+'\\data\\WHO\\indicator_df.xlsx', index=None)

    return df_indicator

def getTimeSeries(island_info):

    print('~ Retrieving time series. ~')

    # Get Dataframe of indicators
    df_indicator = getIndicators()

    # Retrieve information from dictionary
    country_ID = island_info['general_info']['country_ID']

    # Other relevant information
    headers = {'Content-type': 'application/json'}

    # Loop in every indicator
    for idx, indicator in enumerate(tqdm(df_indicator.Code)):

        # Request information from WHO database
        post = requests.post('https://ghoapi.azureedge.net/api/{}'.format(indicator), headers=headers)
        
        # See if the URL is readable
        try:  data = json.loads(post.text)
        except: continue

        # Select values from dictionary
        data_list = data['value']

        # Make a DataFrame out of the data_list dictionary
        df_data_list = pd.DataFrame(data_list)

        # Select relevant columns and sort by year
        df_data_list_selection = df_data_list[(df_data_list.SpatialDim == country_ID) & ((df_data_list.Dim1 == 'BTSX') | (df_data_list.Dim1 == 'TOTL'))].sort_values(['TimeDim'])

        # Check if data is available
        if df_data_list_selection.shape[0] > 0:

            timedim_arr = np.array(df_data_list_selection.TimeDim)
            numericvalues_arr = np.array(df_data_list_selection.NumericValue)
            
            # Cleaning data (no time series, constant values, etc.)
            if len(numericvalues_arr) == 1 or \
            all(x == timedim_arr[0] for x in timedim_arr) or \
            all(x == numericvalues_arr[0] for x in numericvalues_arr) or \
            all(np.isnan(numericvalues_arr)): continue          
            
            else:

                # Create NumPy arrays with only TimeDim (year) and NumericValue
                arr_info_WHO = np.array([indicator, df_indicator.Name[idx], timedim_arr, numericvalues_arr], dtype=object)

                # First iteration
                try: arr_info_WHO_total = np.row_stack((arr_info_WHO_total, arr_info_WHO))

                # Other iterations
                except: arr_info_WHO_total = arr_info_WHO

    # Save information in dictionary
    island_info['WHO']['DataFrame'] = pd.DataFrame(arr_info_WHO_total, columns=['Code', 'Name', 'DateRange', 'TimeSeries'])

    return island_info

def getWHOData(island, country, verbose_init=True, island_info_path=os.getcwd()+'\\data\\info_islands'):

    # Retrieve the dictionary with currently available information about the island.
    island_info = get_info_islands.retrieveInfoIslands(island, country, verbose=verbose_init)

    print('\n-------------------------------------------------------------------')
    print('RETRIEVING World Health Organization (WHO) DATA')
    print('Island:', ', '.join([island, country]))
    print('-------------------------------------------------------------------\n')
    
    # Country ID
    if not 'country_ID' in island_info['general_info'].keys():

        print('~ Retrieving country ID. ~\n')

        # Retrieve country ID (for example, Nauru = NRU)
        island_info = OLD_timeseries_socioeconomics_WorldBank.findCountryID(island_info)

    # If WHO data have NOT already been generated
    if not 'WHO' in island_info.keys():

        # Create key/dict for WHO data
        island_info['WHO'] = {}

        # Since WHO data is available for a whole country (and not a specific island), check if data has already been extracted for that country
        if np.shape(np.argwhere(np.array([country in listt for listt in os.listdir(island_info_path)])))[0] > 1:

            # Retrieve dictionary for another island of that country
            array_listdir = np.array(os.listdir(island_info_path))
            array_listdir = np.delete(array_listdir, np.argwhere(array_listdir == 'info_{}_{}.data'.format(island_info['general_info']['island'], island_info['general_info']['country'])))
            idx_listdir = np.argwhere(np.array([country in listt for listt in array_listdir]))
            fw = open(island_info_path + '\\{}'.format(array_listdir[idx_listdir[0]][0]), 'rb')
            island_info_other_island = pickle.load(fw)    
            fw.close() 

            # Fill information with other island
            if 'WHO' in list(island_info_other_island.keys()):

                island_info['WHO']['DataFrame'] = island_info_other_island['WHO']['DataFrame']
            
            else:

                # Run all functions
                island_info = getTimeSeries(island_info)  
        else:

            # Retrieve time series
            island_info = getTimeSeries(island_info)
    
    # If WHO data have already been generated
    else:
            
        print('~ Information already available. Returning data. ~')

    # Save dictionary
    fw = open(island_info_path + '\\info_{}_{}.data'.format(island_info['general_info']['island'], island_info['general_info']['country']), 'wb')
    pickle.dump(island_info, fw)
    fw.close()

    return island_info