"""
This module allows to retrieve socioeconomics data from World Bank Database (WBD).

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import wbgapi as wb
import numpy as np
from mpetools import get_info_islands
import pandas as pd
import os
import pickle
from tqdm import tqdm

def findCountryID(island_info):

    # Query World Bank (economy section)
    wb_featureset = wb.economy.info(q=island_info['general_info']['country'])

    # Find ID in FeatureSet
    id_country = wb_featureset.items[0]['id']

    # Save in dictionary
    island_info['general_info']['country_ID'] = id_country

    return island_info

def getTimeSeries(island_info):

    print('~ Retrieving time series. ~\n')

    # Retrieve date range and country ID
    date_range = island_info['World Bank']['date_range']
    country_ID = island_info['general_info']['country_ID']

    # Query World Bank Database (series -> actual data)
    series_info = wb.series.info()

    # Create a pandas.DataFrame with all the information available to retrieve
    df_series_info = pd.DataFrame(vars(series_info).get('items'))

    # Empty lists
    list_IDs, list_descriptions, list_timeseries = [], [], []
    list_IDs_nomissingvalues, list_descriptions_nomissingvalues, list_timeseries_nomissingvalues = [], [], []

    # Loop for all information to retrieve
    for idx in tqdm(range(len(df_series_info.id))):
        
        # Data from World Bank for given island and date range
        data_WB = wb.data.DataFrame(df_series_info.id[idx], country_ID, time=date_range)

        # Retrieve time series from data
        time_series = data_WB.loc[data_WB.index == country_ID].values[0]

        # If there is no information available for that index (NaN everywhere), skip to the next index
        if np.all(np.isnan(time_series)): continue

        else:

            # If the time series is full (i.e. no NaN) and is not constant (same value for the whole time series), save time series
            if not np.any(np.isnan(time_series)) and not all(x==time_series[0] for x in time_series):

                # Append lists (no missing values)
                list_IDs_nomissingvalues.append(df_series_info.id[idx])
                list_descriptions_nomissingvalues.append(df_series_info.value[idx])
                list_timeseries_nomissingvalues.append(time_series)
            
            # Append lists (including missing values)
            list_IDs.append(df_series_info.id[idx])
            list_descriptions.append(df_series_info.value[idx])
            list_timeseries.append(time_series)
    
    # Create arrays with IDs, descriptions and time series
    data_arr = np.array([list_IDs, list_descriptions, list_timeseries], dtype=object).T
    data_arr_nomissingvalues = np.array([list_IDs_nomissingvalues, list_descriptions_nomissingvalues, list_timeseries_nomissingvalues], dtype=object).T

    # Create DataFrames with IDs, descriptions and time series
    data_df = pd.DataFrame(data_arr, index=None, columns=['id', 'description', 'time_series'])
    data_df_nomissingvalues = pd.DataFrame(data_arr_nomissingvalues, index=None, columns=['id', 'description', 'time_series'])

    # Save information in dictionary
    island_info['World Bank']['DataFrame'] = data_df
    island_info['World Bank']['DataFrame_no_missing_values'] = data_df_nomissingvalues

    return island_info

def getWorldBankData(island, country, date_range: list, verbose_init=True, island_info_path=os.getcwd()+'\\data\\info_islands'):

    # Retrieve the dictionary with currently available information about the island.
    island_info = get_info_islands.retrieveInfoIslands(island, country, verbose=verbose_init)

    print('\n-------------------------------------------------------------------')
    print('RETRIEVING WORLD BANK DATA')
    print('Island:', ', '.join([island, country]))
    print('-------------------------------------------------------------------\n')
    
    # Country ID
    if not 'country_ID' in island_info['general_info'].keys():

        print('~ Retrieving country ID. ~\n')

        # Retrieve country ID (for example, Nauru = NRU)
        island_info = findCountryID(island_info)

    # If World Bank data have NOT already been generated
    if not 'World Bank' in island_info.keys():

        # Create key/dict for World Bank data
        island_info['World Bank'] = {}

        # Set date range
        island_info['World Bank']['date_range'] = np.arange(date_range[0], date_range[1] + 1)

        # Since World Bank data is available for a whole country (and not a specific island), check if data has already been extracted for that country
        if np.shape(np.argwhere(np.array([country in listt for listt in os.listdir(island_info_path)])))[0] > 1:
            
            print("~ Information available for this country. ~")

            # Retrieve dictionary for another island of that country
            array_listdir = np.array(os.listdir(island_info_path))
            array_listdir = np.delete(array_listdir, np.argwhere(array_listdir == 'info_{}_{}.data'.format(island_info['general_info']['island'], island_info['general_info']['country'])))
            idx_listdir = np.argwhere(np.array([country in listt for listt in array_listdir]))
            fw = open(island_info_path + '\\{}'.format(array_listdir[idx_listdir[0]][0]), 'rb')
            island_info_other_island = pickle.load(fw)    
            fw.close() 

            # Fill information with other island
            if 'World Bank' in list(island_info_other_island.keys()):

                # Check if date range matches the existing date range
                if not np.array_equal(island_info_other_island['World Bank']['date_range'], island_info['World Bank']['date_range']):

                    print("Date range for that other island does not match with the existing date range! Generating data for the new date range.")

                    # Retrieve time series
                    island_info = getTimeSeries(island_info)
                
                # Fill information with other island
                else:

                    island_info['World Bank']['DataFrame'] = island_info_other_island['World Bank']['DataFrame']
                    island_info['World Bank']['DataFrame_no_missing_values'] = island_info_other_island['World Bank']['DataFrame_no_missing_values']
            else:

                # Run all functions
                island_info = getTimeSeries(island_info)   
        else:

            # Retrieve time series
            island_info = getTimeSeries(island_info)

    # If World Bank data have already been generated
    else:

        # Check if date range matches the existing date range
        if not np.array_equal(island_info['World Bank']['date_range'], np.arange(date_range[0], date_range[1] + 1)):

            print("Date range does not match with the existing date range! Generating data for the new date range.")
    
            # Set date range
            island_info['World Bank']['date_range'] = np.arange(date_range[0], date_range[1] + 1)

            # Retrieve time series
            island_info = getTimeSeries(island_info)
        
        else:
            
            print('~ Date range matches with the existing date range. Returning data. ~')

    # Save dictionary
    fw = open(island_info_path + '\\info_{}_{}.data'.format(island_info['general_info']['island'], island_info['general_info']['country']), 'wb')
    pickle.dump(island_info, fw)
    fw.close()

    return island_info