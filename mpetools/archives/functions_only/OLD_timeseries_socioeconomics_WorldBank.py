"""
This module allows to retrieve socioeconomics data from World Bank Database (WBD).
TODO: manage date range
TODO: fit format with other time series

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

class TimeSeriesWorldBank:
    def __init__(self, island, country, date_range, verbose_init=True, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands')):
        self.island = island
        self.country = country
        self.date_range = date_range
        self.verbose_init = verbose_init
        self.island_info_path = island_info_path

    def assign_metadata(self):

        # Add description of this database
        self.island_info['timeseries_WorldBank']['description'] = 'Free and open access to global development data.'
        
        # Add description of this timeseries
        self.island_info['timeseries_WorldBank']['description_timeseries'] = 'TODO'

        # Add source (paper or url)
        self.island_info['timeseries_WorldBank']['source'] = 'https://data.worldbank.org/'

    def find_country_id(self):

        # Query World Bank (economy section)
        wb_featureset = wb.economy.info(q=self.country)

        # Find ID in FeatureSet
        country_id = wb_featureset.items[0]['id']

        # Save in dictionary
        self.island_info['general_info']['country_ID'] = country_id

    def get_timeseries(self):

        print('~ Retrieving time series. ~\n')

        # Retrieve date range and country ID
        date_range = self.island_info['World Bank']['date_range']
        country_ID = self.island_info['general_info']['country_ID']

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
            if np.all(np.isnan(time_series)): 
                continue

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
        self.island_info['timeseries_WorldBank']['DataFrame'] = data_df
        self.island_info['timeseries_WorldBank']['DataFrame_no_missing_values'] = data_df_nomissingvalues

    def main(self):

        # Retrieve the dictionary with currently available information about the island.
        self.island_info = get_info_islands.retrieveInfoIslands(self.island, self.country, verbose=self.verbose_init)

        print('\n-------------------------------------------------------------------')
        print('RETRIEVING WORLD BANK DATA')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')
        
        # Retrieve Country ID
        if not 'country_ID' in self.island_info['general_info'].keys():

            print('~ Retrieving country ID. ~\n')

            # Retrieve country ID (for example, Nauru = NRU)
            self.find_country_id()

        # If World Bank data have NOT already been generated
        if not 'timeseries_WorldBank' in self.island_info.keys():

            # Create key/dict for World Bank data
            self.island_info['timeseries_WorldBank'] = {}
            self.assign_metadata()

            # Set date range
            self.island_info['timeseries_WorldBank']['date_range'] = np.arange(self.date_range[0], self.date_range[1] + 1)

            # Since World Bank data is available for a whole country (and not a specific island), check if data has already been extracted for that country
            if np.shape(np.argwhere(np.array([self.country in listt for listt in os.listdir(self.island_info_path)])))[0] > 1:

                print("~ Information available for this country. ~")

                # Retrieve dictionary for another island of that country
                array_listdir = np.array(os.listdir(self.island_info_path))
                array_listdir = np.delete(array_listdir, np.argwhere(array_listdir == 'info_{}_{}.data'.format(self.island, self.country)))
                idx_listdir = np.argwhere(np.array([self.country in listt for listt in array_listdir]))

                with open(os.path.join(self.island_info_path, str(array_listdir[idx_listdir[0]][0])), 'rb') as fw:
                    island_info_other_island = pickle.load(fw)    

                # Fill information with other island
                if 'timeseries_WorldBank' in list(island_info_other_island.keys()):

                    # Check if date range matches the existing date range
                    if not np.array_equal(island_info_other_island['timeseries_WorldBank']['date_range'], self.island_info['timeseries_WorldBank']['date_range']):

                        print("Date range for that other island does not match with the existing date range! Generating data for the new date range.")

                        # Retrieve time series
                        self.get_timeseries()
                    
                    # Fill information with other island
                    # TODO: FIX
                    else:
                        self.island_info['timeseries_WorldBank']['DataFrame'] = island_info_other_island['timeseries_WorldBank']['DataFrame']
                        self.island_info['timeseries_WorldBank']['DataFrame_no_missing_values'] = island_info_other_island['timeseries_WorldBank']['DataFrame_no_missing_values']

                else:
                    # Run all functions
                    self.get_timeseries()  

            else:
                # Retrieve time series
                self.get_timeseries()

        # If World Bank data have already been generated
        else:

            # Check if date range matches the existing date range
            if not np.array_equal(self.island_info['timeseries_WorldBank']['date_range'], np.arange(self.date_range[0], self.date_range[1] + 1)):
                
                print("Date range does not match with the existing date range! Generating data for the new date range.")
        
                # Set date range
                self.island_info['timeseries_WorldBank']['date_range'] = np.arange(self.date_range[0], self.date_range[1] + 1)

                # Retrieve time series
                self.get_timeseries()  
            
            else:                
                print('~ Date range matches with the existing date range. Returning data. ~')

        # Save dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)
        
        return self.island_info
