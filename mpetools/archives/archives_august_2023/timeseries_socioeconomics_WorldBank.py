"""
This module allows us to retrieve socioeconomics data from World Bank Database (WBD).
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
import datetime

class TimeSeriesWorldBank:
    def __init__(self, island, country, verbose_init=True, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands'), overwrite=False):
        self.island = island
        self.country = country
        self.verbose_init = verbose_init
        self.island_info_path = island_info_path
        self.overwrite = overwrite

    def assign_metadata(self):

        # Add description of this database
        self.island_info['timeseries_WorldBank']['description'] = 'Free and open access to global development data.'

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

        # Retrieve country ID
        country_ID = self.island_info['general_info']['country_ID']

        # Query World Bank Database (series -> actual data)
        series_info = wb.series.info()

        # Create a pandas.DataFrame with all the information available to retrieve
        df_series_info = pd.DataFrame(vars(series_info).get('items'))

        # Add description of this timeseries
        self.island_info['timeseries_WorldBank']['description_timeseries'] = df_series_info.set_index('id').to_dict()['value']

        # Retrieve DataFrame for this country
        df_WBD = wb.data.DataFrame(list(df_series_info.id), country_ID).T

        # Manage datetime and indexes
        df_WBD['datetime'] = [datetime.datetime(year=int(df_WBD.index[idx].replace("YR", "")), month=1, day=1) for idx in range(len(df_WBD.index))]
        df_WBD = df_WBD.set_index('datetime')

        # Drop NaN values (no data at all) and constant values
        df_WBD = df_WBD.dropna(axis=1, how='all')
        df_WBD = df_WBD.drop(columns=df_WBD.columns[df_WBD.nunique() == 1])
        df_WBD = df_WBD.apply(pd.to_numeric)

        # Save information in dictionary
        self.island_info['timeseries_WorldBank']['timeseries'] = df_WBD

    def main(self):

        # Retrieve the dictionary with currently available information about the island
        self.island_info = get_info_islands.retrieve_info_island(self.island, self.country, verbose=self.verbose_init)

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
        if not 'timeseries_WorldBank' in self.island_info.keys() or self.overwrite:

            # Create key/dict for World Bank data
            self.island_info['timeseries_WorldBank'] = {}
            self.assign_metadata()

            # Since World Bank data is available for a whole country (and not a specific island), check if data has already been extracted for that country
            if np.shape(np.argwhere(np.array([self.country in listt for listt in os.listdir(self.island_info_path)])))[0] > 1:

                print("~ Information available for this country. ~")

                # Retrieve dictionary for another island of that country
                array_listdir = np.array(os.listdir(self.island_info_path))
                array_listdir = np.delete(array_listdir, np.argwhere(array_listdir == 'info_{}_{}.data'.format(self.island, self.country)))
                idx_listdir = np.argwhere(np.array([self.country in listt for listt in array_listdir]))

                with open(os.path.join(self.island_info_path, str(array_listdir[idx_listdir[0]][-1])), 'rb') as fw:
                    island_info_other_island = pickle.load(fw)    
                
                if 'timeseries_WorldBank' in list(island_info_other_island.keys()):

                    # Fill information with other island
                    self.island_info['timeseries_WorldBank']['timeseries'] = island_info_other_island['timeseries_WorldBank']['timeseries']

                else:
                    self.get_timeseries()  

            else:
                self.get_timeseries()

        # If World Bank data have already been generated
        else:

            print('~ Information already available. Returning data. ~')

        # Save dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)
        
        return self.island_info
