"""
This module allows us to retrieve socioeconomics and environmental data from World Health Organization (WHO).
NOTE: Inspired by https://towardsdatascience.com/analyze-data-from-the-world-health-organization-global-health-observatory-723418d3642b
TODO: average data from the same year, split PM2.5 (different categories)

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import numpy as np
from mpetools import timeseries_socioeconomics_WorldBank, get_info_islands
import pandas as pd
import os
import requests
import pickle
import json
from tqdm import tqdm
import datetime

class TimeSeriesWHO:
    def __init__(self, island, country, verbose_init=True, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands'), WHO_path=os.path.join(os.getcwd(), 'data', 'WHO'), overwrite=False):
        self.island = island
        self.country = country
        self.verbose_init = verbose_init
        self.island_info_path = island_info_path
        self.WHO_path = WHO_path
        self.overwrite = overwrite

    def assign_metadata(self):

        # Add description of this database
        self.island_info['timeseries_WHO']['description'] = 'This module allows us to retrieve socioeconomics and environmental data from World Health Organization (WHO).'

        # Add source (paper or url)
        self.island_info['timeseries_WHO']['source'] = 'https://www.who.int/data'

    def get_indicators(self):

        file_indicators_WHO = os.path.join(self.WHO_path, 'indicators_df.xlsx')
        file_indicators_WHO_json = os.path.join(self.WHO_path, 'indicators.json')

        # Read DataFrame of indicators
        if os.path.exists(file_indicators_WHO):

            # Load Excel file
            indicators_df = pd.read_excel(file_indicators_WHO)
        
        # If DataFrame does not exist, generate it from .json file
        else:
            
            # Load json file
            indicators_json = json.load(open(file_indicators_WHO_json))['value']

            # Create array for codes and descriptions
            codes = np.array([indicators_json[code]['IndicatorCode'] for code in range(len(indicators_json))])
            dess = np.array([indicators_json[name]['IndicatorName'] for name in range(len(indicators_json))])

            # Create combined array and DataFrame
            indicators_arr = np.column_stack((codes, dess))
            indicators_df = pd.DataFrame(indicators_arr, columns=['Code', 'Name'])

            # Save DataFrame
            indicators_df.to_excel(file_indicators_WHO, index=None)
        
        # Save information in dictionary
        self.island_info['timeseries_WHO']['description_timeseries'] = indicators_df.set_index('Code').to_dict()['Name']
    
    def get_timeseries(self):

        print('~ Retrieving time series. ~')

        # Get Dataframe of indicators
        self.get_indicators()
        indicators_dict = self.island_info['timeseries_WHO']['description_timeseries']

        # Retrieve information from dictionary
        country_id = self.island_info['general_info']['country_ID']

        # Other relevant information
        headers = {'Content-type': 'application/json'}

        idx_WHO = 0
        # Loop in every indicator
        for indicator in tqdm(list(indicators_dict.keys())):

            # Request information from WHO database
            post = requests.post('https://ghoapi.azureedge.net/api/{}'.format(indicator), headers=headers)
            
            # See if the URL is readable
            try:  
                data = json.loads(post.text)
            except: 
                continue

            # Select values from dictionary
            data_list = data['value']

            # Make a DataFrame out of the data_list dictionary
            df_data_list = pd.DataFrame(data_list)

            # Select relevant columns and sort by year
            df_data_list_selection = df_data_list[(df_data_list.SpatialDim == country_id) & ((df_data_list.Dim1 == 'BTSX') | (df_data_list.Dim1 == 'TOTL'))].sort_values(['TimeDim'])

            # Check if data is available
            if df_data_list_selection.shape[0] > 0:

                dfs = df_data_list_selection[['TimeDim', 'NumericValue']].copy()
                try:
                    dfs['datetime'] = [datetime.datetime(year=dfs.TimeDim[idx], month=1, day=1) for idx in dfs.index]

                except:
                    continue
                
                dfs = dfs[['datetime', 'NumericValue']].set_index('datetime')
                dfs = dfs.rename(columns={'NumericValue': indicator})
                dfs_grouped = dfs.groupby('datetime').mean()
                
                if idx_WHO == 0:
                    dfs_t = dfs_grouped

                else:
                    dfs_t = pd.concat([dfs_t, dfs_grouped], axis=1)

                idx_WHO += 1
            
        # Drop NaN values (no data at all) and constant values
        dfs_t = dfs_t.dropna(axis=1, how='all')
        dfs_t = dfs_t.drop(columns=dfs_t.columns[dfs_t.nunique() == 1])
        dfs_t = dfs_t.apply(pd.to_numeric)

        # Save information in dictionary
        self.island_info['timeseries_WHO']['timeseries'] = dfs_t

    def main(self):

        # Retrieve the dictionary with currently available information about the island
        self.island_info = get_info_islands.retrieve_info_island(self.island, self.country, verbose=self.verbose_init)

        print('\n-------------------------------------------------------------------')
        print('RETRIEVING World Health Organization (WHO) DATA')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')
        
        # Retrieve Country ID
        if not 'country_ID' in self.island_info['general_info'].keys():

            print('~ Retrieving country ID. ~\n')

            # Retrieve country ID (for example, Nauru = NRU)
            self.island_info = timeseries_socioeconomics_WorldBank.TimeSeriesWorldBank(self.island, self.country).main()

        # If WHO data have NOT already been generated
        if not 'timeseries_WHO' in self.island_info.keys() or self.overwrite:

            # Create key/dict for WHO data
            self.island_info['timeseries_WHO'] = {}
            self.assign_metadata()

            # Since WHO data is available for a whole country (and not a specific island), check if data has already been extracted for that country
            if np.shape(np.argwhere(np.array([self.country in listt for listt in os.listdir(self.island_info_path)])))[0] > 1:

                print("~ Information available for this country. ~")

                # Retrieve dictionary for another island of that country
                array_listdir = np.array(os.listdir(self.island_info_path))
                array_listdir = np.delete(array_listdir, np.argwhere(array_listdir == 'info_{}_{}.data'.format(self.island, self.country)))
                idx_listdir = np.argwhere(np.array([self.country in listt for listt in array_listdir]))

                with open(os.path.join(self.island_info_path, str(array_listdir[idx_listdir[0]][-1])), 'rb') as fw:
                    island_info_other_island = pickle.load(fw)    
                
                if 'timeseries_WHO' in list(island_info_other_island.keys()):

                    # Fill information with other island
                    self.island_info['timeseries_WHO']['timeseries'] = island_info_other_island['timeseries_WHO']['timeseries']

                else:
                    self.get_timeseries()  

            else:
                self.get_timeseries()

        # If WHO data have already been generated
        else:

            print('~ Information already available. Returning data. ~')

        # Save dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)
        
        return self.island_info