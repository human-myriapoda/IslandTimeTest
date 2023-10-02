"""
This module allows us to retrieve disaster data from EM-DAT.
Excel files have to be downloaded from https://public.emdat.be/data
TODO: add other data sources
TODO: add metadata
TODO: ADD DISASTER PERIODS

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import numpy as np
import pandas as pd
import pickle
import os
import datetime
from datetime import timedelta
from mpetools import get_info_islands

class TimeSeriesDisasters:
    def __init__(self, island, country, verbose_init=True, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands'), disasters_path=os.path.join(os.getcwd(), 'data', 'disasters'), overwrite=False):
        self.island = island
        self.country = country
        self.verbose_init = verbose_init
        self.island_info_path = island_info_path
        self.disasters_path = disasters_path
        self.overwrite = overwrite

    def assign_metadata(self):
        # Add description of this database
        self.island_info['timeseries_disasters']['description'] = 'The International Disaster Database, also called Emergency Events Database (EM-DAT), contains essential core data on the occurrence and effects of over 22,000 mass disasters in the world from 1900 to the present day. The database is compiled from various sources, including UN agencies, non-governmental organisations, insurance companies, research institutes and press agencies.'
        
        # Add description of this timeseries
        self.island_info['timeseries_disasters']['description_timeseries'] = 'timeseries_natural -> number of natural disasters per year | timeseries_technological -> number of technological disasters per year | timeseries_natural_cumulative -> cumulative number of natural disasters | timeseries_technological_cumulative -> cumulative number of technological disasters'

        # Add source (paper or url)
        self.island_info['timeseries_disasters']['source'] = 'https://www.emdat.be/'
    
    def get_timeseries(self):

        print('~ Retrieving disaster data from EM-DAT database. ~')

        # Retrieve data from Excel file
        disasters_data = pd.read_excel(os.path.join(self.disasters_path, 'disasters_database_1975_2023.xlsx'))

        # Get DataFrame for the island
        disasters_data_island = disasters_data[disasters_data.Country == self.country]

        # Save information in dictionary
        self.island_info['timeseries_disasters']['database'] = disasters_data_island

        #### TIME SERIES NUMBER OF EVENTS PER YEAR ####

        print('~ Retrieving time series as number of events per year. ~')

        # Create column for `datetime`
        arr_datetime = np.array([datetime.datetime(year=year, month=1, day=1) \
                        for year in range(disasters_data_island.Year.values[0], \
                                          disasters_data_island.Year.values[-1] + 1)])
        
        # Create empty columns for `natural` and `technological`
        arr_natural = np.zeros_like(arr_datetime)
        arr_technological = np.zeros_like(arr_datetime)

        # Fill number of events per year
        for idx in disasters_data_island.index:
            if disasters_data_island['Disaster Group'][idx] == "Natural": 
                arr_natural[np.argwhere(arr_datetime == datetime.datetime(year=disasters_data_island['Year'][idx], month=1, day=1))[0][0]] += 1

            elif disasters_data_island['Disaster Group'][idx] == "Technological":
                arr_technological[np.argwhere(arr_datetime == datetime.datetime(year=disasters_data_island['Year'][idx], month=1, day=1))[0][0]] += 1

        # Calculate cumulative sum
        arr_natural_cumulative = np.cumsum(arr_natural)
        arr_technological_cumulative = np.cumsum(arr_technological)

        # Create multiple pd.DataFrame
        timeseries_disasters = pd.DataFrame(np.column_stack((arr_datetime, arr_natural, arr_technological, arr_natural_cumulative, arr_technological_cumulative)), \
                                          columns=['datetime', 'natural', 'technological', 'natural_cumulative', 'technological_cumulative'])
        timeseries_disasters = timeseries_disasters.set_index('datetime')
        timeseries_disasters = timeseries_disasters.apply(pd.to_numeric)

        # Save information in dictionary
        self.island_info['timeseries_disasters']['timeseries'] = timeseries_disasters

        #### TIME SERIES CONFOUNDERS ####

        print('~ Retrieving time series as confounder effects. ~')

        # Create a daily range of dates
        def generate_daily_range(start_year, end_year):
            start_date = datetime.datetime(start_year, 1, 1)
            end_date = datetime.datetime(end_year + 1, 1, 1)  # Add one year to include the end year

            current_date = start_date
            while current_date < end_date:
                yield current_date
                current_date += timedelta(days=1)  # Increment by one day

        daily_range_datetime = list(generate_daily_range(1975, datetime.datetime.now().year)) 

        # Columns names for DataFrame
        col_names = np.array([disasters_data_island.loc[idx]['Disaster Type'].replace(' ', '_').lower() + '_' + str(disasters_data_island.loc[idx]['Year']) for idx in disasters_data_island.index], dtype=str)

        # List of geographical areas
        list_geo_locations = ', '.join([self.island_info['general_info'][key_info] for key_info in self.island_info['general_info'].keys()]).split(', ')

        # Create empty dictionary
        dict_disasters_confounders = {'datetime': daily_range_datetime}

        # Create DataFrame
        df_confounders = pd.DataFrame(dict_disasters_confounders).set_index('datetime')

        # Loop through catastrophes
        for it, idx in enumerate(disasters_data_island.index):

            row_event = disasters_data_island.loc[idx]

            if not pd.isnull(row_event['Location']) or not pd.isnull(row_event['Geo Locations']):
                list_location = [list_geo_locations[i] in [row_event['Location'].split(', ') if not pd.isnull(row_event['Location']) else []][0] for i in range(len(list_geo_locations))]
                list_geo_locations = [list_geo_locations[i] in [row_event['Geo Locations'].split(', ') if not pd.isnull(row_event['Geo Locations']) else []][0] for i in range(len(list_geo_locations))]
                list_t = np.array(list_location + list_geo_locations)

                if row_event['Location'] == 'Widespread': 
                    list_t = [True]

            else: 
                list_t = [True]

            # If the event relates to the island
            if np.any(list_t): 
                start_event = datetime.datetime(year=int(row_event['Start Year']), month=int([row_event['End Month'] if not np.isnan(row_event['End Month']) else 1][0]), day=int([row_event['Start Day'] if not np.isnan(row_event['Start Day']) else 1][0]))
                end_event = datetime.datetime(year=int(row_event['End Year']), month=int([row_event['End Month'] if not np.isnan(row_event['End Month']) else 1][0]), day=int([row_event['End Day'] if not np.isnan(row_event['End Day']) else 1][0]))
                df_confounders[col_names[it]] = np.zeros(len(daily_range_datetime))
                df_confounders[col_names[it]][(df_confounders.index >= start_event) & (df_confounders.index <= end_event)] = 1

        # Save information in dictionary
        self.island_info['timeseries_disasters']['confounders'] = df_confounders


    def main(self):
        # Retrieve the dictionary with currently available information about the island
        self.island_info = get_info_islands.retrieve_info_island(self.island, self.country, verbose=self.verbose_init)

        print('\n-------------------------------------------------------------------')
        print('RETRIEVING DISASTERS (EM-DAT) DATA')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        # If Disasters data have NOT already been generated
        if not 'timeseries_disasters' in self.island_info.keys() or self.overwrite:
            # Create key/dict for Disasters data
            self.island_info['timeseries_disasters'] = {}
            self.assign_metadata()
            self.get_timeseries()
        
        # If Disasters data have already been generated
        else:
            print('~ Information already available. Returning data. ~')

        # Save dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)

        return self.island_info
