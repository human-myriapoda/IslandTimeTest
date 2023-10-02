"""
This module retrieves time series of sea surface temperature (SST) from NOAA Coral Reef Watch (CRW) for a given region.
Date range: from 1985
Link: https://coralreefwatch.noaa.gov/product/vs/data.php

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
from mpetools import get_info_islands
import os
import pickle
import pandas as pd
import re
import urllib.request
import numpy as np
import datetime

class TimeSeriesNOAACRW:
    def __init__(self, island, country, verbose_init=True, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands'), overwrite=False):
        self.island = island
        self.country = country
        self.verbose_init = verbose_init
        self.island_info_path = island_info_path
        self.overwrite = overwrite 

    def assign_metadata(self):

        # Add description of this database
        self.island_info['timeseries_CRW']['description'] = 'This module retrieves time series of sea surface temperature (SST) from NOAA Coral Reef Watch (CRW) for a given region.'
        
        # Add description of this timeseries
        self.island_info['timeseries_CRW']['description_timeseries'] = 'Daily mean SST for a 5km virtual region.'

        # Add source (paper or url)
        self.island_info['timeseries_CRW']['source'] = 'https://coralreefwatch.noaa.gov/product/vs/data.php'

    def get_available_stations(self):

        # URL with all >200 stations
        url_CRW = 'https://coralreefwatch.noaa.gov/product/vs/data.php'
        web_CRW = urllib.request.urlopen(url_CRW)
        data_CRW = str(web_CRW.read())

        # Regex pattern for station names
        pattern_CRW = r'_ts_(\w+).png"\>'

        # Find all corresponding patterns
        station_names = re.findall(pattern_CRW, data_CRW)

        # Clean data
        station_names = np.unique(np.array(station_names))
        station_names = station_names[np.argwhere(~np.char.startswith(station_names, 'multiyr')).T[0]]

        # Keep data stored
        self.station_names = station_names
    
    def find_corresponding_station(self):

        possible_patterns = [self.island.lower().replace(' ', '_'), self.country.lower(), '_'.join((self.island.lower(), self.country.lower()))]
        self.station_to_retrieve = False

        for pp in possible_patterns:
            if pp in self.station_names:
                self.station_name = self.station_names[pp == self.station_names][0]
                self.station_to_retrieve = True

    def get_timeseries(self):

        # Retrieve available stations
        self.get_available_stations()

        # Find station to retrieve
        self.find_corresponding_station()

        # If there is a station available
        if self.station_to_retrieve:
            print('~ The station to retrieve is `{}`. ~'.format(self.station_name))

            # Retrieve corresponding URL
            url_station = 'https://coralreefwatch.noaa.gov/product/vs/data/{}.txt'.format(self.station_name)
            web_station = urllib.request.urlopen(url_station)
            data_station = str(web_station.read())

            # Select relevant information
            station_arr_splitted = np.array(data_station.split('\\n'))
            idx_start = np.argwhere(np.char.startswith(station_arr_splitted, 'YYYY'))[0][0]
            station_arr = station_arr_splitted[idx_start:-1]

            # Clean data
            for idx_station in range(1, len(station_arr)): 
                if idx_station == 1:
                    station_arr_cleaned = np.array([item for item in station_arr[idx_station].split(' ') if item != ''], dtype=float)
                else:
                    station_arr_cleaned = np.row_stack((station_arr_cleaned, np.array([item for item in station_arr[idx_station].split(' ') if item != ''], dtype=float)))

            # Create DataFrame
            station_df = pd.DataFrame(station_arr_cleaned, columns = np.array(station_arr[0].split(' '), dtype=str))
            station_df['datetime'] = [datetime.datetime(year=int(station_df.YYYY[idx]), month=int(station_df.MM[idx]), day=int(station_df.DD[idx])) for idx in station_df.index]
            station_df['mean_SST'] = [(station_df.SST_MIN[idx] + station_df.SST_MAX[idx])/2 for idx in station_df.index]
            station_df_final = station_df[['datetime', 'mean_SST']]
            station_df_final = station_df_final.set_index('datetime')
            station_df_final = station_df_final.apply(pd.to_numeric)

            # Save information in dictionary
            self.island_info['timeseries_CRW']['timeseries'] = station_df_final
        
        # If there is no station available
        else:
            print('~ There is no station available for this island. ~')

    def main(self):

        # Retrieve the dictionary with currently available information about the island
        self.island_info = get_info_islands.retrieve_info_island(self.island, self.country, verbose=self.verbose_init)

        print('\n-------------------------------------------------------------------')
        print('RETRIEVING SEA SURFACE TEMPERATURE (NOAA Coral Reef Watch) DATA')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        # If CRW data have NOT already been generated
        if not 'timeseries_CRW' in self.island_info.keys() or self.overwrite:

            # Create key/dict for CRW data
            self.island_info['timeseries_CRW'] = {}
            self.assign_metadata()

            # Run all functions
            self.get_timeseries()
        
        # If CRW data have already been generated
        else:
            print('~ Information already available. Returning data. ~')

        # Save dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)
        
        return self.island_info



