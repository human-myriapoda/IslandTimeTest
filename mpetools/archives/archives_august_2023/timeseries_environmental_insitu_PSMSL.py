"""
This module allows us to retrieve tide-gauge sea-level data from the Permanent Service for Mean Sea Level (PSMSL).
Source: https://psmsl.org/data/obtaining/

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import pandas as pd
import pickle
import os
from mpetools import get_info_islands
import numpy as np
import datetime
import geopy.distance
import urllib.request

class TimeSeriesPSMSL:
    def __init__(self, island, country, number_of_stations=3, verbose_init=True, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands'), PSMSL_path=os.path.join(os.getcwd(), 'data', 'PSMSL'), overwrite=False):
        self.island = island
        self.country = country
        self.verbose_init = verbose_init
        self.island_info_path = island_info_path
        self.PSMSL_path = PSMSL_path
        self.number_of_stations = number_of_stations
        self.overwrite = overwrite

    def assign_metadata(self):

        # Add description of this database
        self.island_info['timeseries_PSMSL']['description'] = 'This module allows us to retrieve tide-gauge sea-level data from the Permanent Service for Mean Sea Level (PSMSL).'

        # Add source (paper or url)
        self.island_info['timeseries_PSMSL']['source'] = 'https://psmsl.org/data/obtaining/'

    def find_closest_stations(self):

        # Read Excel find with all available stations 
        stations_PSMSL = pd.read_excel(os.path.join(self.PSMSL_path, 'stations_PSMSL.xlsx'))

        # Create list of points with lat/lon of the island and of the station
        coords_island = (self.island_info['spatial_reference']['latitude'], self.island_info['spatial_reference']['longitude'])
        coords_stations = [(stations_PSMSL.lat.values[idx], stations_PSMSL.lon.values[idx]) for idx in stations_PSMSL.index]

        # Calculate distance (in km) between the island and every station
        distance_island_stations = np.array([geopy.distance.geodesic(coords_island, coords_stations[idx]).km for idx in range(len(coords_stations))])

        # Sort indices and keep the first three stations
        closest_stations_idx = np.argsort(distance_island_stations)[:self.number_of_stations]
        closest_stations_PSMSL = stations_PSMSL.iloc[closest_stations_idx]
        closest_distance_island_stations = distance_island_stations[closest_stations_idx]

        # Clean DataFrame
        closest_stations_PSMSL = closest_stations_PSMSL.set_index('ID')[['Station_name', 'lat', 'lon', 'Country']].rename(columns={'Station_name': 'station_name', 'lat': 'station_latitude', 'lon': 'station_longitude', 'Country': 'country_ID'})
        closest_stations_PSMSL['distance_from_island'] = closest_distance_island_stations

        # Save for other functions
        self.closest_stations_PSMSL = closest_stations_PSMSL

        # Add description of this timeseries
        self.island_info['timeseries_PSMSL']['description_timeseries'] = closest_stations_PSMSL

    def get_timeseries(self):
        
        # Find closest stations
        self.find_closest_stations()

        # String for order of stations
        order_stations = ['{}{}'.format(i+1, 'th' if i+1 > 3 else ['st', 'nd', 'rd'][i]) for i in range(self.number_of_stations)]

        for idx_PSMSL, station in enumerate(self.closest_stations_PSMSL.index):
            
            print('~ Retrieving data from the {} closest station:'.format(order_stations[idx_PSMSL]), \
                    self.closest_stations_PSMSL.loc[self.closest_stations_PSMSL.index[idx_PSMSL]]['station_name'], \
                    'at a distance of', np.round(self.closest_stations_PSMSL.loc[self.closest_stations_PSMSL.index[idx_PSMSL]]['distance_from_island'], 1), \
                    'km from the island. ~')

            # Retrieve data from corresponding URL
            url_station = 'https://psmsl.org/data/obtaining/rlr.monthly.data/{}.rlrdata'.format(station)
            web_station = urllib.request.urlopen(url_station)
            data_station = str(web_station.read())

            # Data cleaning
            for idx_line, line in enumerate(data_station.split('\\n')[:-1]):
                if idx_line == 0:
                    arr_station = np.array(line.replace('b', '').replace("'", '').strip().split(';'), dtype=float)              
                else:
                    arr_station = np.row_stack((arr_station, np.array(line.replace('b', '').replace("'", '').strip().split(';'), dtype=float)))

            # Replace extreme values by NaN
            arr_station[:, 1][arr_station[:, 1] == -99999] = np.nan

            # Create datetime array
            arr_datetime = np.array([datetime.datetime(int(arr_station[i, 0]), 1, 1) + datetime.timedelta(days=(arr_station[i, 0] - int(arr_station[i, 0])) * 365.25) for i in range(len(arr_station[:, 0]))])

            # Create DataFrame
            df_station = pd.DataFrame(np.column_stack((arr_datetime, arr_station[:, 1])), columns=['datetime', 'sea_level_{}'.format(self.closest_stations_PSMSL.loc[self.closest_stations_PSMSL.index[idx_PSMSL]]['station_name'])])
            df_station = df_station.set_index('datetime')

            # Concatenate information
            if idx_PSMSL == 0:
                df_PSMSL_total = df_station
            
            else:
                df_PSMSL_total = pd.concat([df_PSMSL_total, df_station], axis=1)

        # Save information in dictionary
        df_PSMSL_total = df_PSMSL_total.apply(pd.to_numeric)
        self.island_info['timeseries_PSMSL']['timeseries'] = df_PSMSL_total

    def main(self):

        # Retrieve the dictionary with currently available information about the island
        self.island_info = get_info_islands.retrieve_info_island(self.island, self.country, verbose=self.verbose_init)

        print('\n-------------------------------------------------------------------')
        print('RETRIEVING TIME SERIES FOR TIDE-GAUGE SEA-LEVEL (PSMSL) DATA')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        # If PSMSL data have NOT already been generated
        if not 'timeseries_PSMSL' in self.island_info.keys() or self.overwrite:

            # Create key/dict for PSMSL data
            self.island_info['timeseries_PSMSL'] = {}
            self.assign_metadata()

            # Run all functions
            self.get_timeseries()
        
        # If PSMSL data have already been generated
        else:

            print('~ Information already available. Returning data. ~')

        # Save information in dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)

        return self.island_info
