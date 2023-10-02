"""
This module calculates time series of temperature from remote sensing (ERA5) for a given island.
NOTE: Inspired by Island Health Explorer
TODO: remove seasons?
Citation: Copernicus Climate Change Service (C3S) (2017): ERA5: Fifth generation of ECMWF atmospheric reanalyses of the global climate. Copernicus Climate Change Service Climate Data Store (CDS), (date of access), https://cds.climate.copernicus.eu/cdsapp#!/home

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
from mpetools import get_info_islands
import ee
import os
import pickle
import pandas as pd

class TimeSeriesERA5:
    def __init__(self, island, country, verbose_init=True, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands'), overwrite=False):
        self.island = island
        self.country = country
        self.verbose_init = verbose_init
        self.island_info_path = island_info_path
        self.overwrite = overwrite

    def assign_metadata(self):

        # Add description of this database
        self.island_info['timeseries_ERA5']['description'] = 'This module calculates time series of temperature from remote sensing (ERA5) for a given island.'

        # Add description of this timeseries
        self.island_info['timeseries_ERA5']['description_timeseries'] = 'Available information: Daily mean air temperature at 2m, max air temperature at 2m, min air temperature at 2m, dewpoint temperature at 2m, total precipitation, surface pressure, mean sea level pressure, wind components.'
        
        # Set units
        self.island_info['timeseries_ERA5']['units'] = {'mean_2m_air_temperature': 'K',
                                        'minimum_2m_air_temperature': 'K',
                                        'maximum_2m_air_temperature': 'K',
                                        'dewpoint_2m_temperature': 'K',
                                        'total_precipitation': 'm',
                                        'surface_pressure': 'Pa',
                                        'mean_sea_level_pressure': 'Pa',
                                        'u_component_of_wind_10m': 'm/s',
                                        'v_component_of_wind_10m': 'm/s'}
        
        # Add source (paper or url)
        self.island_info['timeseries_ERA5']['source'] = 'Copernicus Climate Change Service (C3S) (2017): ERA5: Fifth generation of ECMWF atmospheric reanalyses of the global climate. Copernicus Climate Change Service Climate Data Store (CDS), (date of access), https://cds.climate.copernicus.eu/cdsapp#!/home'

    def get_timeseries(self):

        def reduceRegionMean(img):

            # Calculate mean with ee.Reducer
            img_mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=polygon, scale=30).get(info)

            return img.set('date', img.date().format()).set('mean', img_mean)

        # Retrieve ERA-5 collection from GEE
        collection_ERA5 = ee.ImageCollection('ECMWF/ERA5/DAILY')

        # Retrieve information from dictionary or inputs
        polygon = self.island_info['spatial_reference']['polygon']

        # List of informations to retrieve
        list_to_retrieve = list(self.island_info['timeseries_ERA5']['units'].keys())

        # Loop in all information to retrieve
        for idx_ERA5, info in enumerate(list_to_retrieve):

            print('~ Retrieving {}. ~'.format(info.replace('_', ' ').capitalize()))

            # Filter bounds and dates, select information
            collection = collection_ERA5.filterBounds(polygon).select(info)

            # Take mean of the region
            collection_mean = collection.map(reduceRegionMean)

            # Create list with information
            nested_list = collection_mean.reduceColumns(ee.Reducer.toList(2), ['date', 'mean']).values().get(0)

            # Create pandas.DataFrame
            df_ERA5 = pd.DataFrame(nested_list.getInfo(), columns=['datetime', info])

            # Convert to date to datetime
            df_ERA5['datetime'] = pd.to_datetime(df_ERA5['datetime'])
            df_ERA5 = df_ERA5.set_index('datetime')

            if idx_ERA5 == 0:

                df_ERA5_total = df_ERA5

            else:

                df_ERA5_total = pd.concat([df_ERA5_total, df_ERA5], axis=1)
            
        # Save information in dictionary
        df_ERA5 = df_ERA5.apply(pd.to_numeric)
        self.island_info['timeseries_ERA5']['timeseries'] = df_ERA5_total
        
    def main(self):

        # Retrieve the dictionary with currently available information about the island
        self.island_info = get_info_islands.retrieve_info_island(self.island, self.country, verbose=self.verbose_init)

        print('\n-------------------------------------------------------------------')
        print('RETRIEVING DAILY TEMPERATURE, TOTAL PRECIPITATION, SURFACE PRESSURE, MEAN SEA LEVEL PRESSURE, WIND (ERA-5) DATA')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        # If ERA5 data have NOT already been generated
        if not 'timeseries_ERA5' in self.island_info.keys() or self.overwrite:

            # Create key/dict for ERA5 data
            self.island_info['timeseries_ERA5'] = {}
            self.assign_metadata()

            # Run all functions
            self.get_timeseries()
        
        # If ERA5 data have already been generated
        else:
            print('~ Information already available. Returning data. ~')

        # Save dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)
        
        return self.island_info
