"""
This module calculates time series of nighttime light (NTL) from remote sensing (DMSP-OLS) for a given island.
Date range: 1992-2013
Citation: Zhao,Chenchen, Cao,Xin, Chen,Xuehong, & Cui,Xihong. (2020). A Consistent and Corrected Nighttime Light dataset (CCNL 1992-2013) from DMSP-OLS data (Version 1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6644980
GEE link: https://developers.google.com/earth-engine/datasets/catalog/BNU_FGS_CCNL_v1#citations

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
from mpetools import get_info_islands
import ee
import os
import pickle
import pandas as pd

class TimeSeriesNighttimeLight:
    def __init__(self, island, country, verbose_init=True, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands'), overwrite=False):
        self.island = island
        self.country = country
        self.verbose_init = verbose_init
        self.island_info_path = island_info_path
        self.overwrite = overwrite

    def assign_metadata(self):

        # Add description of this database
        self.island_info['timeseries_nighttime_light']['description'] = 'This module calculates time series of nighttime light (NTL) from remote sensing (DMSP-OLS) for a given island.'
        
        # Add description of this timeseries
        self.island_info['timeseries_nighttime_light']['description_timeseries'] = 'TODO'

        # Add source (paper or url)
        self.island_info['timeseries_nighttime_light']['source'] = 'Zhao, Cao, Chen & Cui (2020). A Consistent and Corrected Nighttime Light dataset (CCNL 1992-2013) from DMSP-OLS data (Version 1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6644980. GEE: https://developers.google.com/earth-engine/datasets/catalog/BNU_FGS_CCNL_v1#citations'

    def get_timeseries(self):

        def reduceRegionMean(img):

            # Calculate mean with ee.Reducer
            img_mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=polygon, scale=30).get('b1')

            return img.set('date', img.date().format()).set('mean', img_mean)

        # Retrieve NTL collection from GEE
        collection_NTL = ee.ImageCollection("BNU/FGS/CCNL/v1")

        # Retrieve information from dictionary or inputs
        polygon = self.island_info['spatial_reference']['polygon']

        print('~ Retrieving NTL time series. ~')

        # Filter bounds and dates, select information
        collection = collection_NTL.filterBounds(polygon)

        # Take mean of the region
        collection_mean = collection.map(reduceRegionMean)

        # Create list with information
        nested_list = collection_mean.reduceColumns(ee.Reducer.toList(2), ['date', 'mean']).values().get(0)

        # Create pandas.DataFrame
        df_NTL = pd.DataFrame(nested_list.getInfo(), columns=['datetime', 'nighttime_light'])

        # Convert to date to datetime
        df_NTL['datetime'] = pd.to_datetime(df_NTL['datetime'])
        df_NTL = df_NTL.set_index('datetime')
        
        # If DataFrame is not empty (0)
        if all([el == 0 for el in df_NTL.nighttime_light]):

            print("Time series is empty!")
        
        else:

            # Save information in dictionary
            df_NTL = df_NTL.apply(pd.to_numeric)
            self.island_info['timeseries_nighttime_light']['timeseries'] = df_NTL

    def main(self):

        # Retrieve the dictionary with currently available information about the island
        self.island_info = get_info_islands.retrieve_info_island(self.island, self.country, verbose=self.verbose_init)

        print('\n-------------------------------------------------------------------')
        print('RETRIEVING NIGHTTIME LIGHT DATA')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        # If NTL data have NOT already been generated
        if not 'timeseries_nighttime_light' in self.island_info.keys() or self.overwrite:

            # Create key/dict for NTL data
            self.island_info['timeseries_nighttime_light'] = {}
            self.assign_metadata()

            # Run all functions
            self.get_timeseries()
        
        # If NTL data have already been generated
        else:
            print('~ Information already available. Returning data. ~')

        # Save dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)
        
        return self.island_info