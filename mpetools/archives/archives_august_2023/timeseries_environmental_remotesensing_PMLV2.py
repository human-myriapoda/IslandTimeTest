"""
This module calculates time series of Gross primary product, Vegetation transpiration, Soil evaporation, \
Interception from vegetation canopy, Water body, snow and ice evaporation from remote sensing (PML_V2) for a given island.
Date range: 2000-2020
Citation: Penman-Monteith-Leuning Evapotranspiration V2 (PML_V2) products
GEE link: https://developers.google.com/earth-engine/datasets/catalog/CAS_IGSNRR_PML_V2_v017#description

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
from mpetools import get_info_islands
import ee
import os
import pickle
import pandas as pd
import numpy as np

class TimeSeriesPMLV2:
    def __init__(self, island, country, verbose_init=True, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands'), overwrite=False):
        self.island = island
        self.country = country
        self.verbose_init = verbose_init
        self.island_info_path = island_info_path
        self.overwrite = overwrite

    def assign_metadata(self):

        # Add description of this database
        self.island_info['timeseries_PMLV2']['description'] = 'This module calculates time series of Gross primary product, Vegetation transpiration, Soil evaporation, Interception from vegetation canopy, Water body, snow and ice evaporation from remote sensing (PML_V2) for a given island.'
        
        # Add description of this timeseries
        self.island_info['timeseries_PMLV2']['description_timeseries'] = {'GPP': 'Gross primary product',
                                        'Ec': 'Vegetation transpiration',
                                        'Es': 'Soil evaporation',
                                        'Ei': 'Interception from vegetation canopy',
                                        'ET_water': 'Water body, snow and ice evaporation'}
        
        # Set units
        self.island_info['timeseries_PMLV2']['units'] = {'GPP': 'gC m-2 d-1',
                                        'Ec': 'mm/d',
                                        'Es': 'mm/d',
                                        'Ei': 'mm/d',
                                        'ET_water': 'mm/d'}
        
        # Add source (paper or url)
        self.island_info['timeseries_PMLV2']['source'] = 'Penman-Monteith-Leuning Evapotranspiration V2 (PML_V2) products. GEE: https://developers.google.com/earth-engine/datasets/catalog/CAS_IGSNRR_PML_V2_v017#description'

    def get_timeseries(self):

        def reduceRegionMean(img):

            # Calculate mean with ee.Reducer
            img_mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=polygon, scale=30).get(info_PMLV2)

            return img.set('date', img.date().format()).set('mean', img_mean)

        # Retrieve PML_V2 collection from GEE
        collection_PML_V2 = ee.ImageCollection('CAS/IGSNRR/PML/V2_v017')

        # Retrieve information from dictionary or inputs
        polygon = self.island_info['spatial_reference']['polygon']

        # List of informations to retrieve
        list_to_retrieve = list(self.island_info['timeseries_PMLV2']['description_timeseries'].keys())

        idx_PMLV2 = 0

        # Loop in all information to retrieve
        for info_PMLV2 in list_to_retrieve:

            print('~ Retrieving {}. ~'.format(self.island_info['timeseries_PMLV2']['description_timeseries'][info_PMLV2]))

            # Filter bounds and dates, select information
            collection = collection_PML_V2.filterBounds(polygon).select(info_PMLV2)

            # Take mean of the region
            collection_mean = collection.map(reduceRegionMean)

            # Create list with information
            nested_list = collection_mean.reduceColumns(ee.Reducer.toList(2), ['date', 'mean']).values().get(0)

            if np.shape(np.array(nested_list.getInfo()))[0] == 0:
                continue

            else:

                if idx_PMLV2 == 0:
                    df_PMLV2 = pd.DataFrame(nested_list.getInfo(), columns=['datetime', self.island_info['timeseries_PMLV2']['description_timeseries'][info_PMLV2].replace(',', '').replace(' ', '_').lower()])
            
                else:
                    df_PMLV2[self.island_info['timeseries_PMLV2']['description_timeseries'][info_PMLV2].replace(',', '').replace(' ', '_').lower()] = np.array(nested_list.getInfo())[:, 1].astype('float')

                idx_PMLV2 += 1

        if 'df_PMLV2' in locals():

            # Convert to date to datetime
            df_PMLV2['datetime'] = pd.to_datetime(df_PMLV2['datetime'])
            df_PMLV2.set_index('datetime', inplace=True)
                
            if df_PMLV2.shape[0] > 0:

                # Save information in dictionary
                df_PMLV2 = df_PMLV2.apply(pd.to_numeric)
                self.island_info['timeseries_PMLV2']['timeseries'] = df_PMLV2
        
    def main(self):

        # Retrieve the dictionary with currently available information about the island
        self.island_info = get_info_islands.retrieve_info_island(self.island, self.country, verbose=self.verbose_init)

        print('\n-------------------------------------------------------------------')
        print('RETRIEVING Coupled Evapotranspiration and Gross Primary Product (PML_V2) DATA')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        # If PML_V2 data have NOT already been generated
        if not 'timeseries_PMLV2' in self.island_info.keys() or self.overwrite:

            # Create key/dict for PML_V2 data
            self.island_info['timeseries_PMLV2'] = {}
            self.assign_metadata()

            # Run all functions
            self.get_timeseries()
        
        # If PML_V2 data have already been generated
        else:
            print('~ Information already available. Returning data. ~')

        # Save dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)
        
        return self.island_info
