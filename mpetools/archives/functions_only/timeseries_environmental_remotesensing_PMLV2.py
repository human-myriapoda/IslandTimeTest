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

def retrievePML_V2(island_info):

    def reduceRegionMean(img):

        # Calculate mean with ee.Reducer
        img_mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=polygon, scale=30).get(info)

        return img.set('date', img.date().format()).set('mean', img_mean)

    # Retrieve PML_V2 collection from GEE
    collection_PML_V2 = ee.ImageCollection('CAS/IGSNRR/PML/V2_v017')

    # Retrieve information from dictionary or inputs
    polygon = island_info['spatial_reference']['polygon']
    start_date = island_info['PML_V2']['date_range'][0]
    end_date = island_info['PML_V2']['date_range'][1]

    # List of informations to retrieve
    list_to_retrieve = list(island_info['PML_V2']['descriptions'].keys())

    # Loop in all information to retrieve
    for info in list_to_retrieve:

        print('~ Retrieving {}. ~'.format(island_info['PML_V2']['descriptions'][info]))

        # Filter bounds and dates, select information
        collection = collection_PML_V2.filterBounds(polygon).filterDate(start_date, end_date).select(info)

        # Take mean of the region
        collection_mean = collection.map(reduceRegionMean)

        # Create list with information
        nested_list = collection_mean.reduceColumns(ee.Reducer.toList(2), ['date', 'mean']).values().get(0)

        # Create pandas.DataFrame
        df = pd.DataFrame(nested_list.getInfo(), columns=['date', info])

        # Convert to date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        if df.shape[0] > 0:

            # Save information in dictionary
            island_info['PML_V2'][info] = df
    
    return island_info

def getPML_V2(island, country, date_range=['2000-02-26', '2020-12-26'], verbose_init=True, island_info_path=os.getcwd()+'\\data\\info_islands'):

    # Retrieve the dictionary with currently available information about the island.
    island_info = get_info_islands.retrieveInfoIslands(island, country, verbose=verbose_init)

    print('\n-------------------------------------------------------------------')
    print('RETRIEVING Coupled Evapotranspiration and Gross Primary Product (PML_V2) DATA')
    print('Island:', ', '.join([island, country]))
    print('-------------------------------------------------------------------\n')

    # If PML_V2 data have NOT already been generated
    if not 'PML_V2' in island_info.keys():

        # Create key/dict for PML_V2 data
        island_info['PML_V2'] = {}

        # Set date range (format YYYY-MM-DD)
        island_info['PML_V2']['date_range'] = date_range

        # Set units
        island_info['PML_V2']['units'] = {'GPP': 'gC m-2 d-1',
                                        'Ec': 'mm/d',
                                        'Es': 'mm/d',
                                        'Ei': 'mm/d',
                                        'ET_water': 'mm/d'}

        # Set descriptions
        island_info['PML_V2']['descriptions'] = {'GPP': 'Gross primary product',
                                        'Ec': 'Vegetation transpiration',
                                        'Es': 'Soil evaporation',
                                        'Ei': 'Interception from vegetation canopy',
                                        'ET_water': 'Water body, snow and ice evaporation'}

        # Run all functions
        island_info = retrievePML_V2(island_info)
    
    # If ERA5 data have already been generated
    else:

        print('~ Information already available. Returning data. ~')

    # Save dictionary
    fw = open(island_info_path + '\\info_{}_{}.data'.format(island_info['general_info']['island'], island_info['general_info']['country']), 'wb')
    pickle.dump(island_info, fw)
    fw.close()

    return island_info