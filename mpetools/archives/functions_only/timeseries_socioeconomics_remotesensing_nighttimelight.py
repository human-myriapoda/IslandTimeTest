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

def retrieveNTL(island_info):

    def reduceRegionMean(img):

        # Calculate mean with ee.Reducer
        img_mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=polygon, scale=30).get('b1')

        return img.set('date', img.date().format()).set('mean', img_mean)

    # Retrieve NTL collection from GEE
    collection_NTL = ee.ImageCollection("BNU/FGS/CCNL/v1")

    # Retrieve information from dictionary or inputs
    polygon = island_info['spatial_reference']['polygon']

    print('~ Retrieving NTL time series. ~')

    # Filter bounds and dates, select information
    collection = collection_NTL.filterBounds(polygon)

    # Take mean of the region
    collection_mean = collection.map(reduceRegionMean)

    # Create list with information
    nested_list = collection_mean.reduceColumns(ee.Reducer.toList(2), ['date', 'mean']).values().get(0)

    # Create pandas.DataFrame
    df = pd.DataFrame(nested_list.getInfo(), columns=['date', 'NTL'])

    # Convert to date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # If DataFrame is not empty (0)
    if all([el == 0 for el in df.NTL]):

        print("Time series is empty!")
    
    else:

        # Save information in dictionary
        island_info['Nighttime light']['DataFrame'] = df
    
    return island_info

def getNTL(island, country, verbose_init=True, island_info_path=os.getcwd()+'\\data\\info_islands'):

    # Retrieve the dictionary with currently available information about the island.
    island_info = get_info_islands.retrieveInfoIslands(island, country, verbose=verbose_init)

    print('\n-------------------------------------------------------------------')
    print('RETRIEVING NIGHTTIME LIGHT DATA')
    print('Island:', ', '.join([island, country]))
    print('-------------------------------------------------------------------\n')

    # If NTL data have NOT already been generated
    if not 'Nighttime light' in island_info.keys():

        # Create key/dict for NTL data
        island_info['Nighttime light'] = {}

        # Run all functions
        island_info = retrieveNTL(island_info)
    
    # If NTL data have already been generated
    else:

        print('~ Information already available. Returning data. ~')

    # Save dictionary
    fw = open(island_info_path + '\\info_{}_{}.data'.format(island_info['general_info']['island'], island_info['general_info']['country']), 'wb')
    pickle.dump(island_info, fw)
    fw.close()

    return island_info