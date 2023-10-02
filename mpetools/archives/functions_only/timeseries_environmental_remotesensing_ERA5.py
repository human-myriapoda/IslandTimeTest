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
import numpy as np

def retrieveERA5(island_info):

    def reduceRegionMean(img):

        # Calculate mean with ee.Reducer
        img_mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=polygon, scale=30).get(info)

        return img.set('date', img.date().format()).set('mean', img_mean)

    # Retrieve ERA-5 collection from GEE
    collection_ERA5 = ee.ImageCollection('ECMWF/ERA5/DAILY')

    # Retrieve information from dictionary or inputs
    polygon = island_info['spatial_reference']['polygon']
    start_date = island_info['ERA5']['date_range'][0]
    end_date = island_info['ERA5']['date_range'][1]

    # List of informations to retrieve
    list_to_retrieve = list(island_info['ERA5']['units'].keys())

    # Loop in all information to retrieve
    for info in list_to_retrieve:

        print('~ Retrieving {}. ~'.format(info.replace('_', ' ').capitalize()))

        # Filter bounds and dates, select information
        collection = collection_ERA5.filterBounds(polygon).filterDate(start_date, end_date).select(info)

        # Take mean of the region
        collection_mean = collection.map(reduceRegionMean)

        # Create list with information
        nested_list = collection_mean.reduceColumns(ee.Reducer.toList(2), ['date', 'mean']).values().get(0)

        # Create pandas.DataFrame
        df = pd.DataFrame(nested_list.getInfo(), columns=['date', info])

        # Convert to date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Save information in dictionary
        island_info['ERA5'][info] = df
    
    return island_info

def getERA5(island, country, date_range: list, verbose_init=True, island_info_path=os.getcwd()+'\\data\\info_islands'):

    # Retrieve the dictionary with currently available information about the island.
    island_info = get_info_islands.retrieveInfoIslands(island, country, verbose=verbose_init)

    print('\n-------------------------------------------------------------------')
    print('RETRIEVING DAILY TEMPERATURE, TOTAL PRECIPITATION, SURFACE PRESSURE, MEAN SEA LEVEL PRESSURE, WIND (ERA-5) DATA')
    print('Island:', ', '.join([island, country]))
    print('-------------------------------------------------------------------\n')

    # If ERA5 data have NOT already been generated
    if not 'ERA5' in island_info.keys():

        # Create key/dict for ERA5 data
        island_info['ERA5'] = {}

        # Set date range (format YYYY-MM-DD)
        island_info['ERA5']['date_range'] = date_range

        # Set units
        island_info['ERA5']['units'] = {'mean_2m_air_temperature': 'K',
                                        'minimum_2m_air_temperature': 'K',
                                        'maximum_2m_air_temperature': 'K',
                                        'dewpoint_2m_temperature': 'K',
                                        'total_precipitation': 'm',
                                        'surface_pressure': 'Pa',
                                        'mean_sea_level_pressure': 'Pa',
                                        'u_component_of_wind_10m': 'm/s',
                                        'v_component_of_wind_10m': 'm/s'}

        # Run all functions
        island_info = retrieveERA5(island_info)
    
    # If ERA5 data have already been generated
    else:

        # Check if date range matches the existing date range
        if not np.array_equal(island_info['ERA5']['date_range'], date_range):

            print("Date range does not match with the existing date range! Generating data for the new date range.")
    
            # Set date range
            island_info['ERA5']['date_range'] = date_range

            # Retrieve time series
            island_info = retrieveERA5(island_info)
        
        else:
            
            print('~ Date range matches with the existing date range. Returning data. ~')

    # Save dictionary
    fw = open(island_info_path + '\\info_{}_{}.data'.format(island_info['general_info']['island'], island_info['general_info']['country']), 'wb')
    pickle.dump(island_info, fw)
    fw.close()

    return island_info