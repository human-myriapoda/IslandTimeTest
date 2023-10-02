"""
This module lists all available images for a given point/geometry for a list of satellites.
Optionally the user can provide a cloud cover threshold to find cloud-free images.
It returns a dictionary with filtered ImageCollection for each satellite.

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Load modules 
import ee
import datetime
import os
import pickle

def getSatelliteAvailability(island_info: dict, list_sat_user=None, date_range_user=None, cloud_threshold=100, type_image='SR'):

        fw = open(os.getcwd() + '\\data\\satellite_info\\dict_satellite.data', 'rb')
        dict_satellite = pickle.load(fw)

        # Create geometry point
        point = ee.Geometry.Point([island_info['longitude'], island_info['latitude']])

        # Create date range
        if date_range_user is None and list_sat_user is None:

                date_range = [dict_satellite['L5']['SR dataset availability'][0], datetime.datetime.now()]

        elif date_range_user is None and list_sat_user is not None:

                novelty_index_sat = [dict_satellite[sat]['novelty index'] for sat in list_sat_user]
                min_idx, max_idx = novelty_index_sat.index(min(novelty_index_sat)), novelty_index_sat.index(max(novelty_index_sat))
                date_range = [dict_satellite[list_sat_user[min_idx]]['SR dataset availability'][0], dict_satellite[list_sat_user[max_idx]]['SR dataset availability'][1]]

        else: 
                date_range = date_range_user
                

        # Create satellite list
        if list_sat_user is None: list_sat = list(dict_satellite.keys())
        else: list_sat = list_sat_user

        # Create empty dictionary for filtered ImageCollection
        image_collection_dict = {'Description': 'Filtered (cloud threshold of {}%) ImageCollection for satellites of interest for {}, {}'.format(cloud_threshold, island_info['island'], island_info['country'])}

        # Loop in the list of satellites
        for sat in list_sat:

                collection_sat = ee.ImageCollection(dict_satellite[sat]['{} GEE Snipper'.format(type_image)]) \
                                   .filterDate(date_range[0], date_range[1]) \
                                   .filterBounds(point) \
                                   .filterMetadata(dict_satellite[sat]['cloud label'], 'less_than', cloud_threshold)

                size_collection_sat = collection_sat.size().getInfo()
                print(sat, size_collection_sat)
                if size_collection_sat > 0:
                        image_collection_dict[sat] = collection_sat

        return image_collection_dict
    