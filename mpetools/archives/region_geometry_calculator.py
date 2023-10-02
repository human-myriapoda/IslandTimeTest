"""
This module generates a ee.Geometry object for a given island based on a cloud-free Landsat 8 image.
TODO: adapt it for weirdly-shape islands.

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Load modules 
import ee
import numpy as np
import geemap
import os
import pickle

def spectralIndex_NDVI(image, geo_polygon, band1='B5', band2='B4'):

    arr_band1 = np.array(image.sampleRectangle(geo_polygon).get(band1).getInfo())
    arr_band2 = np.array(image.sampleRectangle(geo_polygon).get(band2).getInfo())

    arr_NDVI = (arr_band1 - arr_band2)/(arr_band1 + arr_band2)

    return arr_NDVI

def generateGeoPolygon(key, lat, lon, var):

    if key == 'top-left':
        geo_polygon = ee.Geometry.Polygon(
        [[[lon-var, lat+var],
            [lon-var, lat],
            [lon, lat],
            [lon, lat+var]]], None, False)

    elif key == 'top-right':
        geo_polygon = ee.Geometry.Polygon(
        [[[lon, lat+var],
            [lon, lat],
            [lon+var, lat],
            [lon+var, lat+var]]], None, False)

    elif key == 'bottom-left':
        geo_polygon = ee.Geometry.Polygon(
        [[[lon-var, lat],
            [lon-var, lat-var],
            [lon, lat-var],
            [lon, lat]]], None, False) 

    elif key == 'bottom-right':
        geo_polygon = ee.Geometry.Polygon(
        [[[lon, lat],
            [lon, lat-var],
            [lon+var, lat-var],
            [lon+var, lat]]], None, False)

    elif key == 'final':
        geo_polygon = ee.Geometry.Polygon(
        [[[lon-var, lat+var],
            [lon-var, lat-var],
            [lon+var, lat-var],
            [lon+var, lat+var]]], None, False)
    
    return geo_polygon

def mapping_geometry(image, geo_polygon, lat, lon):

    # Mapping for optional verification
    Map = geemap.Map(center=[lat, lon], zoom=12)
    vizParams = {'bands': ['B4', 'B3', 'B2']}
    Map.addLayer(image.clip(geo_polygon), vizParams, 'Landsat 8 image')
    Map.centerObject(image, 9)  

    return Map

def regionGeometryCalculator(island_info: dict, var_init=0.001, var_change=0.005, thresold_ndvi=0.2, var_limit=0.065, overwrite=False, coordinates_path=os.getcwd()+'/data/coordinates_geometry'):

    # Specify a geometric point in GEE
    lat = float(island_info['latitude'])
    lon = float(island_info['longitude'])
    point = ee.Geometry.Point([lon, lat])

    # Availability – Landsat-8 (OLI/TIRS)
    collection_L8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA').filterBounds(point)
    size_L8  = collection_L8.size().getInfo()
    print('Landsat-8 (OLI/TIRS):', size_L8, 'Tier-1 TOA are available')

    # Pick cloud-free image
    image = collection_L8.sort('CLOUD_COVER').first()
    
    # Verify if geometry already exists
    if island_info['geometry'] is None:

        print('We are calculating the geometry for the first time! \n')
    
    elif island_info['geometry'] is not None and not overwrite:

        print("The geometry has already been calculated, call island_info['geometry'] \n")
        Map = mapping_geometry(image, island_info['geometry'], lat, lon)

        return Map, island_info
    
    elif island_info['geometry'] is not None and overwrite:

        print("The geometry has already been calculated, but the user wishes to recalculate it. \n")

    print('--------------------------------------------------------------')
    print('CALCULATING GEOMETRY FOR ', island_info['island'], island_info['country'], ' located at ', island_info['latitude'], island_info['longitude'])
    print('--------------------------------------------------------------\n')

    dict_coords = {"top-left": ['tl', 0, 0],
                        "top-right": ['tr', 0, -1],
                        "bottom-left": ['bl', -1, 0],
                        "bottom-right": ['br', -1, -1]}

    # Initial conditions for while loop
    var = var_init

    # While loop
    while var < var_limit:

        bool_borders = []
        bool_full = []
        print('var = ', var)

        # For loop for every 'corner'
        for key in dict_coords.keys():
            
            # Generate polygon for a given 'corner'
            geo_polygon = generateGeoPolygon(key, lat, lon, var)

            # Calculate NDVI for every pixel of that 'corner'
            arr_NDVI = spectralIndex_NDVI(image, geo_polygon)

            # Create array with the NDVI values on the border of the 'corner'
            arr_borders = np.concatenate((arr_NDVI[dict_coords[key][1], :], arr_NDVI[:, dict_coords[key][2]]))

            # Test conditions
            bool_borders.append(np.any(arr_borders > thresold_ndvi))
            bool_full.append(np.all(arr_NDVI < thresold_ndvi))

        # If we reach the limit, the loop ends and we calculate the region as is
        if (var >= (var_limit - var_change)):  

            print('Maximum limit reached, code will stop')
            geo_polygon_final = generateGeoPolygon(key='final', lat=lat, lon=lon, var=var)
            Map = mapping_geometry(image, geo_polygon_final, lat, lon)
            
            # Update dict with geometry
            island_info['geometry'] = geo_polygon_final
            fw = open(coordinates_path+'/info_{}_{}.data'.format(island_info['island'], island_info['country']), 'wb')
            pickle.dump(island_info, fw)
            fw.close()

            return Map, island_info

        # Still land on the borders -> we expand the region at the next iteration
        if np.any(np.array(bool_borders)):
            var += var_change

        # Only water -> no island -> we expand the region at the next iteration
        elif np.all(np.array(bool_full)):
            var += var_change

        # We found an island surrounded by water -> loop ends
        else:
            var += var_change
            print('Done!')
            geo_polygon_final = generateGeoPolygon(key='final', lat=lat, lon=lon, var=var)
            Map = mapping_geometry(image, geo_polygon_final, lat, lon)
            
            # Update dict with geometry
            island_info['geometry'] = geo_polygon_final
            fw = open(coordinates_path+'/info_{}_{}.data'.format(island_info['island'], island_info['country']), 'wb')
            pickle.dump(island_info, fw)
            fw.close()

            return Map, island_info






