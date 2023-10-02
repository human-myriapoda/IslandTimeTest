"""
This module allows to retrieve all the information needed before generating time series.
1. From the name of the island, retrieve coordinates (latitude, longitude).
2. From coordinates, generate an approximate square region around the island.
3. Build a dictionary with ImageCollections for a list of satellites.

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Load modules 
import numpy as np
import os
import urllib.request
import re
import requests
import pickle
from fastkml import kml
import geemap
import datetime
import ee
import osmnx as ox
import geojson

############################################################################################################
# STEP 1: COORDINATES
############################################################################################################

"""
This section contains all the functions needed for extracting the latitude and longitude of any given island (by its name).
TODO: merge KML function in the main function 

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

def coordinatesRetrieve_kml(country, file_kml):

    # Read .kml file
    with open(file_kml, 'rt', encoding="utf-8") as myfile: 
        doc = myfile.read()
    k = kml.KML()
    k.from_string(doc)
    string_kml = k.to_string(prettyprint=True)

    # Regex patterns for lat, lon and name of the islands
    pattern_coord = r'\-\d{2}\.\d{6}\,\d{2}\.\d{6}'
    pattern_islands = r'name\>(.*?)\</ns1:name'

    # Use `regex` to find all those patterns
    coord = re.findall(pattern_coord, string_kml)
    islands = re.findall(pattern_islands, string_kml)

    # Remove the two first occurences (not necessary)
    if islands[1] == country or islands[1] == 'Islands':
        islands = islands[2:]

    else:
        islands = islands[1:]

    for i in range(len(islands)):

        # Empty dictionary
        dict_kml = {}

        # Fill the dictionary
        dict_kml['general_info']['island'] = islands[i]
        dict_kml['general_info']['country'] = country
        dict_kml['spatial_reference']['latitude'] = coord[i].split(',')[1]
        dict_kml['spatial_reference']['longitude'] = coord[i].split(',')[0]

        # Save a .data file
        path_info = os.getcwd() + '\\data\\info_islands'
        fw = open(path_info+'\\info_{}_{}.data'.format(islands[i], country), 'wb')
        pickle.dump(dict_kml, fw)
        fw.close()

def coordinatesRetrieve_wikipedia(island):

    # To fit Wikipedia's format
    place = island.replace(" ", "_")

    # Try this syntax: only the name of the island
    try:
        web_url = urllib.request.urlopen('https://en.wikipedia.org/wiki/{}'.format(place))

    # Try this syntax: the name of the island + _(island)
    except:
        try:
            web_url = urllib.request.urlopen('https://en.wikipedia.org/wiki/{}_(island)'.format(place))

        # Wikipedia page doesn't exist
        except:
            lat = np.nan
            lon = np.nan

            return lat, lon

    # Read data from website
    data = str(web_url.read())

    # Regex patterns for latitude and longitude
    pattern_lat = '"wgCoordinates":{"lat":(.*),"lon":'
    pattern_lon = '"lon":(.*)},"wg' #EditSubmit

    # Find the coordinates from the patterns
    try:
        lat = re.search(pattern_lat, data).group(1)
        lon = re.search(pattern_lon, data).group(1)
        lat = lat.replace('\\n', '')
        lon = lon.replace('\\n', '')

    # No patterns found
    except: 
        lat = np.nan
        lon = np.nan

    return lat, lon

def coordinatesRetrieve_geokeo(island, country):

    # To make sure we find the island and not another place with the same name
    place = "{}, {}".format(island, country)

    # Request url (from GeoKeo website)
    url_gk = 'https://geokeo.com/geocode/v1/search.php?q={}&api=YOUR_API_KEY'.format(place)
    resp = requests.get(url=url_gk)
    data = resp.json()

    # Retrieve the coordinates (from GeoKeo website)
    if 'status' in data:
        if data['status'] == 'ok':
            lat = data['results'][0]['geometry']['location']['lat']
            lon = data['results'][0]['geometry']['location']['lng']

        else:
            lat, lon = np.nan, np.nan
    else:
        lat, lon = np.nan, np.nan

    return lat, lon

def coordinatesConsensus(island, country, source='GeoKeo', verbose=True):

    # Extract the coordinates from Wikipedia
    lat_w, lon_w = coordinatesRetrieve_wikipedia(island)
    if verbose: print(lat_w, lon_w)

    # Extract the coordinates from GeoKeo
    lat_gk, lon_gk = coordinatesRetrieve_geokeo(island, country)
    if verbose: print(lat_gk, lon_gk)

    # Compare the values and if they are too different -> visual inspection
    if not np.nan in (lat_w, lon_w, lat_gk, lon_gk):
        lat_close = np.allclose(np.array(lat_w, dtype=float), np.array(lat_gk, dtype=float), atol=1e-1)
        lon_close = np.allclose(np.array(lon_w, dtype=float), np.array(lon_gk, dtype=float), atol=1e-1)

        if False in (lat_close, lon_close):
            ver = input('Is this your island [y/n]? Please verify those coordinates '+ str(lat_gk) +', ' + str(lon_gk) + ' on Google Maps: ')

            if ver == 'n':
                lat, lon = input('Please enter the latitude/longitude: ').replace(',', ' ').split(' ')

            elif ver == 'wiki':
                lat, lon = lat_w, lon_w        

            else:
                lat, lon = lat_gk, lon_gk

        else:
            lat, lon = lat_gk, lon_gk
    else:
        ver = input('Is this your island [y/n/wiki]? Please verify those coordinates '+ str(lat_gk) +', ' + str(lon_gk) + ' on Google Maps: ')

        if ver == 'n':
            lat, lon = input('Please enter the latitude/longitude: ').replace(',', ' ').split(' ')

        elif ver == 'wiki':
            lat, lon = lat_w, lon_w

        else:
            lat, lon = lat_gk, lon_gk

    return lat, lon

def coordinatesRetrieve(island_info, method='new'):

    # NEW METHOD (using OpenStreetMap)
    if method == 'new':

        area = ox.geocode_to_gdf(island_info['general_info']['island'] + ', ' + island_info['general_info']['country'])
        lat, lon = area.lat[0], area.lon[0]
    
    else: 

        # We call the function `coordinates_consensus` to retrieve lat, lon (OLD METHOD)
        lat, lon = coordinatesConsensus(island_info['general_info']['island'], island_info['general_info']['country'])

    # We save this info in the dictionary
    island_info['spatial_reference']['latitude'] = float(lat)
    island_info['spatial_reference']['longitude'] = float(lon)

    return island_info

############################################################################################################
# STEP 2: GEOMETRY
############################################################################################################

"""
This section generates a ee.Geometry object for a given island based on a cloud-free Landsat 8 image.
TODO: adapt it for weirdly-shape islands.
TODO: implement Map return

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

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

def mappingGeometry(image, geo_polygon, lat, lon):

    # Mapping for optional verification
    Map = geemap.Map(center=[lat, lon], zoom=12)
    vizParams = {'bands': ['B4', 'B3', 'B2']}
    Map.addLayer(image.clip(geo_polygon), vizParams, 'Satellite image')
    Map.centerObject(image, 9)  

    return Map

def regionGeometryCalculator(island_info: dict, dict_geometry: dict, verbose=True):

    # Extract dict_geometry
    var_init = dict_geometry['var_init']
    var_change = dict_geometry['var_change']
    thresold_NDVI = dict_geometry['thresold_NDVI']
    var_limit = dict_geometry['var_limit']

    # Specify a geometric point in GEE
    lat = float(island_info['spatial_reference']['latitude'])
    lon = float(island_info['spatial_reference']['longitude'])
    point = ee.Geometry.Point([lon, lat])

    # Availability – Landsat-8 (OLI/TIRS)
    collection_L8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA').filterBounds(point)

    # Pick cloud-free image
    image = collection_L8.sort('CLOUD_COVER').first()

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
        if verbose: print('var = ', var)

        # For loop for every 'corner'
        for key in dict_coords.keys():
            
            # Generate polygon for a given 'corner'
            geo_polygon = generateGeoPolygon(key, lat, lon, var)

            # Calculate NDVI for every pixel of that 'corner'
            arr_NDVI = spectralIndex_NDVI(image, geo_polygon)

            # Create array with the NDVI values on the border of the 'corner'
            arr_borders = np.concatenate((arr_NDVI[dict_coords[key][1], :], arr_NDVI[:, dict_coords[key][2]]))

            # Test conditions
            bool_borders.append(np.any(arr_borders > thresold_NDVI))
            bool_full.append(np.all(arr_NDVI < thresold_NDVI))

        # If we reach the limit, the loop ends and we calculate the region as is
        if (var >= (var_limit - var_change)):  

            if verbose: print('Maximum limit reached, code will stop')
            geo_polygon_final = generateGeoPolygon(key='final', lat=lat, lon=lon, var=var)
            Map = mappingGeometry(image, geo_polygon_final, lat, lon)
            
            # Update dict with geometry
            island_info['spatial_reference']['polygon'] = geo_polygon_final

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
            if verbose: print('Done!')
            geo_polygon_final = generateGeoPolygon(key='final', lat=lat, lon=lon, var=var)
            Map = mappingGeometry(image, geo_polygon_final, lat, lon)
            
            # Update dict with geometry
            island_info['spatial_reference']['polygon'] = geo_polygon_final

            return Map, island_info

def calculateGeometry(island_info: dict, dict_geometry: dict, method='new', verbose=True):

    if method == 'new':

        # New method (OpenStreetMap)
        try:
            area = ox.geocode_to_gdf(island_info['general_info']['island'] + ', ' + island_info['general_info']['country'])

            polygon_square_OSM = ee.Geometry.Polygon(
            [[[area.bbox_west[0], area.bbox_north[0]],
                [area.bbox_west[0], area.bbox_south[0]],
                [area.bbox_east[0], area.bbox_south[0]],
                [area.bbox_east[0], area.bbox_north[0]],
                [area.bbox_west[0], area.bbox_north[0]]]], None, False)      

            polygon_OSM_geojson = geojson.Polygon(list((area.geometry[0].exterior.coords)))
            polygon_OSM = ee.Geometry.Polygon(polygon_OSM_geojson['coordinates']) 

            island_info['spatial_reference']['polygon'] = polygon_square_OSM
            island_info['spatial_reference']['polygon_OSM'] = polygon_OSM
        
        except:

            # Old method
            _, island_info = regionGeometryCalculator(island_info, dict_geometry)
    
    else:

        # Old method
        _, island_info = regionGeometryCalculator(island_info, dict_geometry)

    return island_info


############################################################################################################
# STEP 3: SATELLITE AVAILABILITY
############################################################################################################

"""
This section lists all available images for a given point/geometry for a list of satellites.
Optionally the user can provide a cloud cover threshold to find cloud-free images.
It returns a dictionary with filtered ImageCollection for each satellite.

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

def getSatelliteAvailability(island_info: dict, list_sat_user=None, date_range_user=None, cloud_threshold=100, type_image='SR', verbose=True):

        fw = open(os.getcwd() + '\\data\\info_satellite\\dict_satellite.data', 'rb')
        dict_satellite = pickle.load(fw)
        fw.close()

        # Create geometry point
        point = ee.Geometry.Point([island_info['spatial_reference']['longitude'], island_info['spatial_reference']['latitude']])

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
        image_collection_dict = {'Description': 'Filtered (cloud threshold of {}%) ImageCollection for satellites of interest for {}, {}'.format(cloud_threshold, island_info['general_info']['island'], island_info['general_info']['country'])}

        # Loop in the list of satellites
        for sat in list_sat:

                collection_sat = ee.ImageCollection(dict_satellite[sat]['{} GEE Snipper'.format(type_image)]) \
                                   .filterDate(date_range[0], date_range[1]) \
                                   .filterBounds(point) \
                                   .filterMetadata(dict_satellite[sat]['cloud label'], 'less_than', cloud_threshold)

                size_collection_sat = collection_sat.size().getInfo()
                if verbose: print(sat, size_collection_sat)
                if size_collection_sat > 0:
                        image_collection_dict[sat] = collection_sat

        island_info['image_collection_dict'] = image_collection_dict

        return island_info

############################################################################################################
# MAIN FUNCTION
############################################################################################################

def getInfoIsland(island, country, toDo=None, dict_geometry={'var_init': 0.001, 'var_change': 0.002, 'thresold_NDVI': 0.2, 'var_limit': 0.065}, list_sat_user=None, date_range_user=None, cloud_threshold=10, type_image='SR', method='new', verbose=False):

    print('\n-------------------------------------------------------------------')
    print('PRE-TIME-SERIES INFORMATION')
    print('Island:', ', '.join([island, country]))
    print('-------------------------------------------------------------------\n')

    # Define path for data
    island_info_path = os.getcwd()+'\\data\\info_islands'

    # If the path in which the data will be stored doesn't exist, we create it
    if not os.path.exists(island_info_path): os.makedirs(island_info_path)

    # Check what information is already available
    if os.path.isfile(island_info_path + '\\info_{}_{}.data'.format(island, country)): 

        # Load the .data file with pickle
        fw = open(island_info_path + '\\info_{}_{}.data'.format(island, country), 'rb')
        island_info = pickle.load(fw)    
        fw.close()

        # If actions are defined by the user, skip to steps
        if toDo is None:
            
            toDo = {'coordinates': True, 'polygon': True, 'availability': True}

            # 1. Coordinates
            if island_info['spatial_reference']['latitude'] is not None and island_info['spatial_reference']['longitude'] is not None: toDo['coordinates'] = False

            # 2. Geometry
            if island_info['spatial_reference']['polygon'] is not None: toDo['polygon'] = False

            # 3. Image availability
            if island_info['image_collection_dict'] is not None: toDo['availability'] = False

            # If all available information is already available, return dictionary
            if not any(toDo.values()):

                print('~ All information is already available, returning information ~\n')
                return island_info

            else: print('~ The following information will be extracted/calculated:', ' and '.join([key for (key, val) in toDo.items() if val]), '~\n')
            
        else: print('~ The user wishes to extract/calculate', ' and '.join([key for (key, val) in toDo.items() if val]), '~\n')

    # No file exists
    else: 

        print('~ All information will be extracted/calculated. ~\n')
        island_info = {'general_info': {'island': island, 'country': country}, \
                       'spatial_reference': {'latitude': None, 'longitude': None, 'polygon': None}, \
                       'image_collection_dict': None}

        toDo = {'coordinates': True, 'polygon': True, 'availability': True}
    
    # (Re-)Calculating missing information
    for missing_info in [key for (key, val) in toDo.items() if val]:

        # Step 1: Extract coordinates (latitude, longitude)
        if missing_info == 'coordinates':
            island_info = coordinatesRetrieve(island_info, method)

        # Step 2: Calculate geometry (ee.Geometry.Polygon)
        elif missing_info == 'polygon':
            island_info = calculateGeometry(island_info, dict_geometry, method)

        # Step 3: Build a dictionary with satellite availability
        elif missing_info == 'availability':
            island_info = getSatelliteAvailability(island_info, list_sat_user, date_range_user, cloud_threshold, type_image)

    # Save dictionary
    fw = open(island_info_path + '\\info_{}_{}.data'.format(island, country), 'wb')
    pickle.dump(island_info, fw)
    fw.close()

    return island_info