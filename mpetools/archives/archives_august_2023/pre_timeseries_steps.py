"""
This module allows us to retrieve all the information needed before generating time series.
1. From the name of the island, retrieve coordinates (latitude, longitude).
2. From coordinates, generate an approximate square region around the island.
3. Build a dictionary with ImageCollections for a list of satellites.

TODO: adapt it for weirdly-shaped islands.
TODO: implement Map return

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Load modules 
import numpy as np
import os
import urllib.request
import re
import requests
import pickle
import datetime
import ee
import osmnx as ox
import geojson
from wikidataintegrator import wdi_core

class PreTimeSeries:
    def __init__(self, island, country, alt_name=None, to_do=None, \
                 dict_geometry={'var_init': 0.001, 'var_change': 0.001, 'thresold_ndvi': 0.2, 'var_limit': 0.065}, \
                 list_sat_user=None, date_range_user=None, cloud_threshold=10, image_type='SR', method='new', relevant_properties_wikidata=['P361', 'P131', 'P206', 'P2044'], \
                 verbose=True, wikidata_id=None, overwrite=False):
        
        # Constructor (initialise instance variables)
        self.island = island
        self.country = country
        self.alt_name = alt_name
        self.to_do = to_do
        self.dict_geometry = dict_geometry
        self.list_sat_user = list_sat_user
        self.date_range_user = date_range_user
        self.cloud_threshold = cloud_threshold
        self.image_type = image_type
        self.method = method
        self.relevant_properties_wikidata = relevant_properties_wikidata
        self.verbose = verbose
        self.wikidata_id = wikidata_id
        self.overwrite = overwrite

    def retrieve_coordinates_wikipedia(self):
        # To fit Wikipedia's format
        place = self.island.replace(" ", "_")

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
        data_url = str(web_url.read())

        # Regex patterns for latitude and longitude
        pattern_lat = '"wgCoordinates":{"lat":(.*),"lon":'
        pattern_lon = '"lon":(.*)},"wg' #EditSubmit

        # Find the coordinates from the patterns
        try:
            lat = re.search(pattern_lat, data_url).group(1)
            lon = re.search(pattern_lon, data_url).group(1)
            lat = lat.replace('\\n', '')
            lon = lon.replace('\\n', '')

        # No patterns found
        except: 
            lat = np.nan
            lon = np.nan

        return lat, lon

    def retrieve_coordinates_geokeo(self):
        # To make sure we find the island and not another place with the same name
        place = "{}, {}".format(self.island, self.country)

        # Request url (from GeoKeo website)
        url_gk = 'https://geokeo.com/geocode/v1/search.php?q={}&api=YOUR_API_KEY'.format(place)
        resp = requests.get(url=url_gk)
        data_url = resp.json()

        # Retrieve the coordinates (from GeoKeo website)
        if 'status' in data_url:
            if data_url['status'] == 'ok':
                lat = data_url['results'][0]['geometry']['location']['lat']
                lon = data_url['results'][0]['geometry']['location']['lng']

            else:
                lat, lon = np.nan, np.nan

        else:
            lat, lon = np.nan, np.nan

        return lat, lon

    def coordinates_consensus(self):
        # Extract the coordinates from Wikipedia
        lat_w, lon_w = self.retrieve_coordinates_wikipedia()
        if self.verbose: 
            print('Coordinates from Wikipedia (lat/lon):', lat_w, lon_w)

        # Extract the coordinates from GeoKeo
        lat_gk, lon_gk = self.retrieve_coordinates_geokeo()
        if self.verbose: 
            print('Coordinates from GeoKeo (lat/lon):', lat_gk, lon_gk)

        # Compare the values and if they are too different -> visual inspection
        if not np.nan in (lat_w, lon_w, lat_gk, lon_gk):
            lat_close = np.allclose(np.array(lat_w, dtype=float), np.array(lat_gk, dtype=float), atol=1e-1)
            lon_close = np.allclose(np.array(lon_w, dtype=float), np.array(lon_gk, dtype=float), atol=1e-1)

            if not (lat_close and lon_close):
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

    def retrieve_coordinates(self):
        
        # New method (using OpenStreetMap)
        if self.method == 'new':
            try:
                area = ox.geocode_to_gdf(self.island + ', ' + self.country)
                lat, lon = area.lat[0], area.lon[0]

            except:
                # We call the function `coordinates_consensus` to retrieve lat, lon (old method)
                print('Island not available in OpenStreetMap. Will use other methods.')
                lat, lon = self.coordinates_consensus()
        
        else: 
            # We call the function `coordinates_consensus` to retrieve lat, lon (old method)
            lat, lon = self.coordinates_consensus()

        # We save this info in the dictionary
        self.island_info['spatial_reference']['latitude'] = float(lat)
        self.island_info['spatial_reference']['longitude'] = float(lon)

    def spectral_index_ndvi(self, image, geo_polygon, band1='B5', band2='B4'):
        # Get arrays from GEE image
        arr_band1 = np.array(image.sampleRectangle(geo_polygon).get(band1).getInfo())
        arr_band2 = np.array(image.sampleRectangle(geo_polygon).get(band2).getInfo())

        # Equation for NDVI
        arr_ndvi = (arr_band1 - arr_band2)/(arr_band1 + arr_band2)

        return arr_ndvi

    def create_polygon(self, key, lat, lon, var):
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

    def calculate_geometry_using_spectral_indexes(self):
        # Extract dict_geometry
        var_init = self.dict_geometry['var_init']
        var_change = self.dict_geometry['var_change']
        thresold_NDVI = self.dict_geometry['thresold_ndvi']
        var_limit = self.dict_geometry['var_limit']

        # Specify a geometric point in GEE
        lat = float(self.island_info['spatial_reference']['latitude'])
        lon = float(self.island_info['spatial_reference']['longitude'])
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
            if self.verbose: 
                print('var = ', var)

            # For loop for every 'corner'
            for key in dict_coords.keys():               
                # Generate polygon for a given 'corner'
                geo_polygon = self.create_polygon(key, lat, lon, var)

                # Calculate NDVI for every pixel of that 'corner'
                arr_NDVI = self.spectral_index_ndvi(image, geo_polygon)

                # Create array with the NDVI values on the border of the 'corner'
                arr_borders = np.concatenate((arr_NDVI[dict_coords[key][1], :], arr_NDVI[:, dict_coords[key][2]]))

                # Test conditions
                bool_borders.append(np.any(arr_borders > thresold_NDVI))
                bool_full.append(np.all(arr_NDVI < thresold_NDVI))

            # If we reach the limit, the loop ends and we calculate the region as is
            if (var >= (var_limit - var_change)):  
                if self.verbose: 
                     print('Maximum limit reached, code will stop')

                geo_polygon_final = self.create_polygon(key='final', lat=lat, lon=lon, var=var)
                
                return geo_polygon_final

            # Still land on the borders -> we expand the region at the next iteration
            if np.any(np.array(bool_borders)):
                var += var_change

            # Only water -> no island -> we expand the region at the next iteration
            elif np.all(np.array(bool_full)):
                var += var_change

            # We found an island surrounded by water -> loop ends
            else:
                var += var_change
                if self.verbose: 
                    print('Done!')

                geo_polygon_final = self.create_polygon(key='final', lat=lat, lon=lon, var=var)
                
                return geo_polygon_final

    def calculate_geometry(self):
        # New method (OpenStreetMap)
        if self.method == 'new':
            try:
                area = ox.geocode_to_gdf(self.island + ', ' + self.country)

                polygon_square_OSM = ee.Geometry.Polygon(
                [[[area.bbox_west[0]-0.001, area.bbox_north[0]+0.001],
                    [area.bbox_west[0]-0.001, area.bbox_south[0]-0.001],
                    [area.bbox_east[0]+0.001, area.bbox_south[0]-0.001],
                    [area.bbox_east[0]+0.001, area.bbox_north[0]+0.001],
                    [area.bbox_west[0]-0.001, area.bbox_north[0]+0.001]]], None, False)      

                polygon_OSM_geojson = geojson.Polygon(list((area.geometry[0].exterior.coords)))
                polygon_OSM = ee.Geometry.Polygon(polygon_OSM_geojson['coordinates']) 

                self.island_info['spatial_reference']['polygon'] = polygon_square_OSM
                self.island_info['spatial_reference']['polygon_OSM'] = polygon_OSM

            # Old method
            except:               
                polygon_old_method = self.calculate_geometry_using_spectral_indexes()
                self.island_info['spatial_reference']['polygon'] = polygon_old_method

        # Old method
        else:
            polygon_old_method = self.calculate_geometry_using_spectral_indexes()
            self.island_info['spatial_reference']['polygon'] = polygon_old_method

    def get_satellite_availability(self):

        # Open satellite information from file
        file_path = os.path.join(os.getcwd(), 'data', 'info_satellite', 'dict_satellite.data')
        with open(file_path, 'rb') as fw:
            dict_satellite = pickle.load(fw)

        # Create geometry point
        point = ee.Geometry.Point([self.island_info['spatial_reference']['longitude'], self.island_info['spatial_reference']['latitude']])

        # Create date range
        if self.date_range_user is None and self.list_sat_user is None:
                date_range = [dict_satellite['L5']['SR dataset availability'][0], datetime.datetime.now()]

        elif self.date_range_user is None and self.list_sat_user is not None:
                novelty_index_sat = [dict_satellite[sat]['novelty index'] for sat in self.list_sat_user]
                min_idx, max_idx = novelty_index_sat.index(min(novelty_index_sat)), novelty_index_sat.index(max(novelty_index_sat))
                date_range = [dict_satellite[self.list_sat_user[min_idx]]['SR dataset availability'][0], dict_satellite[self.list_sat_user[max_idx]]['SR dataset availability'][1]]

        else: 
                date_range = self.date_range_user
                
        # Create satellite list
        if self.list_sat_user is None: 
             list_sat = list(dict_satellite.keys())

        else: 
             list_sat = self.list_sat_user

        # Create empty dictionary for filtered ImageCollection
        image_collection_dict = {'description': 'Filtered (cloud threshold of {}%) ImageCollection for satellites of interest'.format(self.cloud_threshold)}

        # Loop in the list of satellites
        for sat in list_sat:
                collection_sat = ee.ImageCollection(dict_satellite[sat]['{} GEE Snipper'.format(self.image_type)]) \
                                   .filterDate(date_range[0], date_range[1]) \
                                   .filterBounds(point) \
                                   .filterMetadata(dict_satellite[sat]['cloud label'], 'less_than', self.cloud_threshold)

                size_collection_sat = collection_sat.size().getInfo()

                if self.verbose: 
                     print(sat, size_collection_sat)

                if size_collection_sat > 0:
                        image_collection_dict[sat] = collection_sat

        self.island_info['image_collection_dict'] = image_collection_dict
    
    def get_other_info(self):

        place = self.island + ', ' + self.country

        try:
            osm_type, osm_id = ox.geocode_to_gdf(place).osm_type.values[0], ox.geocode_to_gdf(place).osm_id.values[0]
        
        except:
            print('No other information available.')

        if self.wikidata_id is None:
            url_other = 'https://www.openstreetmap.org/{}/{}'.format(osm_type, str(osm_id))
            web_url_other = urllib.request.urlopen(url_other)
            data_url_other = str(web_url_other.read())

            # Find wikidata ID
            pattern_wikidata_id = r'Q\d+'
            matches = re.findall(pattern_wikidata_id, data_url_other)
            wikidata_id = np.unique(np.array(matches))[0]

        else:
            wikidata_id = self.wikidata_id

        # Fetch information about the Wikidata item
        wikidata = wdi_core.WDItemEngine(wd_item_id=wikidata_id).get_wd_json_representation()

        # List of properties to retrieve
        list_properties = self.relevant_properties_wikidata

        # Loop through the properties
        for prop in list_properties:
            
            if prop in wikidata['claims'].keys():

                claim_prop = wikidata['claims'][prop]
                name_prop = wdi_core.WDItemEngine(wd_item_id=prop).get_label()

                if len(claim_prop) == 1:

                    try:
                        self.island_info['general_info'][name_prop] = wdi_core.WDItemEngine(wd_item_id=claim_prop[0]['mainsnak']['datavalue']['value']['id']).get_label()
                    except:
                        self.island_info['general_info'][name_prop] = claim_prop[0]['mainsnak']['datavalue']['value']['amount']

                else:
                    try:
                        self.island_info['general_info'][name_prop] = ', '.join([wdi_core.WDItemEngine(wd_item_id=claim_prop[lll]['mainsnak']['datavalue']['value']['id']).get_label() for lll in range(len(claim_prop))])

                    except:
                        self.island_info['general_info'][name_prop] = ', '.join([claim_prop[lll]['mainsnak']['datavalue']['value']['amount'] for lll in range(len(claim_prop))])
                        
            else:
                continue


    def main(self):
        print('\n-------------------------------------------------------------------')
        print('PRE-TIME-SERIES INFORMATION')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        # Define path for data
        island_info_path = os.path.join(os.getcwd(), 'data', 'info_islands')

        # If the path in which the data will be stored doesn't exist, we create it
        if not os.path.exists(island_info_path): 
            os.makedirs(island_info_path)

        # Check what information is already available
        info_file_path = os.path.join(island_info_path, 'info_{}_{}.data'.format(self.island, self.country))
        
        if os.path.isfile(info_file_path):
            # Load the .data file with pickle
            with open(info_file_path, 'rb') as f:
                self.island_info = pickle.load(f)

            # If actions are defined by the user, skip to steps
            if self.to_do is None:       
                self.to_do = {'coordinates': True, 'polygon': True, 'availability': True, 'other': True}

                # 1. Coordinates
                if self.island_info['spatial_reference']['latitude'] is not None and \
                    self.island_info['spatial_reference']['longitude'] is not None: 
                    self.to_do['coordinates'] = False

                # 2. Geometry
                if self.island_info['spatial_reference']['polygon'] is not None: 
                    self.to_do['polygon'] = False

                # 3. Image availability
                if self.island_info['image_collection_dict'] is not None: 
                    self.to_do['availability'] = False

                # If all available information is already available, return dictionary
                if not any(self.to_do.values()):
                    print('~ All information is already available, returning information. ~')
                    return self.island_info
                else: 
                    print('~ The following information will be extracted/calculated:', \
                          ' and '.join([key for (key, val) in self.to_do.items() if val]), '~\n')                
            else: 
                print('~ The user wishes to extract/calculate', ' and '.join([key for (key, val) in self.to_do.items() if val]), '~\n')

        # No file exists
        else: 
            print('~ All information will be extracted/calculated. ~\n')
            self.island_info = {'general_info': {'island': self.island, 'country': self.country}, \
                        'spatial_reference': {'latitude': None, 'longitude': None, 'polygon': None}, \
                        'image_collection_dict': None}
            
            if self.alt_name is not None:
                self.island_info['general_info']['alt_name'] = self.alt_name

            self.to_do = {'coordinates': True, 'polygon': True, 'availability': True, 'other': True}
        
        # (Re-)Calculating missing information
        for missing_info in [key for (key, val) in self.to_do.items() if val]:
            # Step 1: Extract coordinates (latitude, longitude)
            if missing_info == 'coordinates' or self.overwrite:
                self.retrieve_coordinates()

            # Step 2: Calculate geometry (ee.Geometry.Polygon)
            elif missing_info == 'polygon' or self.overwrite:
                self.calculate_geometry()

            # Step 3: Build a dictionary with satellite availability
            elif missing_info == 'availability' or self.overwrite:
                self.get_satellite_availability()

            # Step 4: Get other information (e.g. elevation, population, etc.)
            elif missing_info == 'other' or self.overwrite:
                self.get_other_info()

        # Save dictionary
        with open(info_file_path, 'wb') as f:
            pickle.dump(self.island_info, f)

        return self.island_info
