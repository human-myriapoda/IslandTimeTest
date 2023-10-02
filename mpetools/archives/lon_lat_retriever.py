"""
This module contains all the functions needed for extracting the latitude and longitude of any
given island (by its name).

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

def coordinatesRetrieve_kml(country, file_kml, coordinates_path=os.getcwd()+'\\data\\coordinates_geometry'):

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
        dict_kml['country'] = country
        dict_kml['island'] = islands[i]
        dict_kml['latitude'] = coord[i].split(',')[1]
        dict_kml['longitude'] = coord[i].split(',')[0]

        # Save a .data file
        fw = open(coordinates_path+'/coordinates_{}_{}.data'.format(islands[i], country), 'wb')
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
    pattern_lon = '"lon":(.*)},"wg'

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

def coordinatesConsensus(island, country, source='GeoKeo'):

    # Extract the coordinates from Wikipedia
    lat_w, lon_w = coordinatesRetrieve_wikipedia(island)
    print(lat_w, lon_w)

    # Extract the coordinates from GeoKeo
    lat_gk, lon_gk = coordinatesRetrieve_geokeo(island, country)
    print(lat_gk, lon_gk)

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

def coordinatesRetrieve(island, country, coordinates_path=os.getcwd()+'\\data\\coordinates_geometry', overwrite=False):
    
    print('Retrieving coordinates for', island, country, ':)')

    # If the path in which the data will be stored doesn't exist, we create it
    if not os.path.exists(coordinates_path): os.makedirs(coordinates_path)

    # Check if the coordinates have already been extracted
    if os.path.isfile(coordinates_path+'\\info_{}_{}.data'.format(island, country)) and not overwrite:

        print("Coordinates have already been extracted, let's load them!")

        # Load the .data file with pickle
        fw = open(coordinates_path+'\\info_{}_{}.data'.format(island, country), 'rb')
        dict_coordinates = pickle.load(fw)
    
    # If not, we continue with the code :)
    else:
        print('We are extracting the coordinates for the first time or recalculating them!')
        
        # We create an empty dictionary
        dict_coordinates = {}
        
        # We put some info in the dictionary
        dict_coordinates['island'] = island
        dict_coordinates['country'] = country

        # We call the function `coordinates_consensus` to retrieve lat, lon
        lat, lon = coordinatesConsensus(island, country)

        # We save this info in the dictionary
        dict_coordinates['latitude'] = float(lat)
        dict_coordinates['longitude'] = float(lon)

        # Put an empty value for geometry, which can be calculated with mpetools.region_geometry_calculator
        dict_coordinates['geometry'] = None

        fw = open(coordinates_path+'\\info_{}_{}.data'.format(island, country), 'wb')
        pickle.dump(dict_coordinates, fw)
        fw.close()

    return dict_coordinates
