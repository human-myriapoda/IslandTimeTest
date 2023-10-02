"""
This module calculates time series of vegetation health (NDVI) from remote sensing for a given island.
TODO: Otsu thresholding
TODO: Cloud masking
TODO: Vegetation health around the coast (distance threshold)

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
from mpetools import get_info_islands, timeseries_environmental_remotesensing_CoastSat
import ee
import os
import pickle
import pandas as pd
import numpy as np
import datetime
import shapely
import pyproj
import matplotlib.pyplot as plt
from tqdm import tqdm

class TimeSeriesVegetation:
    def __init__(self, island, country, verbose_init=True, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands'), overwrite=False):
        self.island = island
        self.country = country
        self.verbose_init = verbose_init
        self.island_info_path = island_info_path
        self.overwrite = overwrite

    def assign_metadata(self):

        # Add description of this database
        self.island_info['timeseries_vegetation']['description'] = 'This module calculates time series of vegetation health (NDVI) from remote sensing for a given island.'
        
        # Add description of this timeseries
        self.island_info['timeseries_vegetation']['description_timeseries'] = 'TODO'

        # Add source (paper or url)
        self.island_info['timeseries_vegetation']['source'] = 'This work'

    def get_bands(self, satellite_name, colour_list, mode=None):

        # If single colour, convert it to a list
        if type(colour_list) is not list: colour_list = [colour_list]

        # Read band data for the satellite of interest
        bands_df = pd.read_excel(os.path.join(os.getcwd(), 'data', 'info_satellite', 'bands_satellite.xlsx'), sheet_name=satellite_name)

        if mode == 'help':
            print('Select colours from:\n')
            print(bands_df)
            return

        try:
            list_bands_to_colours = [bands_df.band[bands_df.description == colour].item() for colour in colour_list]
            list_spatial_resolutions = [bands_df.spatial_resolution[bands_df.description == colour].item() for colour in colour_list]

        except ValueError:
            print('One or multiple colours in the list are not available for', satellite_name)
            print('Please choose from the available bands:\n')
            print(bands_df)
            raise Exception('Bands not available')

        return list_bands_to_colours, list_spatial_resolutions

    def create_vegetation_masks(self, sat):

        if all(element in self.island_info['timeseries_vegetation'].keys() for element in ['mask_total_vegetation_{}'.format(sat), 'mask_coastal_vegetation_{}'.format(sat), 'mask_transects_vegetation_{}'.format(sat)]):
            print('~ Vegetation masks already available for this satellite. Returning data. ~')
            return self.island_info['timeseries_vegetation']['mask_total_vegetation_{}'.format(sat)], self.island_info['timeseries_vegetation']['mask_coastal_vegetation_{}'.format(sat)], self.island_info['timeseries_vegetation']['mask_transects_vegetation_{}'.format(sat)]

        print('~ Creating vegetation masks. ~')
        
        # Retrieve transects, spatial extent and reference shoreline
        if 'transects' in self.island_info['spatial_reference'].keys():
            transects = self.island_info['spatial_reference']['transects']
            spatial_extent = np.array(self.island_info['spatial_reference']['polygon'].getInfo()['coordinates'][0])
            reference_shoreline = self.island_info['spatial_reference']['reference_shoreline']
        
        else:
            reference_shoreline, transects = timeseries_environmental_remotesensing_CoastSat.TimeSeriesCoastSat(self.island, self.country, \
                                                                                              date_range=['2021-01-01', '2021-03-01'], \
                                                                                              verbose_init=False, reference_shoreline_transects_only=True).main() 
            self.island_info['spatial_reference']['reference_shoreline'] = reference_shoreline
            self.island_info['spatial_reference']['transects'] = transects 
        
        # Reproject the spatial extent
        src_crs = pyproj.CRS('EPSG:4326')
        tgt_crs = pyproj.CRS('EPSG:3857')
        transformer = pyproj.Transformer.from_crs(src_crs, tgt_crs, always_xy=True)
        x_reprojected, y_reprojected = transformer.transform(spatial_extent[:, 0], spatial_extent[:, 1])
        spatial_extent = np.column_stack((x_reprojected, y_reprojected))

        # Create shapely.geometry.Polygon and shapely.geometry.LineString objects
        polygon_reference_shoreline = shapely.geometry.Polygon(reference_shoreline)
        linestring_reference_shoreline = shapely.geometry.LineString(reference_shoreline)

        # Calculate the bounding box of the polygon
        min_x, min_y = np.min(spatial_extent, axis=0)
        max_x, max_y = np.max(spatial_extent, axis=0)

        # Define the grid parameters
        if sat == 'S2':
            grid_resolution = self.island_info['image_collection_dict'][sat].select('B4').projection().nominalScale().getInfo()  # Adjust this to control the grid resolution
        else:
            grid_resolution = self.island_info['image_collection_dict'][sat].select('SR_B4').projection().nominalScale().getInfo()
            
        x_grid = np.arange(min_x, max_x, grid_resolution)
        y_grid = np.arange(min_y, max_y, grid_resolution)

        # Create a meshgrid from the x and y grid
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Create an empty mask array with the same shape as the meshgrid
        mask_total_vegetation, mask_coastal_vegetation = np.zeros(xx.shape, dtype=bool), np.zeros(xx.shape, dtype=bool)

        # Create dictionary for transect masks
        mask_transects_vegetation = {}
        for transect in transects.keys():
            mask_transects_vegetation[transect] = np.zeros(xx.shape, dtype=bool)

        # Fill masks
        for i in tqdm(range(xx.shape[0])):
            for j in range(xx.shape[1]):
                point = shapely.geometry.Point(xx[i, j], yy[i, j])

                # Total vegetation
                if polygon_reference_shoreline.contains(point):
                    mask_total_vegetation[i, j] = True
                                
                # Coastal vegetation
                if polygon_reference_shoreline.contains(point) and point.distance(linestring_reference_shoreline) < 100:
                    mask_coastal_vegetation[i, j] = True
                
                # Transect vegetation
                for transect in transects.keys():
                    linestring_transect = shapely.geometry.LineString(transects[transect])
                    if polygon_reference_shoreline.contains(point) and point.distance(linestring_reference_shoreline) < 100 and point.distance(linestring_transect) < 100:
                        mask_transects_vegetation[transect][i, j] = True
        
        # Save masks in dictionary
        self.island_info['timeseries_vegetation']['mask_total_vegetation_{}'.format(sat)] = mask_total_vegetation
        self.island_info['timeseries_vegetation']['mask_coastal_vegetation_{}'.format(sat)] = mask_coastal_vegetation
        self.island_info['timeseries_vegetation']['mask_transects_vegetation_{}'.format(sat)] = mask_transects_vegetation

        return mask_total_vegetation, mask_coastal_vegetation, mask_transects_vegetation
    
    def get_timeseries(self):
        
        # Retrieve polygon 
        polygon = self.island_info['spatial_reference']['polygon']
        
        # Loop in every satellite
        for idx_sat, sat in enumerate(self.island_info['image_collection_dict']):

            # Namely for cloud threshold
            if sat == 'description':
                print('Cloud thresold information:', self.island_info['image_collection_dict'][sat])
                continue
            
            else:
                print('Satellite:', sat)
                collection = self.island_info['image_collection_dict'][sat]
                bands_ndvi, spatial_res = self.get_bands(sat, ['NIR', 'Red'])
                spatial_res_ndvi = min(spatial_res)

            # Retrieve masks
            mask_total_vegetation, mask_coastal_vegetation, mask_transects_vegetation = self.create_vegetation_masks(sat)

            
            # Calculate the NDVI
            def calculate_ndvi(image):
                ndvi = image.normalizedDifference(bands_ndvi)
                return image.addBands(ndvi.rename('NDVI'))

            # Map the equation to the entire collection
            ndvi_collection = collection.map(calculate_ndvi)

            # Set the threshold value
            threshold = 0.3

            #masked_ndvi_collection = ndvi_collection.map(lambda image: image.updateMask(mask_total_vegetation))

            '''
            # Calculate the vegetation area for each image
            def calculate_vegetation_area(image):
                # Create a binary mask of vegetation/non-vegetation
                vegetation_mask = image.select('NDVI').gte(threshold)

                # Calculate the area of vegetation within the ROI
                vegetation_area = vegetation_mask.multiply(ee.Image.pixelArea()).reduceRegion(
                    reducer=ee.Reducer.sum(),
                    geometry=polygon,
                    scale=spatial_res_ndvi
                ).get('NDVI')

                # Return the image with an added property for vegetation area
                return image.set('vegetation_area', vegetation_area)

            # Map the function over the collection
            vegetation_area_collection = ndvi_collection.map(calculate_vegetation_area)
        
            # Extract the dates and vegetation areas from the image collection
            dates = vegetation_area_collection.aggregate_array('system:time_start')
            areas = vegetation_area_collection.aggregate_array('vegetation_area')

            # Convert the lists to numpy arrays for plotting
            dates = np.array(ee.List(dates).getInfo())
            dates = [datetime.datetime.utcfromtimestamp(date / 1000) for date in dates]
            areas = np.array(ee.List(areas).getInfo())

            if idx_sat == 1:
                df_vegetation_area = pd.DataFrame({'datetime': dates, 'vegetation_area_{}'.format(sat): areas})
                df_vegetation_area = df_vegetation_area.set_index('datetime')
            
            else:
                df_vegetation_area = pd.concat([df_vegetation_area, pd.DataFrame({'datetime': dates, 'vegetation_area_{}'.format(sat): areas}).set_index('datetime')], axis=1)

        df_vegetation_area = df_vegetation_area.apply(pd.to_numeric)
        self.island_info['timeseries_vegetation']['timeseries'] = df_vegetation_area
        '''
        
    def main(self):

        # Retrieve the dictionary with currently available information about the island
        self.island_info = get_info_islands.retrieve_info_island(self.island, self.country, verbose=self.verbose_init)

        print('\n-------------------------------------------------------------------')
        print('RETRIEVING ISLAND SIZE AND VEGETATION AREA DATA')
        print('Island:', ', '.join([self.island, self.country]))
        print('Satellites:', ' '.join(list(self.island_info['image_collection_dict'].keys())[1:]))
        print('-------------------------------------------------------------------\n')

        # If vegetation data have NOT already been generated
        if not 'timeseries_vegetation' in self.island_info.keys():

            # Create key/dict for vegetation data
            self.island_info['timeseries_vegetation'] = {}
            self.assign_metadata()

            # Run all functions
            self.get_timeseries()
        
        # If vegetation data have already been generated
        else:
            if self.overwrite:
                self.get_timeseries()
            else:
                print('~ Information already available. Returning data. ~')

        # Save dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)
        
        return self.island_info

'''
def getBands(satellite_name, colour_list, mode=None):

    # If single colour, convert it to a list
    if type(colour_list) is not list: colour_list = [colour_list]

    # Read band data for the satellite of interest
    bands_df = pd.read_excel(os.getcwd() + '\\data\\satellite_info\\bands_satellite.xlsx', sheet_name=satellite_name)

    if mode == 'help':
        print('Select colours from:\n')
        print(bands_df)
        return

    try:
        list_bands_to_colours = [bands_df.band[bands_df.description == colour].item() for colour in colour_list]

    except ValueError:
        print('One or multiple colours in the list are not available for', satellite_name)
        print('Please choose from the available bands:\n')
        print(bands_df)
        raise Exception('Bands not available')

    return list_bands_to_colours


def calculateSpectralIndex():
    pass

def calculateAreaVegetation(image):
    pass

def maskClouds(image, sat):
    
    # Get band
    cloud_mask_band = getBands(sat, 'Cloud mask')
    cloud_mask_band_image = image.select(cloud_mask_band)

    # Cloud Bit Mask
    cloud_bit_mask = ee.Number(2).pow(10).int()
    cirrus_bit_mask = ee.Number(2).pow(11).int()

    # Mask
    cloud_mask = cloud_mask_band_image.bitwiseAnd(cloud_bit_mask).eq(0) and (cloud_mask_band_image.bitwiseAnd(cirrus_bit_mask).eq(0))

    #Update image
    image.updateMask(cloud_mask)

    return image

def spectralIndex(index_name: str, image, sat: str):
    
    if index_name == 'NDWI':

        # Get bands
        bands = getBands(sat, ['Green', 'NIR'])
        mask_NDWI = image.normalizedDifference(bands).rename('NDWI')

        return image.addBands(mask_NDWI)

    elif index_name == 'NDVI':

        # Get bands
        bands = getBands(sat, ['NIR', 'Red'])
        mask_NDVI = image.normalizedDifference(bands).rename('NDVI')

        return image.addBands(mask_NDVI)

    elif index_name == 'EVI':

        # Get bands
        bands = getBands(sat, ['NIR', 'Red', 'Blue'])
        pass

def generateTimeSeriesAreaVegetation(island_info):

    print('\n-------------------------------------------------------------------')
    print('VEGETATION AREA TIME SERIES')
    print('Island:', ', '.join([island_info['general_info']['island'], island_info['general_info']['country']]))
    print('Satellites:', ' '.join(list(island_info['image_collection_dict'].keys())[1:]))
    print('-------------------------------------------------------------------\n')

    
    for (idx, sat) in enumerate(island_info['image_collection_dict']):
        
        # Temporary printing
        print(idx, sat)

        # Namely for cloud threshold
        if sat == 'Description':
            print('Cloud thresold information:', island_info['image_collection_dict']['Description'])
        
        # Sentinel-2
        if sat == 'L8':
            print('Landsat-8')
            imageCollection = island_info['image_collection_dict'][sat]
            #image = maskClouds(imageCollection.first(), sat)
            image = imageCollection.first().clip(island_info['geometry'])
            image = spectralIndex('NDVI', image, sat=sat)

            return image

'''