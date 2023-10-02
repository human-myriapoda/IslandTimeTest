"""
This module allows us to retrieve data from CoastSat (https://github.com/kvos/CoastSat).
TODO: tidal correction

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import os
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate
from coastsatmaster.coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects
from mpetools import get_info_islands
import osmnx as ox
import shapely
import pyproj

# Other commands
plt.ion()
warnings.filterwarnings("ignore")

class TimeSeriesCoastSat:
    def __init__(self, island, country, date_range, sat_list=['L5', 'L7', 'L8', 'L9', 'S2'], collection='C02', verbose_init=True, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands'), coastsat_data_path=os.path.join(os.getcwd(), 'data', 'coastsat'), overwrite=False, plot_results=True, distance_between_transects=100, length_transect=250, reference_shoreline_transects_only=False):
        self.island = island
        self.country = country
        self.date_range = date_range
        self.sat_list = sat_list
        self.collection = collection
        self.verbose_init = verbose_init
        self.island_info_path = island_info_path
        self.coastsat_data_path = coastsat_data_path
        self.overwrite = overwrite
        self.plot_results = plot_results
        self.distance_between_transects = distance_between_transects
        self.length_transect = length_transect
        self.reference_shoreline_transects_only = reference_shoreline_transects_only

    def assign_metadata(self):

        # Add description of this database
        self.island_info['timeseries_coastsat']['description'] = 'This module allows us to retrieve time series of coastline position along transects using CoastSat.'

        # Add description of this timeseries
        self.island_info['timeseries_coastsat']['description_timeseries'] = 'TODO'

        # Add source (paper or url)
        self.island_info['timeseries_coastsat']['source'] = 'https://github.com/kvos/CoastSat'

    def get_reference_shoreline(self, metadata, settings):

        print('~ Retrieving reference shoreline from OpenStreetMap or manually. ~')
        
        # Check if OSM data is available
        try:
            gdf_coastline = ox.features_from_place(', '.join([self.island, self.country]), {'natural': 'coastline'})
            print('OSM coastline data available for this island.')

        except:
            print('No OSM data available for this island, we will manually define the reference shoreline.')
            reference_shoreline = SDS_preprocess.get_reference_sl(metadata, settings)
        
            return reference_shoreline

        # Get geometry of the coastline
        gdf_coastline_geometry = np.array(gdf_coastline['geometry'])

        # Loop over the coordinates of the coastline
        for idx_coastline in range(len(gdf_coastline_geometry)):
            
            # If the geometry is a shapely.geometry.linestring.LineString
            if type(gdf_coastline_geometry[idx_coastline]) == shapely.geometry.linestring.LineString:
                x_coastline = np.array([gdf_coastline_geometry[idx_coastline].xy[0][i] for i in range(len(gdf_coastline_geometry[idx_coastline].xy[0]))])
                y_coastline = np.array([gdf_coastline_geometry[idx_coastline].xy[1][i] for i in range(len(gdf_coastline_geometry[idx_coastline].xy[1]))])

            # If the geometry is a shapely.geometry.polygon.Polygon
            elif type(gdf_coastline_geometry[idx_coastline]) == shapely.geometry.polygon.Polygon:
                x_coastline = np.array([gdf_coastline_geometry[idx_coastline].exterior.xy[0][i] for i in range(len(gdf_coastline_geometry[idx_coastline].exterior.xy[0]))])
                y_coastline = np.array([gdf_coastline_geometry[idx_coastline].exterior.xy[1][i] for i in range(len(gdf_coastline_geometry[idx_coastline].exterior.xy[1]))])                

            # Interpolate between points to fill the gaps
            for pts in range(len(x_coastline) - 1):

                # Create a new array of points between the two points
                xx_coastline = np.array([x_coastline[pts], x_coastline[pts+1]])
                yy_coastline = np.array([y_coastline[pts], y_coastline[pts+1]])

                # Interpolate between the two points
                xx_coastline_linspace = np.linspace(xx_coastline[0], xx_coastline[1], 300)
                interpolation = interpolate.interp1d(xx_coastline, yy_coastline)

                # If first pair of points
                if pts == 0:
                    xx_coastline_full = xx_coastline_linspace
                    yy_coastline_full = interpolation(xx_coastline_linspace)

                # Concatenate with previous results
                else:
                    xx_coastline_full = np.concatenate((xx_coastline_full, xx_coastline_linspace))
                    yy_coastline_full = np.concatenate((yy_coastline_full, interpolation(xx_coastline_linspace)))

            # If first iteration
            if idx_coastline == 0:
                x_coastline_all = xx_coastline_full
                y_coastline_all = yy_coastline_full
            
            # Concatenate with previous results
            else:
                x_coastline_all = np.concatenate((x_coastline_all, xx_coastline_full))
                y_coastline_all = np.concatenate((y_coastline_all, yy_coastline_full))

        # Define the source and target coordinate systems
        src_crs = pyproj.CRS('EPSG:4326')
        tgt_crs = pyproj.CRS('EPSG:3857')

        # Create a transformer
        transformer = pyproj.Transformer.from_crs(src_crs, tgt_crs, always_xy=True)

        # Reproject the data
        x_reprojected, y_reprojected = transformer.transform(x_coastline_all, y_coastline_all)

        # Combine data into a numpy array  
        reference_shoreline = np.column_stack((x_reprojected, y_reprojected))

        return reference_shoreline

    def get_transects(self):

        print('~ Creating transects separated by {} m. ~'.format(self.distance_between_transects))

        # Retrieve reference shoreline
        reference_shoreline = self.island_info['spatial_reference']['reference_shoreline']

        # Double the coordinates of the reference shoreline (to ensure cyclic boundary conditions)
        reference_shoreline_cyclic = np.row_stack((np.flip(reference_shoreline, axis=0), reference_shoreline))

        # Create a shapely.geometry.Polygon with the reference shoreline
        polygon_shoreline = shapely.geometry.Polygon(reference_shoreline)

        # Create empty dictionary for transects
        transects = {}

        # Starting conditions for the loop
        idx_equidistant_transects = []
        start_point_transect = 0

        # Loop to find indices for equidistant transects
        while start_point_transect < (len(reference_shoreline) - 1):

            idx_equidistant_transects.append(start_point_transect)
            idx_distance = start_point_transect + 1

            # Calculate the distance between current and next points
            distance_between_points = np.hypot((reference_shoreline[start_point_transect, 0] - reference_shoreline[idx_distance, 0]),\
                                                (reference_shoreline[start_point_transect, 1] - reference_shoreline[idx_distance, 1]))
            
            # Continue to check distances until threshold is met
            while distance_between_points < self.distance_between_transects:
                idx_distance += 1

                if idx_distance >= (len(reference_shoreline) - 1):
                    break
                
                # Recalculate distance
                distance_between_points = np.hypot((reference_shoreline[start_point_transect, 0] - reference_shoreline[idx_distance, 0]),\
                                                    (reference_shoreline[start_point_transect, 1] - reference_shoreline[idx_distance, 1]))
    
            start_point_transect = idx_distance        
        
        # Convert list to array
        idx_equidistant_transects = np.array(idx_equidistant_transects)

        # Useful recurrent variables for the loop
        min_x = np.min(reference_shoreline_cyclic[:, 0])
        max_x = np.max(reference_shoreline_cyclic[:, 0])
        range_min_max = np.linspace(min_x - 0.5*(max_x - min_x), max_x + 0.5*(max_x - min_x), 5000)

        # Loop over the number of transects
        for idx_transect, idx_coords in enumerate(int(len(reference_shoreline_cyclic)/2) + idx_equidistant_transects):  

            # Get an array of 4 points around the coordinates of interest
            x_around = reference_shoreline_cyclic[(idx_coords-2):(idx_coords+4), 0]
            y_around = reference_shoreline_cyclic[(idx_coords-2):(idx_coords+4), 1]

            # Fit an affine function to the 4 points
            m_shoreline, _ = np.polyfit(x_around, y_around, 1)

            # Get index for middle point
            idx_middle_point = int(len(x_around)/2 - 0.5)

            # Get perpendicular affine function
            m_perpendicular = -1 / m_shoreline
            b_perpendicular = y_around[idx_middle_point] - m_perpendicular * x_around[idx_middle_point]

            # Create array of perpendicular coordinates (using the affine function)
            x_perpendicular = range_min_max
            y_perpendicular = m_perpendicular * x_perpendicular + b_perpendicular

            # Create an array of shapely.geometry.Point with the perpendicular coordinates
            points_perpendicular = [shapely.geometry.Point(x, y) for (x, y) in zip(x_perpendicular, y_perpendicular)]

            # Indices inside the polygon (island)
            idx_inside_polygon = np.argwhere(polygon_shoreline.contains(points_perpendicular)).flatten()

            if len(idx_inside_polygon) < 2:
                continue

            # Determine the direction of the perpendicular vector by calculating the distance between the middle point and the first and last point inside the polygon
            first_idx_inside_polygon, last_idx_inside_polygon = idx_inside_polygon[0], idx_inside_polygon[-1]
            dist_first_idx_inside_polygon = np.hypot(x_perpendicular[first_idx_inside_polygon] - x_around[idx_middle_point], y_perpendicular[first_idx_inside_polygon] - y_around[idx_middle_point])
            dist_last_idx_inside_polygon = np.hypot(x_perpendicular[last_idx_inside_polygon] - x_around[idx_middle_point], y_perpendicular[last_idx_inside_polygon] - y_around[idx_middle_point])
            
            # Distance between the middle point and the other side of the island
            max_distance_inside_polygon = np.max([dist_first_idx_inside_polygon, dist_last_idx_inside_polygon])

            if dist_first_idx_inside_polygon < dist_last_idx_inside_polygon:

                # To avoid the transect to be too close to the other side island, we take a reduced number of points inside the polygon
                if (max_distance_inside_polygon/2) < self.length_transect:
                    effective_length_transect = int(max_distance_inside_polygon/2)

                else:
                    effective_length_transect = self.length_transect

                # Reduce the number of points inside the polygon (island)
                idx_inside_polygon_reduced = idx_inside_polygon[np.argmin(np.abs(np.hypot(x_perpendicular[idx_inside_polygon] - x_around[idx_middle_point], \
                                                                                          y_perpendicular[idx_inside_polygon] - y_around[idx_middle_point]) \
                                                                                         - effective_length_transect))]

                # Number of points outside of the polygon (island)
                idx_outside_polygon_reduced = np.argmin(np.abs(np.hypot(x_perpendicular[:idx_inside_polygon[0]] - x_around[idx_middle_point], \
                                                                                          y_perpendicular[:idx_inside_polygon[0]] - y_around[idx_middle_point]) \
                                                                                         - self.length_transect))

                # Get vectors including the same distance outside of the polygon (island)
                x_transect = x_perpendicular[idx_outside_polygon_reduced:idx_inside_polygon_reduced] 
                y_transect = y_perpendicular[idx_outside_polygon_reduced:idx_inside_polygon_reduced]
                vector_transect = np.column_stack((x_transect, y_transect))

            else:

                # Flip the order of indices inside the polygon
                idx_inside_polygon = np.flip(idx_inside_polygon)

                # To avoid the transect to be too close to the other side island, we take a reduced number of points inside the polygon
                if (max_distance_inside_polygon/2) < self.length_transect:
                    effective_length_transect = int(max_distance_inside_polygon/2)

                else:
                    effective_length_transect = self.length_transect

                # Reduce the number of points inside the polygon (island)
                idx_inside_polygon_reduced = idx_inside_polygon[np.argmin(np.abs(np.hypot(x_perpendicular[idx_inside_polygon] - x_around[idx_middle_point], \
                                                                                          y_perpendicular[idx_inside_polygon] - y_around[idx_middle_point]) \
                                                                                         - effective_length_transect))]

                # Number of points outside of the polygon (island)
                idx_outside_polygon_reduced = np.argmin(np.abs(np.hypot(x_perpendicular[idx_inside_polygon[0]+1:] - x_around[idx_middle_point], \
                                                                                          y_perpendicular[idx_inside_polygon[0]+1:] - y_around[idx_middle_point]) \
                                                                                         - self.length_transect))
                
                # Get vectors including the same distance outside of the polygon (island)
                x_transect = x_perpendicular[idx_inside_polygon_reduced:idx_inside_polygon[0]+1+idx_outside_polygon_reduced] 
                y_transect = y_perpendicular[idx_inside_polygon_reduced:idx_inside_polygon[0]+1+idx_outside_polygon_reduced]
                vector_transect = np.column_stack((x_transect, y_transect))

            # Get the first and last point of the transect
            transect_inside_outside = vector_transect[[0, -1], :]

            # To make sure that the transect is oriented from the island to the ocean (inside -> outside)
            if polygon_shoreline.contains(shapely.geometry.Point(transect_inside_outside[0, 0], transect_inside_outside[0, 1])):
                transects[idx_transect] = transect_inside_outside
            else:
                transects[idx_transect] = np.flip(transect_inside_outside, axis=0)
        
        # Save transects in dictionary
        self.island_info['spatial_reference']['transects'] = transects

    def get_timeseries(self):

        # Define ara of interest
        polygon = [self.island_info['spatial_reference']['polygon'].getInfo()['coordinates'][0]]

        # Define date range
        dates = self.date_range

        # Define site name
        sitename = '_'.join([self.island, self.country])

        # Define path to save data
        filepath_data = self.coastsat_data_path

        # Define list of satellites
        sat_list = self.sat_list

        # Define collection
        collection = self.collection

        # Put all the inputs into a dictionnary
        inputs = {'polygon': polygon,
            'dates': dates,
            'sat_list': sat_list,
            'sitename': sitename,
            'filepath': filepath_data,
            'landsat_collection': collection}
        
        if self.reference_shoreline_transects_only:
            try:
                _ = ox.features_from_place(', '.join([self.island, self.country]), {'natural': 'coastline'})
                metadata = None
            
            except:
                # Check if data is available
                if os.path.exists(os.path.join(filepath_data, sitename)) and not self.overwrite:
                    metadata = SDS_download.get_metadata(inputs)
                
                # If data is not available
                else:
                    metadata = SDS_download.retrieve_images(inputs)
        else:
                # Check if data is available
                if os.path.exists(os.path.join(filepath_data, sitename)) and not self.overwrite:
                    metadata = SDS_download.get_metadata(inputs)
                
                # If data is not available
                else:
                    metadata = SDS_download.retrieve_images(inputs)            

        # Settings for shoreline extraction
        settings = { 
            # general parameters:
            'cloud_thresh': 0.5,        # threshold on maximum cloud cover
            'dist_clouds': 50,         # distance around clouds where shoreline can't be mapped
            'output_epsg': 3857,       # epsg code of spatial reference system desired for the output

            # quality control:
            'check_detection': False,    # if True, shows each shoreline detection to the user for validation
            'adjust_detection': False,  # if True, allows user to adjust the postion of each shoreline by changing the threhold
            'save_figure': True,        # if True, saves a figure showing the mapped shoreline for each image

            # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
            'min_beach_area': 1,     # minimum area (in metres^2) for an object to be labelled as a beach
            'min_length_sl': 1,       # minimum length (in metres) of shoreline perimeter to be valid
            'cloud_mask_issue': False,  # switch this parameter to True if sand pixels are masked (in black) on many images  
            'sand_color': 'default',    # 'default', 'latest', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
            'pan_off': False,           # True to switch pansharpening off for Landsat 7/8/9 imagery
            'max_dist_ref': 100,         # maximum distance (in pixels) between a valid shoreline and the reference shoreline

            # add the inputs defined previously
            'inputs': inputs,
        }

        if not self.reference_shoreline_transects_only:

            # Save .jpg of the preprocessed RGB images
            if not os.path.exists(os.path.join(filepath_data, sitename, 'jpg_files')) and not self.overwrite:
                SDS_preprocess.save_jpg(metadata, settings, use_matplotlib=True)

        # Get reference shoreline
        reference_shoreline = self.get_reference_shoreline(metadata, settings)
        settings['reference_shoreline'] = reference_shoreline

        # Save information in dictionary
        self.island_info['spatial_reference']['reference_shoreline'] = reference_shoreline
        self.island_info['timeseries_coastsat']['inputs'] = inputs
        self.island_info['timeseries_coastsat']['settings'] = settings

        if not self.reference_shoreline_transects_only:

            # Output file
            file_output = os.path.join(filepath_data, sitename, sitename + '_output.pkl')

            # Check if shoreline positions have already been extracted
            if os.path.exists(file_output) and not self.overwrite:

                with open(file_output, 'rb') as f:
                    output = pickle.load(f)

            # Extract shoreline positions
            else:
                output = SDS_shoreline.extract_shorelines(metadata, settings)

            # Removes duplicates (images taken on the same date by the same satellite)
            output = SDS_tools.remove_duplicates(output) 

            # Remove inaccurate georeferencing (set threshold to 10 m)
            output = SDS_tools.remove_inaccurate_georef(output, 10) 

            # Plot mapped shorelines
            if self.plot_results:

                # Define figure
                fig = plt.figure(figsize=[15, 8])

                # Plot every shoreline
                for i in range(len(output['shorelines'])):
                    sl = output['shorelines'][i]
                    date = output['dates'][i]
                    plt.plot(sl[:, 0], sl[:, 1], '.', label=date.strftime('%d-%m-%Y'))

                # Aesthetic parameters
                plt.legend()
                plt.axis('equal')
                plt.xlabel('Eastings')
                plt.ylabel('Northings')
                plt.grid(linestyle=':', color='0.5')
                plt.show(block=False)
        
        # Get transects
        if 'transects' in self.island_info['spatial_reference'].keys() and not self.overwrite:
            transects = self.island_info['spatial_reference']['transects']
        
        else:
            self.get_transects()
            transects = self.island_info['spatial_reference']['transects']

        if not self.reference_shoreline_transects_only:
            
            # Along-shore distance over which to consider shoreline points to compute the median intersection
            settings_transects = {'along_dist': 25}
            cross_distance = SDS_transects.compute_intersection(output, transects, settings_transects) 

            # Remove outliers
            settings_outliers = {'max_cross_change': 40,             # maximum cross-shore change observable between consecutive timesteps
                                'otsu_threshold': [-0.5, 0],        # min and max intensity threshold use for contouring the shoreline
                                'plot_fig': False}           # whether to plot the intermediate steps
                                
            cross_distance = SDS_transects.reject_outliers(cross_distance, output, settings_outliers)        

            # Create a dictionary with results
            dict_timeseries = {'datetime': output['dates']}

            # Loop over transects
            for key in cross_distance.keys():
                dict_timeseries[key] = cross_distance[key]

            # Create and save DataFrame
            df_timeseries = pd.DataFrame(dict_timeseries).set_index('datetime')
            fn = os.path.join(settings['inputs']['filepath'], settings['inputs']['sitename'], 'transect_time_series.csv')
            df_timeseries.to_csv(fn, sep=',')
            
            # Save information in dictionary
            self.island_info['timeseries_coastsat']['timeseries'] = df_timeseries
        
    def main(self):

        # Retrieve the dictionary with currently available information about the island
        self.island_info = get_info_islands.retrieve_info_island(self.island, self.country, verbose=self.verbose_init)

        print('\n-------------------------------------------------------------------')
        print('RETRIEVING COASTLINE POSITION (CoastSat) DATA')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        # If coastsat data have NOT already been generated
        if not 'timeseries_coastsat' in self.island_info.keys() or self.overwrite:

            # Create key/dict for coastsat data
            self.island_info['timeseries_coastsat'] = {}
            self.assign_metadata()

            # Run all functions
            self.get_timeseries()
        
        # If coastsat data have already been generated
        else:
            print('~ Information already available. Returning data. ~')

        # Save dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)

        if self.reference_shoreline_transects_only:
            return self.island_info['spatial_reference']['reference_shoreline'], self.island_info['spatial_reference']['transects']
        
        else:
            return self.island_info