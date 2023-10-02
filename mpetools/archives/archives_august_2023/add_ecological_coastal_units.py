"""
This module retrieves Ecological Coastal Units (ECUs) and add the information for each transect.
Source: https://www.esri.com/arcgis-blog/products/arcgis-living-atlas/mapping/ecus-available/
TODO: retrieve reference shoreline and transects if not already available

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

from mpetools import get_info_islands, timeseries_environmental_remotesensing_CoastSat
import os
import pickle
import geopandas as gpd
import shapely
import numpy as np

class AddEcologicalCoastalUnits:
    def __init__(self, island, country, verbose_init=True, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands'), ECU_path=os.path.join(os.getcwd(), 'data', 'ECU'), overwrite=False):
        self.island = island
        self.country = country
        self.verbose_init = verbose_init
        self.island_info_path = island_info_path
        self.ECU_path = ECU_path
        self.overwrite = overwrite
        self.dict_rename = {'MEAN_SIG_W': 'Mean Significant Wave Height', 
                            'TIDAL_RANG': 'Tidal Range', 
                            'CHLOROPHYL': 'Chlorophyll-a', 
                            'TURBIDITY': 'Turbidity',
                            'TEMP_MOIST': 'Temperature Moist',
                            'EMU_PHYSIC': 'EMU Physical',
                            'REGIONAL_S': 'Regional Sinuosity',
                            'MAX_SLOPE': 'Max Slope',
                            'OUTFLOW_DE': 'Outflow Density',
                            'ERODIBILIT': 'Erodibility',
                            'LENGTH_GEO': 'Geographical Length',
                            'chl_label': 'Chlorophyll-a Descriptor',
                            'river_labe': 'River Descriptor',
                            'sinuosity_': 'Sinuosity Descriptor',
                            'slope_labe': 'Slope Descriptor',
                            'tidal_labe': 'Tidal Descriptor',
                            'turbid_lab': 'Turbidity Descriptor',
                            'wave_label': 'Wave Descriptor',
                            'CSU_Descri': 'CSU Descriptor',
                            'CSU_ID': 'CSU ID',
                            'OUTFLOW__1': 'Outflow Density Rescaled',
                            'Shape_Leng': 'Shape Length',
                            'geometry': 'Geometry'}

    def add_info(self):

        print('~ Retrieving Ecological Coastal Units. ~')
        
        # Retrieve transects
        if 'transects' in self.island_info['spatial_reference'].keys():
            transects = self.island_info['spatial_reference']['transects']
        else:
            reference_shoreline, transects = timeseries_environmental_remotesensing_CoastSat.TimeSeriesCoastSat(self.island, self.country, \
                                                                                              date_range=['2021-01-01', '2021-03-01'], \
                                                                                              verbose_init=False, reference_shoreline_transects_only=True).main()

        # Read ECU shapefile
        shapefile_ECU = os.path.join(self.ECU_path, 'ECU_{}_shapefile.shp'.format(self.country))
        gdf_ECU = gpd.read_file(shapefile_ECU)
        geometry_ECU = gdf_ECU.geometry

        # Create empty dictionary for transect ECU characteristics
        transects_ECU_characteristics = {}

        for key in transects.keys():
            transect = transects[key]
            linestring_transect = shapely.geometry.LineString(transect)

            # Retrieve ECU for this transect
            try:

                # Index of ECU that intersects with transect
                idx_transect_ECU = np.argwhere(np.array([linestring_transect.intersects(linestring_ECU) for linestring_ECU in geometry_ECU])).flatten()[0]

                # Get corresponding ECU
                df_transect_ECU = gdf_ECU[gdf_ECU.index==idx_transect_ECU].set_index('MasterKey')

            # ECU not available for this transect    
            except:
                continue

            # Rename columns for aesthetics
            df_transect_ECU = df_transect_ECU.rename(columns=self.dict_rename)

            # Save information in dictionary
            transects_ECU_characteristics[key] = df_transect_ECU
        
        # Save information in dictionary
        self.island_info['spatial_reference']['reference_shoreline'] = reference_shoreline
        self.island_info['spatial_reference']['transects'] = transects
        self.island_info['spatial_reference']['transects_ECU_characteristics'] = transects_ECU_characteristics

    def main(self):

        # Retrieve the dictionary with currently available information about the island
        self.island_info = get_info_islands.retrieve_info_island(self.island, self.country, verbose=self.verbose_init)

        print('\n-------------------------------------------------------------------')
        print('CHARACTERISING TRANSECTS WITH ECOLOGICAL COASTAL UNITS (ECUs)')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        # If ECUs have NOT already been extracted
        if not 'transects_ECU_characteristics' in self.island_info['spatial_reference'].keys() or self.overwrite:

            # Retrieve information about ECUs
            self.add_info()
        
        # If ECUs have already been extracted
        else:
            print('~ Information already available. Returning data. ~')

        # Save dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)

        return self.island_info