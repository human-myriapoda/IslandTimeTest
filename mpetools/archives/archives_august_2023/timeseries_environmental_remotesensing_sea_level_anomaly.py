"""
This module calculates time series of sea level anomaly from remote sensing (Copernicus) for a given island.
TODO: remove seasons?
Citation: Copernicus Climate Change Service (C3S) (2017): ERA5: Fifth generation of ECMWF atmospheric reanalyses of the global climate. Copernicus Climate Change Service Climate Data Store (CDS), (date of access), https://cds.climate.copernicus.eu/cdsapp#!/home
NOTE: inspired by https://towardsdatascience.com/read-era5-directly-into-memory-with-python-511a2740bba0

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
from mpetools import get_info_islands
import os
import pickle
import pandas as pd
import xarray as xr
import numpy as np

class TimeSeriesSeaLevelAnomaly:
    def __init__(self, island, country, verbose_init=True, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands'), overwrite=False):
        self.island = island
        self.country = country
        self.verbose_init = verbose_init
        self.island_info_path = island_info_path
        self.overwrite = overwrite

    def assign_metadata(self):

        # Add description of this database
        self.island_info['timeseries_sea_level_anomaly']['description'] = 'This module calculates time series of sea level anomaly from remote sensing (Copernicus) for a given island.'

        # Add description of this timeseries
        self.island_info['timeseries_sea_level_anomaly']['description_timeseries'] = 'Sea level anomaly in metres (m)'
        
        # Set units
        self.island_info['timeseries_sea_level_anomaly']['units'] = {'sea_level_anomaly': 'm'}

        # Add source (paper or url)
        self.island_info['timeseries_sea_level_anomaly']['source'] = 'Copernicus Climate Change Service (C3S) (2017): ERA5: Fifth generation of ECMWF atmospheric reanalyses of the global climate. Copernicus Climate Change Service Climate Data Store (CDS), (date of access), https://cds.climate.copernicus.eu/cdsapp#!/home'

    def get_timeseries(self):

        print('~ Retrieving time series. ~')

        # Retrieve lat/lon of interest
        latitude, longitude = self.island_info['spatial_reference']['latitude'], self.island_info['spatial_reference']['longitude']

        # Retrieve all Copernicus files (sea level gridded data)
        files = os.listdir(os.path.join(os.getcwd(), 'data', 'copernicus_data'))

        # Create empty lists
        date_list = []
        sla_list = []

        # Loop through all files
        for idx, file in enumerate(files):
            if file.endswith('.nc'):

                # Open file as xarray dataset
                ds = xr.open_dataset(os.path.join(os.getcwd(), 'data', 'copernicus_data', file))
                time = ds['time']

                # Retrieve index for given lat/lon (one time only)
                if idx == 0:

                    # Find the index of the grid point nearest a specific lat/lon   
                    abslat = np.abs(ds.latitude - latitude)
                    abslon = np.abs(ds.longitude - longitude)
                    idx_grid = np.maximum(abslon, abslat)
                    ([xloc], [yloc]) = np.where(idx_grid == np.min(idx_grid))

                # Retrieve the closest grid to the islad
                point_ds = ds.isel({'longitude': xloc, 'latitude': yloc})

                # Append list with values
                sla_list.append(point_ds.sla.values[0])
                date_list.append(time.values[0])

        # Create DataFrame
        df_sea_level_anomaly = pd.DataFrame({'datetime': date_list, 'sea_level_anomaly': sla_list})
        df_sea_level_anomaly = df_sea_level_anomaly.set_index('datetime')
        
        # Save information in dictionary
        df_sea_level_anomaly = df_sea_level_anomaly.apply(pd.to_numeric)
        self.island_info['timeseries_sea_level_anomaly']['timeseries'] = df_sea_level_anomaly
        
    def main(self):

        # Retrieve the dictionary with currently available information about the island
        self.island_info = get_info_islands.retrieve_info_island(self.island, self.country, verbose=self.verbose_init)

        print('\n-------------------------------------------------------------------')
        print('RETRIEVING SEA LEVEL ANOMALY (Copernicus) DATA')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        # If sea_level_anomaly data have NOT already been generated
        if not 'timeseries_sea_level_anomaly' in self.island_info.keys() or self.overwrite:

            # Create key/dict for sea_level_anomaly data
            self.island_info['timeseries_sea_level_anomaly'] = {}
            self.assign_metadata()

            # Run all functions
            self.get_timeseries()
        
        # If sea_level_anomaly data have already been generated
        else:
            print('~ Information already available. Returning data. ~')

        # Save dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)
        
        return self.island_info