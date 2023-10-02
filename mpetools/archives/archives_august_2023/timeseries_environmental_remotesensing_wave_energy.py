"""
This module calculates time series of wave energy from remote sensing (ERA5) for a given island.
TODO: remove seasons?
Citation: Copernicus Climate Change Service (C3S) (2017): ERA5: Fifth generation of ECMWF atmospheric reanalyses of the global climate. Copernicus Climate Change Service Climate Data Store (CDS), (date of access), https://cds.climate.copernicus.eu/cdsapp#!/home
NOTE: inspired by https://towardsdatascience.com/read-era5-directly-into-memory-with-python-511a2740bba0
TODO: allow user to choose date range

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
from mpetools import get_info_islands
import os
import pickle
import pandas as pd
import xarray as xr
import cdsapi
import urllib.request
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta

class TimeSeriesWaveEnergy:
    def __init__(self, island, country, verbose_init=True, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands'), overwrite=False):
        self.island = island
        self.country = country
        self.verbose_init = verbose_init
        self.island_info_path = island_info_path
        self.overwrite = overwrite

    def assign_metadata(self):

        # Add description of this database
        self.island_info['timeseries_wave_energy']['description'] = 'This module calculates time series of wave energy from remote sensing (ERA5) for a given island.'

        # Add description of this timeseries
        self.island_info['timeseries_wave_energy']['description_timeseries'] = 'Wave energy flux in watts per metre of wave front'
        
        # Set units
        self.island_info['timeseries_wave_energy']['units'] = {'wave_energy': 'kW/m'}

        # Add source (paper or url)
        self.island_info['timeseries_wave_energy']['source'] = 'Copernicus Climate Change Service (C3S) (2017): ERA5: Fifth generation of ECMWF atmospheric reanalyses of the global climate. Copernicus Climate Change Service Climate Data Store (CDS), (date of access), https://cds.climate.copernicus.eu/cdsapp#!/home'

    def wave_energy_equation(self, significant_wave_height, mean_wave_period):
        
        # SI units
        rho =  1025 # kg/m^3
        g = 9.81 # m/s^2
        wave_energy = (rho * g **2 * significant_wave_height ** 2 * mean_wave_period) / (64 * np.pi)

        return wave_energy
    
    def get_ERA5_cdsapi(self, var_name, area, dates, grid=[0.5, 0.5], dataset_name='reanalysis-era5-single-levels-monthly-means'):

        # Query API
        cds = cdsapi.Client(url = "https://cds.climate.copernicus.eu/api/v2", key = "200721:d13b27b3-32f8-4315-a9c0-e65dc3eb6fdd")

        # Parameters
        params = dict(
            format = "netcdf",
            product_type = "reanalysis",
            variable = var_name,
            grid = grid,
            area = area,
            date = list(dates.strftime('%Y-%m-%d %H:%M')) \
               if isinstance(dates, pd.core.indexes.datetimes.DatetimeIndex)\
               else dates)

        # What to do if asking for monthly means
        # NOTE: taken from https://towardsdatascience.com/read-era5-directly-into-memory-with-python-511a2740bba0
        if dataset_name in ["reanalysis-era5-single-levels-monthly-means", 
                            "reanalysis-era5-pressure-levels-monthly-means",
                            "reanalysis-era5-land-monthly-means"]:
            params["product_type"] = "monthly_averaged_reanalysis"
            _ = params.pop("date")
            params["time"] = "00:00"
            
            # If time is in list of pandas format
            if isinstance(dates, list):
                dates_pd = pd.to_datetime(dates)
                params["year"] = sorted(list(set(dates_pd.strftime("%Y"))))
                params["month"] = sorted(list(set(dates_pd.strftime("%m"))))
            else:
                params["year"] = sorted(list(set(dates.strftime("%Y"))))
                params["month"] = sorted(list(set(dates.strftime("%m"))))
            
        # File object
        fl = cds.retrieve(dataset_name, params) 
        
        # Load into memory and return xarray dataset
        with urllib.request.urlopen(fl.location) as f:
            return xr.open_dataset(f.read())

    def get_timeseries(self):

        # Define area of interest
        polygon = self.island_info['spatial_reference']['polygon'].getInfo()['coordinates'][0]
        area = [polygon[0][0], polygon[0][1], polygon[2][0], polygon[2][1]]

        # Define date range
        def datetime_range(start_date, end_date):
            current_date = start_date
            while current_date < end_date:
                yield current_date
                current_date += relativedelta(months=1)

        start_date = datetime.datetime(2000, 1, 1)  # Start date
        end_date = datetime.datetime(2022, 12, 31)  # End date

        dates_dt = [date for date in datetime_range(start_date, end_date)]
        dates_str = [date.strftime('%Y-%m-%d') for date in datetime_range(start_date, end_date)]

        # Retrieve significant_wave_height
        ds_iews = self.get_ERA5_cdsapi(var_name='significant_wave_height', area=area, dates=dates_str)

        # Retrieve mean_wave_period
        ds_mwp = self.get_ERA5_cdsapi(var_name='mean_wave_period', area=area, dates=dates_str)

        # Get time series and flatten array
        mwp = ds_mwp.mwp.values.flatten()
        iews = ds_iews.iews.values.flatten()

        # Calculate wave energy
        wave_energy = self.wave_energy_equation(iews, mwp)

        # Create DataFrame
        df_wave_energy = pd.DataFrame({'wave_energy': wave_energy}, index=dates_dt)
        
        # Save information in dictionary
        df_wave_energy = df_wave_energy.apply(pd.to_numeric)
        self.island_info['timeseries_wave_energy']['timeseries'] = df_wave_energy
        
    def main(self):

        # Retrieve the dictionary with currently available information about the island
        self.island_info = get_info_islands.retrieve_info_island(self.island, self.country, verbose=self.verbose_init)

        print('\n-------------------------------------------------------------------')
        print('RETRIEVING WAVE ENERGY (ERA-5) DATA')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        # If wave_energy data have NOT already been generated
        if not 'timeseries_wave_energy' in self.island_info.keys() or self.overwrite:

            # Create key/dict for wave_energy data
            self.island_info['timeseries_wave_energy'] = {}
            self.assign_metadata()

            # Run all functions
            self.get_timeseries()
        
        # If wave_energy data have already been generated
        else:
            print('~ Information already available. Returning data. ~')

        # Save dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)
        
        return self.island_info
