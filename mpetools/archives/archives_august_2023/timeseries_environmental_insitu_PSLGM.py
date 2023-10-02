"""
This module allows us to retrieve data from the Pacific Sea Level and Geodetic Monitoring Project (Monthly Sea Level and Meteorological Statistics).
Available information: Monthly sea level, barometric pressure, water temperature and air temperature.
Source: http://www.bom.gov.au/oceanography/projects/spslcmp/data/monthly.shtml
TODO: skip if data is not available, better variable names

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import pandas as pd
import pickle
import os
from mpetools import get_info_islands
import numpy as np
import datetime

class TimeSeriesPSLGM:
    def __init__(self, island, country, verbose_init=True, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands'), PSLGM_path=os.path.join(os.getcwd(), 'data', 'PSLGM'), overwrite=False):
        self.island = island
        self.country = country
        self.verbose_init = verbose_init
        self.island_info_path = island_info_path
        self.PSLGM_path = PSLGM_path
        self.overwrite = overwrite

    def assign_metadata(self):

        # Add description of this database
        self.island_info['timeseries_PSLGM']['description'] = 'This module allows us to retrieve data from the Pacific Sea Level and Geodetic Monitoring Project (Monthly Sea Level and Meteorological Statistics).'
        
        # Add description of this timeseries
        self.island_info['timeseries_PSLGM']['description_timeseries'] = 'Available information: Monthly sea level, barometric pressure, water temperature and air temperature.'

        # Add source (paper or url)
        self.island_info['timeseries_PSLGM']['source'] = 'http://www.bom.gov.au/oceanography/projects/spslcmp/data/monthly.shtml'

    def get_timeseries(self):

        for idx_PSLGM, info in enumerate(['sea_level', 'barometric_pressure', 'water_temperature', 'air_temperature']):

            try:
                # Read data from file
                data_PSLGM = np.array(open(os.path.join(self.PSLGM_path, '{}_{}_{}.txt'.format(self.island, \
                                                                        self.country, \
                                                                        info)), 'r').readlines())
                
            except: 

                print('No PSLGM information for this island.')
                
                return self.island_info

            print("~ Retrieving {}. ~".format(info.replace('_', ' ').capitalize()))

            # Select rows corresponding to the data
            data_cleaned = np.array(data_PSLGM[np.argwhere(np.char.startswith(data_PSLGM, '      Mth'))[0][0] + 1 : \
                                        np.argwhere(np.char.startswith(data_PSLGM, '      Totals'))[0][0]])

            for row in range(len(data_cleaned)):

                row_cleaned = np.array(data_cleaned[row].replace('\n', '').split(' '))
                row_fully_cleaned = row_cleaned[row_cleaned != ''].astype(float)

                # Expected length is 8 (otherwise there is missing data)
                if len(row_fully_cleaned) != 8: continue

                # First iteration
                if row == 0: full_data = row_fully_cleaned
                
                # Stack with existing data
                else: full_data = np.row_stack((full_data, row_fully_cleaned))
        
            # Create pandas.DataFrame
            df_PSLGM = pd.DataFrame(full_data, columns=['Month', 'Year', 'Gaps', 'Good', 'Minimum', 'Maximum', 'Mean', 'StD'])
            df_PSLGM['datetime'] = [datetime.datetime(year=int(df_PSLGM.Year[idx]), month=int(df_PSLGM.Month[idx]), day=1) for idx in df_PSLGM.index]
            df_PSLGM = df_PSLGM.set_index('datetime')
            df_PSLGM = df_PSLGM.rename(columns={'Mean': info})

            if idx_PSLGM == 0:
                df_PSLGM_total = df_PSLGM[[info]]
            
            else:
                df_PSLGM_total = pd.concat([df_PSLGM_total, df_PSLGM[info]], axis=1)

        # Save information in dictionary
        df_PSLGM_total = df_PSLGM_total.apply(pd.to_numeric)
        self.island_info['timeseries_PSLGM']['timeseries'] = df_PSLGM_total

    def main(self):

        # Retrieve the dictionary with currently available information about the island
        self.island_info = get_info_islands.retrieve_info_island(self.island, self.country, verbose=self.verbose_init)

        print('\n-------------------------------------------------------------------')
        print('RETRIEVING TIME SERIES FOR SEA LEVELS, BAROMETRIC PRESSURES, WATER TEMPERATURES AND AIR TEMPERTATURES (PSLGM) DATA')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        # If PSLGM data have NOT already been generated
        if not 'timeseries_PSLGM' in self.island_info.keys() or self.overwrite:

            # Create key/dict for PSLGM data
            self.island_info['timeseries_PSLGM'] = {}
            self.assign_metadata()

            # Run all functions
            self.get_timeseries()
        
        # If PSLGM data have already been generated
        else:

            print('~ Information already available. Returning data. ~')

        # Save information in dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)

        return self.island_info
