"""
This module retrieves climate indices: monthly atmospheric and ocean time series (e.g., ENSO).
Link: https://psl.noaa.gov/data/climateindices/list/
TODO: add all references for climate indices.
TODO: deal with bimonthly data?
TODO: select relevant indices for island

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
from mpetools import get_info_islands
import os
import pickle
import pandas as pd
import re
import urllib.request
import numpy as np
import datetime

class TimeSeriesClimateIndices:
    def __init__(self, island, country, verbose_init=True, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands'), overwrite=False):
        self.island = island
        self.country = country
        self.verbose_init = verbose_init
        self.island_info_path = island_info_path
        self.overwrite = overwrite

    def assign_metadata(self):

        # Add description of this database
        self.island_info['timeseries_climate_indices']['description'] = 'This module retrieves climate indices: monthly atmospheric and ocean time series (e.g., ENSO).'
        
        # Add description of this timeseries
        self.island_info['timeseries_climate_indices']['description_timeseries'] = { 'pna': 'Pacific North American Index', \
                                                                                    'epo': 'East Pacific/North Pacific Oscillation', \
                                                                                    'wp': 'Western Pacific Index', \
                                                                                    'ea': 'Eastern Atlantic/Western Russia', \
                                                                                    'nao': 'North Atlantic Oscillation', \
                                                                                    'jonesnao': 'North Atlantic Oscillation (Jones)', \
                                                                                    'soi': 'Southern Oscillation Index', \
                                                                                    'nina3.anom': 'Eastern Tropical Pacific SST (anomalies)', \
                                                                                    'nina3': 'Eastern Tropical Pacific SST (mean values)', \
                                                                                    'censo': 'Bivariate ENSO Timeseries', \
                                                                                    'tna': 'Tropical Northern Atlantic Index', \
                                                                                    'tsa': 'Tropical Southern Atlantic Index', \
                                                                                    'whwp': 'Western Hemisphere Warm Pool', \
                                                                                    'oni': 'Oceanic Niño Index', \
                                                                                    'meiv2': 'Multivariate ENSO Index (MEI V2)', \
                                                                                    'nina1.anom': 'Extreme Eastern Tropical Pacific SST (anomalies)', \
                                                                                    'nina1': 'Extreme Eastern Tropical Pacific SST (mean values)', \
                                                                                    'nina4.anom': 'Central Tropical Pacific SST (anomalies)', \
                                                                                    'nina4': 'Central Tropical Pacific SST (mean values)', \
                                                                                    'nina34.anom': 'East Central Tropical Pacific SST (anomalies)', \
                                                                                    'nina34': 'East Central Tropical Pacific SST (mean values)', \
                                                                                    'pdo': 'Pacific Decadal Oscillation', \
                                                                                    'ipotpi.hadisst2': 'Tripole Index for the Interdecadal Pacific Oscillation', \
                                                                                    'noi': 'Northern Oscillation Index', \
                                                                                    'np': 'North Pacific Pattern', \
                                                                                    'tni': 'Indices of El Niño Evolution', \
                                                                                    'hurr': 'Monthly Totals Atlantic Hurricanes and Named Tropical Storms', \
                                                                                    'ao': 'Arctic Oscillation', \
                                                                                    'aao': 'Antarctic Oscillation', \
                                                                                    'pacwarm': 'Pacific Warmpool Area Average', \
                                                                                    'eofpac': 'Tropical Pacific SST EOF', \
                                                                                    'atltri': 'Atlantic Tripole SST EOF', \
                                                                                    'amon.us': 'Atlantic multidecadal Oscillation (unsmoothed)', \
                                                                                    'ammsst': 'Atlantic Merodional Mode', \
                                                                                    'NTA_ersst': 'North Tropical Atlantic SST Index', \
                                                                                    'CAR_ersst': 'Caribbean SST Index', \
                                                                                    'amon.sm': 'Atlantic Multidecadal Oscillation (smoothed)', \
                                                                                    'qbo': 'Quasi-Biennial Oscillation', \
                                                                                    'espi': 'ENSO Precipitation Index', \
                                                                                    'indiamon': 'Central Indian Precipitation', \
                                                                                    'sahelrain': 'Sahel Standardized Rainfall', \
                                                                                    'swmonsoon': 'Area Averaged Precipitation for Arizona and New Mexico', \
                                                                                    'brazilrain': 'Northeast Brazil Rainfall Anomaly', \
                                                                                    'solar': 'Solar Flux (10.7cm)', \
                                                                                    'gmsst': 'Global Mean Land/Ocean Temperature', \
                                                                                    }  
                                                                                    
        # Add source (paper or url)
        self.island_info['timeseries_climate_indices']['source'] = 'https://psl.noaa.gov/data/climateindices/list/'

    def retrieve_list_urls(self):

        # URL for list of all climate indices
        url_PSL = 'https://psl.noaa.gov/data/climateindices/list/'
        web_PSL = urllib.request.urlopen(url_PSL)
        data_PSL = str(web_PSL.read())

        # Find all URLs within the data from the website
        pattern_url = r'href="([^"]+)"'
        complete_list_urls = np.array(re.findall(pattern_url, data_PSL))

        # Clean list of URLs (only keep data URLs)
        list_urls_first_cleaning = complete_list_urls[np.argwhere((np.char.startswith(complete_list_urls, r'/enso/')) | (np.char.startswith(complete_list_urls, r'/data/'))).T[0]]
        list_urls_second_cleaning = list_urls_first_cleaning[np.argwhere((~np.char.endswith(list_urls_first_cleaning, r'html')) & \
                                                                         (~np.char.endswith(list_urls_first_cleaning, r'/')) & \
                                                                         (np.char.find(list_urls_first_cleaning, "list") == -1) & \
                                                                         (np.char.find(list_urls_first_cleaning, ".long.") == -1)).T[0]][1:]

        # Delete depecrated climate indices (no recent updates)
        list_urls_third_cleaning = np.delete(list_urls_second_cleaning, np.argwhere(list_urls_second_cleaning=='/data/correlation/trend.data')[0][0])
        list_urls_cleaned = np.delete(list_urls_third_cleaning, np.argwhere(list_urls_third_cleaning=='/data/correlation/glaam.data.scaled')[0][0]) 
        
        # Save information
        self.list_urls = list_urls_cleaned

    def get_timeseries(self):

        # Retrieve list of URL from NOAA PSL website
        self.retrieve_list_urls()

        # Loop through all climate indices
        for idx, climate_index in enumerate(self.island_info['timeseries_climate_indices']['description_timeseries'].keys()):

            print('~ Retrieving {}. ~'.format(self.island_info['timeseries_climate_indices']['description_timeseries'][climate_index]))

            # Find corresponding URL
            url_climate_index = self.list_urls[np.argwhere(np.char.endswith(self.list_urls, '{}.data'.format(climate_index)))[0][0]]

            try:
                # Retrieve data from URL
                web_climate_index = urllib.request.urlopen('https://psl.noaa.gov' + url_climate_index)
                data_climate_index = str(web_climate_index.read())

            except:
                print('Climate index not available.')
                continue

            # Split into a list and retrieve metadata
            list_data_climate_index = data_climate_index.replace('\\t', '').split('\\n')
            list_data_climate_index = [row.strip() for row in list_data_climate_index]
            _, startyear, endyear = list_data_climate_index[0].replace("'", ' ').split()
            nan_value = list_data_climate_index[np.argwhere(np.char.startswith(list_data_climate_index, endyear))[0][0] + 1].strip()
            
            # Keep actual data
            actual_data_climate_index = list_data_climate_index[np.argwhere(np.char.startswith(list_data_climate_index, startyear))[0][0] : \
                                                                np.argwhere(np.char.startswith(list_data_climate_index, endyear))[0][0] + 1]

            # Loop through every row -> data cleaning & stacking
            for row in range(len(actual_data_climate_index)):

                # String -> array
                row_cleaned = np.fromstring(actual_data_climate_index[row], dtype=float, sep=' ')

                # Store data
                if row == 0: 
                    arr_data_climate_index = row_cleaned
                else: 
                    arr_data_climate_index = np.row_stack((arr_data_climate_index, row_cleaned))

            # Array -> DataFrame
            df_data_climate_index = pd.DataFrame(arr_data_climate_index, columns=['year', 'janval', 'febval', 'marval', 'aprval', 'mayval', 'junval', 'julval', 'augval', 'sepval', 'octval', 'novval', 'decval'])

            # From this DataFrame -> create new DataFrame to match the format of other time series
            arr_timeseries_climate_index = np.array(['datetime', climate_index], dtype=object)

            # Retrieve data for every year and month
            for idx_year in df_data_climate_index.index:
                for idx_month in range(1, len(df_data_climate_index.columns[1:]) + 1):
                    arr_timeseries_climate_index = np.row_stack((arr_timeseries_climate_index, \
                                                                 np.array([datetime.datetime(year=int(df_data_climate_index.year[idx_year]), \
                                                                                             month=idx_month, day=1), \
                                                                                             df_data_climate_index[df_data_climate_index.columns[1:][idx_month-1]][idx_year]])))

            # Replace extreme values by NaN
            arr_timeseries_climate_index[:, 1][arr_timeseries_climate_index[:, 1] == float(nan_value)] = np.nan

            # Build final DataFrame
            df_timeseries_climate_index = pd.DataFrame(arr_timeseries_climate_index[1:], columns=arr_timeseries_climate_index[0])
            df_timeseries_climate_index = df_timeseries_climate_index.set_index('datetime')

            # Combine data with other climate indices
            if idx == 0:
                df_timeseries_climate_indices = df_timeseries_climate_index
            
            else:
                df_timeseries_climate_indices = pd.concat([df_timeseries_climate_indices, df_timeseries_climate_index], axis=1)

        # Save information in dictionary
        df_timeseries_climate_indices = df_timeseries_climate_indices.apply(pd.to_numeric)
        self.island_info['timeseries_climate_indices']['timeseries'] = df_timeseries_climate_indices
        
    def main(self):

        # Retrieve the dictionary with currently available information about the island
        self.island_info = get_info_islands.retrieve_info_island(self.island, self.country, verbose=self.verbose_init)

        print('\n-------------------------------------------------------------------')
        print('RETRIEVING CLIMATE INDICES (NOAA Physical Sciences Laboratory) DATA')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        # If climate_indices data have NOT already been generated
        if not 'timeseries_climate_indices' in self.island_info.keys() or self.overwrite:

            # Create key/dict for climate_indices data
            self.island_info['timeseries_climate_indices'] = {}
            self.assign_metadata()

            # Run all functions
            self.get_timeseries()
        
        # If climate_indices data have already been generated
        else:
            print('~ Information already available. Returning data. ~')

        # Save dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)
        
        return self.island_info