"""
This module allows us to match time frames and frequency (no NaN values and no extrapolation).
TODO: find time frames with the most time series
TODO: characterise time series (?)

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import pandas as pd
import pickle
import os
from mpetools import IslandTime
import numpy as np
import datetime
import pytz
import matplotlib.pyplot as plt
from scipy import stats

class TimeSeriesPreProcess:
    def __init__(self, island_info, date_range=['2015-01-01', '2022-01-01'], frequency='monthly', plot_confounders=False, find_optimal_time_frame=True):
        self.island_info = island_info
        self.date_range = date_range
        self.frequency = frequency
        self.plot_confounders = plot_confounders
        self.find_optimal_time_frame = find_optimal_time_frame
        self.preprocessed_timeseries_path = os.path.join(os.getcwd(), 'data', 'preprocessed_timeseries_islands')
        self.island = self.island_info['general_info']['island']
        self.country = self.island_info['general_info']['country']

    def main(self):

        print('\n-------------------------------------------------------------------')
        print('Time series pre-processing')
        print('Retrieving all available time series between {} and {} at a {} frequency'.format(self.date_range[0], self.date_range[1], self.frequency))
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        # Retrieve all available time series from `island_info` dictionary
        list_timeseries = [self.island_info[key]['timeseries'] for key in self.island_info.keys() if 'timeseries' in self.island_info[key]]

        # Make sure they all share the same UTC datetime index
        for timeseries in list_timeseries:
            if timeseries.index.tzinfo is None:
                timeseries.index = [pytz.utc.localize(timeseries.index[i]) for i in range(len(timeseries.index))]

        # Combine them in one DataFrame
        df_timeseries = pd.concat(list_timeseries, axis=1)
        df_timeseries = df_timeseries.apply(pd.to_numeric)

        # Retrieve all available confounders from `island_info` dictionary
        list_confounders = [self.island_info[key]['confounders'] for key in self.island_info.keys() if 'confounders' in self.island_info[key]]

        # Make sure they all share the same UTC datetime index
        for confounders in list_confounders:
            if confounders.index.tzinfo is None:
                confounders.index = [pytz.utc.localize(confounders.index[i]) for i in range(len(confounders.index))]

        # Combine them in one DataFrame
        df_confounders = pd.concat(list_confounders, axis=1)
        df_confounders = df_confounders.apply(pd.to_numeric)

        # Plot confounders
        if self.plot_confounders:
            plt.figure()
            df_confounders.plot()
            plt.show()

        # Replace outlier values with NaN using z-score (only abnormally high values)
        df_timeseries_remove_outliers = df_timeseries.copy() # create a copy of the DataFrame
        threshold_zscore = 20 # modify this value if needed

        # Calculate the z-score for each column
        log_df_timeseries_remove_outliers = np.log(np.abs(df_timeseries_remove_outliers))
        z_scores = stats.zscore(log_df_timeseries_remove_outliers, nan_policy='omit')

        # Outlier mask
        outliers_mask = z_scores.values > threshold_zscore

        # Replace outliers with NaN for each column
        df_timeseries_remove_outliers[outliers_mask] = np.nan

        # Group by frequency
        df_timeseries_frequency = df_timeseries_remove_outliers.groupby(pd.Grouper(freq=self.frequency[0].capitalize())).mean()

        # Fill NaN with mean (only for time series with very few NaN)
        df_timeseries_frequency_rolling = df_timeseries_frequency.fillna(df_timeseries_frequency.rolling(2, min_periods=1, center=True).mean())

        # Start and end dates as `datetime`
        start_date = pytz.utc.localize(datetime.datetime.strptime(self.date_range[0], '%Y-%m-%d'))
        end_date = pytz.utc.localize(datetime.datetime.strptime(self.date_range[1], '%Y-%m-%d'))

        # Select time frame
        df_timeseries_dates = df_timeseries_frequency_rolling[(df_timeseries_frequency_rolling.index >= start_date) \
                                                                    & (df_timeseries_frequency_rolling.index <= end_date)]
    
        # Remove time series with too many NaN
        #df_timeseries_frequency_rolling_dropna = df_timeseries_final.dropna(axis=1, how='any')

        # Statistical test for trends

        # Statistical test for seasonality

        # Statistical test for stationarity

        # Add DataFrame for characterisation

        # Raw DataFrame

        # Cleaned, pre-processed DataFrame

        # Save pre-processed time series

        # Return pre-processed time series

        return df_timeseries_frequency_rolling