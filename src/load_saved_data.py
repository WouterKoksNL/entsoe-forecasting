"""Load saved ENTSO-E raw output"""

import pandas as pd
from .utils import (convert_index_to_datetime, get_hourly)

def detrend(target_series, forecast_series):
    trends = (target_series.groupby(target_series.index.month).transform('mean'))
    detrended_target = target_series - trends
    detrended_forecast = forecast_series - trends
    return detrended_target, detrended_forecast, trends


def load_load_data(years, zone, folder="data/input_entsoe", hourly_flag=True):
    load_df = pd.concat([pd.read_csv(f'{folder}/load_and_forecasts/{zone}/{year}.csv', index_col=0, parse_dates=True) for year in years], axis=0)
    load_series = load_df['Actual Load']
    load_forecast_series = load_df['Forecasted Load']
    mean_load = load_series.mean()
    load_series = convert_index_to_datetime(load_series)
    load_forecast_series = convert_index_to_datetime(load_forecast_series)
    if load_series.isna().sum() > 10:
        raise ValueError(f"Load series for {zone} has too many NaN values. Please check the data.")
    load_forecast_series.ffill(inplace=True)

    if hourly_flag:
        # keep only hourly values
        load_series = get_hourly(load_series, set_hour_indices=False)
        load_forecast_series = get_hourly(load_forecast_series, set_hour_indices=False)
    return load_series, load_forecast_series, mean_load


def load_generation_data(years, zone, folder="data/input_entsoe", carrier='Wind Onshore', hourly_flag=True):
    target_series = pd.concat([
        pd.read_csv(f'{folder}/generation/{zone}/{year}.csv', index_col=0, parse_dates=True, dtype=float)[carrier] 
        for year in years], axis=0)
    
    wind_forecast_series =  pd.concat([
        pd.read_csv(f'{folder}/generation_forecasts/{carrier}/{zone}/{year}.csv', index_col=0, parse_dates=True, dtype=float)[carrier] 
        for year in years], axis=0)
    
    target_series = convert_index_to_datetime(target_series)
    wind_forecast_series = convert_index_to_datetime(wind_forecast_series)

    # drop values that are not in both series
    target_series = target_series[target_series.index.isin(wind_forecast_series.index)]
    wind_forecast_series = wind_forecast_series[wind_forecast_series.index.isin(target_series.index)]


    capacity = pd.read_csv(f'{folder}/mean_capacities/{carrier.replace(" ", "_")}/{zone}.csv', index_col=0).loc[zone].values[0]
    
    if target_series.isna().sum() > 10:
        raise ValueError(f"Target series for {zone} in {carrier} has too many NaN values. Please check the data.")
    # fill missing values with mean daily trend
    trends = (target_series.groupby(target_series.index.month).transform('mean')) + (target_series.groupby(target_series.index.hour).transform('mean'))

    target_series.fillna(trends, inplace=True)
    wind_forecast_series.fillna(trends, inplace=True)
    
    if hourly_flag:
        # keep only hourly values
        target_series = get_hourly(target_series, set_hour_indices=False)
        wind_forecast_series = get_hourly(wind_forecast_series, set_hour_indices=False)

    

    return target_series, wind_forecast_series, capacity