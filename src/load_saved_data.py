import pandas as pd
from .utils import convert_index_to_datetime

def detrend(target_series, forecast_series):
    trends = (target_series.groupby(target_series.index.month).transform('mean'))
    detrended_target = target_series - trends
    detrended_forecast = forecast_series - trends
    return detrended_target, detrended_forecast, trends


def load_load_data(years, zone, folder="data/input_entsoe"):
    load_df = pd.concat([pd.read_csv(f'{folder}/load_and_forecasts/{zone}/{year}.csv', index_col=0, parse_dates=True) for year in years], axis=0)
    load_series = load_df['Actual Load']
    load_forecast_series = load_df['Forecasted Load']
    mean_load = load_series.mean()
    load_series = convert_index_to_datetime(load_series)
    load_forecast_series = convert_index_to_datetime(load_forecast_series)
    load_forecast_series.ffill(inplace=True)
    return load_series, load_forecast_series, mean_load


def load_generation_data(years, zone, folder="data/input_entsoe", carrier='Wind Onshore'):
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
    
    # fill missing values with mean daily trend
    trends = (target_series.groupby(target_series.index.month).transform('mean')) + (target_series.groupby(target_series.index.hour).transform('mean'))

    target_series.fillna(trends, inplace=True)
    wind_forecast_series.fillna(trends, inplace=True)
    # if detrend_flag:
    #     target_series, wind_forecast_series = detrend(target_series, wind_forecast_series)
    

    return target_series, wind_forecast_series, capacity