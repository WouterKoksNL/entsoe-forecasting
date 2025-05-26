import pandas as pd
# src/load_saved_data
def set_datetime_index(ser):
    ser.index = pd.to_datetime(ser.index, utc=True)
    return ser

def detrend(target_series, forecast_series):
    trends = (target_series.groupby(target_series.index.month).transform('mean'))
    detrended_target = target_series - trends
    detrended_forecast = forecast_series - trends
    return detrended_target, detrended_forecast


def load_load_data(years, zone, detrend_flag=True):

    load_df = pd.concat([pd.read_csv(f'data/entsoe_load_{year}{zone}.csv', index_col=0, parse_dates=True) for year in years], axis=0)

    load_series = load_df['Actual Load']
    load_forecast_series = load_df['Forecasted Load']
    mean_load = load_series.mean()
    load_series = set_datetime_index(load_series)
    load_forecast_series = set_datetime_index(load_forecast_series)
    if detrend_flag:
        load_series, load_forecast_series = detrend(load_series, load_forecast_series)
    
    load_forecast_series.ffill(inplace=True)
    return load_series, load_forecast_series, mean_load

def load_wind_data(years, zone, error_type="wind", detrend_flag=True):
    wind_series = pd.concat([pd.read_csv(f'data/entsoe_{error_type}_{year}{zone}.csv', index_col=0, parse_dates=True)['Wind Onshore'] for year in years], axis=0)
    wind_forecast_series =  pd.concat([
        pd.read_csv(f'data/entsoe_{error_type}_{year}{zone}_forecast.csv', index_col=0, parse_dates=True)['Wind Onshore'] for year in years
        ], axis=0)
    wind_series = set_datetime_index(wind_series)
    wind_forecast_series = set_datetime_index(wind_forecast_series)

    capacity = pd.read_csv('data/entsoe_capacities_Wind Onshore_mean.csv', index_col=0).loc[zone].values[0]

    if detrend_flag:
        wind_series, wind_forecast_series = detrend(wind_series, wind_forecast_series)
    
    wind_forecast_series.ffill(inplace=True)
    return wind_series, wind_forecast_series, capacity