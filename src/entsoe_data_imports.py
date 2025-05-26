import pandas as pd
from entsoe import EntsoePandasClient


def get_capacities(
        client: EntsoePandasClient, 
        zones, 
        years, 
        carrier='Wind Onshore'):
    caps_df = pd.DataFrame(index=zones, columns=years, dtype=float)
    for year in years:
        start = pd.Timestamp(f'{year}0101', tz='Europe/Brussels')
        end = pd.Timestamp(f'{year}0102', tz='Europe/Brussels') # only year of starting day matters

        for zone in zones:
            try:
                installed_cap = client.query_installed_generation_capacity(country_code=zone, start=start, end=end, psr_type=None)[carrier]
                caps_df.loc[zone, year] = installed_cap.values[0]
            except:
                print(f"Error fetching data for {zone} in {year}. Setting capacity to 0.")
                caps_df.loc[zone, year] = 0

    caps_df.to_csv(f'data/entsoe_capacities_{carrier}.csv')
    mean_cap = caps_df.mean(axis=1)
    mean_cap.to_csv(f'data/entsoe_capacities_{carrier}_mean.csv')
    return 


def get_generation(
        client: EntsoePandasClient, 
        zones, 
        years
        ):
    for year in years:
        start = pd.Timestamp(f'{year}0101', tz='Europe/Brussels')
        end = pd.Timestamp(f'{year+1}0101', tz='Europe/Brussels')

        for country_code in zones:
            try:
                df = client.query_generation(country_code, start=start, end=end, psr_type=None, include_eic=False)
                df.to_csv(f'data/entsoe_wind_{year}' + country_code + '.csv')
            except:
                print('Error downloading data for ' + country_code)
    return 

def get_load_and_forecast(
        client: EntsoePandasClient, 
        zones, 
        years
        ):
    
    for year in years:
        start = pd.Timestamp(f'{year}0101', tz='Europe/Brussels')
        end = pd.Timestamp(f'{year+1}0101', tz='Europe/Brussels')
        for zone in zones:
            try:
                load_df = client.query_load_and_forecast(zone, start=start, end=end)
                load_df.to_csv(f'data/entsoe_load_{year}{zone}.csv')
            except:
                print('Error downloading load data for ' + zone)
    return 

def get_wind_solar_forecast(client, zones, years, carrier='Wind Onshore'):
    for year in years:
        start = pd.Timestamp(f'{year}0101', tz='Europe/Brussels')
        end = pd.Timestamp(f'{year+1}0101', tz='Europe/Brussels')
        for zone in zones:
            try:
                forecast_df = client.query_wind_and_solar_forecast(zone, start=start, end=end, psr_type=None)
                forecast_series = forecast_df[carrier]
                forecast_series.to_csv(f'data/entsoe_wind_{year}{zone}_forecast.csv')
            except:
                print('Error downloading forecast data for ' + zone)
    return