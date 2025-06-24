import pandas as pd
from entsoe import EntsoePandasClient
from utils import make_dir

def get_capacities(
        client: EntsoePandasClient, 
        zones: list[str], 
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

    carrier_save_name = carrier.replace(' ', '_')
    zones_save_name = "_".join(zones)
    make_dir(f'data/input_entsoe/mean_capacities/{carrier_save_name}')
    make_dir(f'data/input_entsoe/capacities/{carrier_save_name}')
    caps_df.to_csv(f'data/input_entsoe/capacities/{carrier_save_name}/{zones_save_name}.csv')
    mean_cap = caps_df.mean(axis=1)
    mean_cap.to_csv(f'data/input_entsoe/mean_capacities/{carrier_save_name}/{zones_save_name}.csv')
    return 

def get_start_end_dates(
        year: int,
):
    start= pd.Timestamp(f'{year}0101', tz='Europe/Brussels')
    end = pd.Timestamp(f'{year+1}0101', tz='Europe/Brussels')
    return start, end

def get_generation(
        client: EntsoePandasClient, 
        zones, 
        years,
        ):

    for year in years:
        start, end = get_start_end_dates(year)
        for zone in zones:
            zone_dir = f"data/input_entsoe/generation/{zone}"
            make_dir(zone_dir)
            try:
                df = client.query_generation(zone, start=start, end=end, psr_type=None, include_eic=False)
                if isinstance(df.columns, pd.MultiIndex):
                    # extract the columns in which the second level is 'Actual Aggregated'
                    if 'Actual Aggregated' in df.columns.get_level_values(1):
                        df = df.loc[:, df.columns.get_level_values(1) == 'Actual Aggregated']
                        df.columns = df.columns.droplevel(1)
                    else:
                            print(f"Warning: 'Actual Aggregated' not found in columns for {zone} in {year}, \
                            while df.columns is a MultiIndex. Check the data structure.")
                df.to_csv(f'data/input_entsoe/generation/{zone}/{year}.csv')
            except:
                print('Error downloading data for ' + zone)
    return 

def get_load_and_forecast(
        client: EntsoePandasClient, 
        zones, 
        years
        ):
    
    for year in years:
        start, end = get_start_end_dates(year)
        for zone in zones:
            zone_dir = f"data/input_entsoe/load_and_forecasts/{zone}"
            make_dir(zone_dir)
            try:
                load_df = client.query_load_and_forecast(zone, start=start, end=end)
                load_df.to_csv(f'{zone_dir}/{year}.csv')
            except:
                print('Error downloading load data for ' + zone)
    return 

def get_wind_solar_forecast(client, zones, years, carriers=['Wind Onshore']):
    for year in years:
        start, end = get_start_end_dates(year)
        for zone in zones:
            try:
                forecast_df = client.query_wind_and_solar_forecast(zone, start=start, end=end, psr_type=None)
                for carrier in carriers:
                    zone_dir = f"data/input_entsoe/generation_forecasts/{carrier}/{zone}"
                    make_dir(zone_dir)
                    forecast_series = forecast_df[carrier]
                    forecast_series.to_csv(f'data/input_entsoe/generation_forecasts/{carrier}/{zone}/{year}.csv')
            except:
                print('Error downloading forecast data for ' + zone)
    return

def get_wind_solar_intraday_forecast(client, zones, years, carrier='Wind Onshore'):
    for year in years:
        start, end = get_start_end_dates(year)
        for zone in zones:
            zone_dir = f"data/input_entsoe/intraday_generation_forecasts/{carrier}/{zone}"
            make_dir(zone_dir)
            try:
                forecast_df = client.query_intraday_forecast(zone, start=start, end=end, psr_type=None)
                forecast_series = forecast_df[carrier]
                forecast_series.to_csv(f'data/input_entsoe/generation_forecasts/{carrier}/{zone}/{year}.csv')
            except:
                print('Error downloading forecast data for ' + zone)
    return