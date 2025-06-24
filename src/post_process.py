from src.utils import get_hourly
import pandas as pd
import numpy as np


def restructure_forecast_df(target, da_forecast, forecasts_df, n_steps=1):
    """Restructure the forecast DataFrame to prepare for plotting forecast updates in different stages. 
    Ththe forecasts_df must have the same index as the target and da_forecast, so it must have been preprocessed."""
    hourly_forecasts = get_hourly(forecasts_df)
    hourly_forecasts = hourly_forecasts.clip(0)
    hourly_forecasts.columns = (hourly_forecasts.columns / n_steps).astype(int)
    hourly_target = get_hourly(target)
    default_forecast = get_hourly(da_forecast)

    forecast_hours = np.concatenate([np.array([-12, -9]), np.array(range(-2, 24))])

    delivery_hours = range(24)

    # Container for final forecasts
    forecasts_dict = {}

    for forecast_hour in forecast_hours:
        forecast_values = []
        for delivery_hour in delivery_hours:
            # Calculate lead time
            lead_time = delivery_hour - forecast_hour

            if lead_time in hourly_forecasts.columns:
                val = hourly_forecasts.loc[delivery_hour, lead_time]
            elif lead_time <= 0:
                val = hourly_target.loc[delivery_hour]
            else:
                val = default_forecast.loc[delivery_hour]
            forecast_values.append(val)

        forecasts_dict[forecast_hour] = pd.Series(forecast_values, index=delivery_hours)

    # Final forecasts DataFrame
    restructured_forecasts = pd.DataFrame(forecasts_dict)
    return restructured_forecasts
