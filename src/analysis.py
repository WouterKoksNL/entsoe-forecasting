import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, root_mean_squared_error

from .load_saved_data import (
    load_load_data,
    load_generation_data,
)
from typing import Callable




def get_fitting_params(
        error_type: str, 
        years: list[int], 
        zones: list[str], 
        train_and_test_function: Callable, 
        fitting_function: Callable,
        n_lags=3, 
        max_lead_time=12,
        data_folder: str = "data/input_entsoe"
        ) -> pd.DataFrame:
    """
    Calculate RMSE w.r.t. lead time for each zone and fit the asymptotic function to the RMSE values.

    Args:
        error_type (str): Name of data type to be used for the analysis, either "wind" or "load".
        years (list[int]): List of years to load data for. Must be downloaded from ENTSO-E first. 
        zones (list[str]): List of zones, must be in the format as used by ENTSO-E, e.g. 'NO_2'
        train_and_test_function (Callable): A function that fits a model to the data and returns test values and predictions.
        fitting_function (Callable): Fit for the RMSE function w.r.t. lead time.
        n_lags (int, optional): Number of past timesteps to be used. Defaults to 3.
        max_lead_time (int, optional): Forecasting horizon. Defaults to 12.

    Raises:
        NotImplementedError: _description_

    Returns:
        pd.DataFrame: DataFrame with fitting parameters for each zone.
    """
    param_df = pd.DataFrame(index=zones, columns=['a', 'b'])
    for zone in zones:
        if error_type == "load":
            target_series, forecast_series, scaling_value = load_load_data(years, zone, folder=data_folder)
        else:
            target_series, forecast_series, scaling_value = load_generation_data(years, zone, folder=data_folder, carrier=error_type)


            # raise ValueError(f"Data for zone {zone} contains NaN values. Please check the data.")
        rmse_per_step = []
        for n_lead_time in range(1, max_lead_time + 1):                 
            y_test, y_pred = train_and_test_function(target_series, forecast_series, target_series - forecast_series, target_series.index.hour ,n_lags=n_lags, n_lead_time=n_lead_time)
            
            rmse = root_mean_squared_error(y_test, y_pred) / scaling_value
            rmse_per_step.append(rmse)

        horizons = np.arange(1, max_lead_time + 1)
        asymptotic_params, _ = curve_fit(fitting_function, horizons, rmse_per_step)
        a, b = asymptotic_params

        r_squared = r2_score(rmse_per_step, fitting_function(horizons, *asymptotic_params))

        param_df.loc[zone, 'a'] = a
        param_df.loc[zone, 'b'] = b
        param_df.loc[zone, 'r_squared'] = r_squared
    return param_df
