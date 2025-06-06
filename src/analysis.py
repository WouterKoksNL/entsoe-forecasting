import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, root_mean_squared_error

from .load_saved_data import (
    load_load_data,
    load_generation_data,
)
from typing import Callable




def get_forecasts(
        error_type: str, 
        years: list[int], 
        zones: list[str], 
        train_and_test_function: Callable, 
        n_lags=3, 
        lead_time_range: range | list[int] = range(1, 13),
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
    y_test_dict = {}
    y_pred_dict = {}
    scaling_value_dict = {}
    for zone in zones:
        y_test_dict[zone] = {}
        y_pred_dict[zone] = {}
        if error_type == "load":
            target_series, forecast_series, scaling_value = load_load_data(years, zone, folder=data_folder)
        else:
            target_series, forecast_series, scaling_value = load_generation_data(years, zone, folder=data_folder, carrier=error_type)
        scaling_value_dict[zone] = scaling_value

        for n_lead_time in lead_time_range:                 
            y_test, y_pred = train_and_test_function(target_series, forecast_series, target_series - forecast_series, target_series.index.hour ,n_lags=n_lags, n_lead_time=n_lead_time)
            y_test_dict[zone][n_lead_time] = y_test
            y_pred_dict[zone][n_lead_time] = y_pred
           
    return y_test_dict, y_pred_dict, scaling_value_dict


def calc_fitting_params(
        lead_time_range,
        zones: list[str],
        y_test_dict: dict[str, dict[int, np.ndarray]],
        y_pred_dict: dict[str, dict[int, np.ndarray]],
        scaling_value_dict: dict[str, float],
        fitting_function: Callable,
    ):
    """_summary_

    Args:
        lead_time_range (_type_): _description_
        zones (list[str]): _description_
        y_test_dict (dict[str, dict[int, np.ndarray]]): _description_
        y_pred_dict (dict[str, dict[int, np.ndarray]]): _description_
        scaling_value_dict (dict[str, float]): _description_
        fitting_function (Callable): _description_
    """
    param_df = pd.DataFrame(index=zones, columns=['a', 'b'])
    rmse_dict = {}
    for zone in zones:
        scaling_value = scaling_value_dict[zone]
        rmse_ser = pd.Series(index=lead_time_range, dtype=float)
        for lead_time in lead_time_range:
            y_test = y_test_dict[lead_time]
            y_pred = y_pred_dict[lead_time]
            
            rmse_ser.loc[lead_time] = root_mean_squared_error(y_test, y_pred) / scaling_value

            
        asymptotic_params, _ = curve_fit(fitting_function, lead_time_range, rmse_ser)
        a, b = asymptotic_params

        r_squared = r2_score(rmse_ser, fitting_function(lead_time_range, *asymptotic_params))

        rmse_dict[zone] = rmse_ser
        param_df.loc[zone, 'a'] = a
        param_df.loc[zone, 'b'] = b
        param_df.loc[zone, 'r_squared'] = r_squared
    return param_df, rmse_dict