import pandas as pd
import numpy as np

import pickle
from typing import Callable

from .load_saved_data import (
    load_load_data,
    load_generation_data,
)
from .utils import make_dir
from .load_config import ConfigSettings
from .algorithms.get_algorithm import get_algorithm
from .forecast_data_handling import save_data


def get_correct_time_indices(y_pred_dict, target_index, max_lead_time):
        pred_dict = {}
        for n_lead_time in range(1, max_lead_time + 1):
                start_index = target_index.size - y_pred_dict[n_lead_time].size
                y_pred_ser = pd.Series(y_pred_dict[n_lead_time], index=target_index[start_index:])
                pred_dict[n_lead_time] = y_pred_ser
        return pred_dict

def get_forecasts(
        error_type: str, 
        zone: str, 
        years: list[int], 
        train_and_test_function: Callable, 
        n_lags=3, 
        lead_time_range: range | list[int] = range(1, 13),
        data_folder: str = "data/input_entsoe",
        train_test_split: float = 0.8,
        pickle_output_flag: bool = True,
        forecast_pickle_dir: str = "data/pickled_forecasts",
        ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """
    Calculate RMSE w.r.t. lead time for each zone and fit the asymptotic function to the RMSE values. 
    Returns the test values and predictions for each lead time.
    Optionally saves the results as pickle files.

    Args:
        error_type (str): Type of error to forecast, e.g., 'load', 'Wind Onshore', 'Solar'.
        zone (str): Zone for which to calculate the forecasts. Must conform with ENTSO-E convention. 
        years (list[int]): List of years for which to calculate the forecasts.
        train_and_test_function (Callable): Function to train and test the model (in src.algorithms)
        n_lags (int, optional): Number of lags to use in the model. Defaults to 3.
        lead_time_range (range | list[int], optional): Range of lead times to consider. Defaults to range(1, 13).
        data_folder (str, optional): Folder where input data is stored. Defaults to "data/input_entsoe".
        train_test_split (float, optional): Proportion of data to use for training. Defaults to 0.8.
        pickle_output_flag (bool, optional): Whether to save the results as pickle files. Defaults to True.
        forecast_pickle_dir (str, optional): Directory where pickle files will be saved. Defaults to "data/pickled_forecasts".


    """
    y_test_dict = {}
    y_pred_dict = {}


    print(f"Processing error type {error_type}...")

    if error_type == "load":
        target_series, forecast_series, scaling_value = load_load_data(years, zone, folder=data_folder)
    else:
        target_series, forecast_series, scaling_value = load_generation_data(years, zone, folder=data_folder, carrier=error_type)
    if target_series.isna().sum() > 10:
        raise ValueError(f"Target series for {zone} in {error_type} has too many NaN values. Please check the data.")
    target_series.ffill(inplace=True)
    forecast_series.ffill(inplace=True)
    scaled_target = target_series / scaling_value
    scaled_forecast = forecast_series / scaling_value
    scaled_target_mean = scaled_target.mean()
    scaled_target_std = scaled_target.std()
    scaled_forecast_mean = scaled_forecast.mean()
    scaled_forecast_std = scaled_forecast.std()
    difference = scaled_target - scaled_forecast
    difference_mean = difference.mean()
    difference_std = difference.std()
    # Normalize the series
    normalized_target = (scaled_target - scaled_target_mean) / scaled_target_std
    normalized_forecast = (scaled_forecast - scaled_forecast_mean) / scaled_forecast_std
    difference = (difference - difference_mean) / difference_std

    target_series_hour_sin = np.sin(target_series.index.hour * (2 * np.pi / 24))
    target_series_series_dayofyear_sin = np.sin(target_series.index.dayofyear * (2 * np.pi / 365))
    target_series_hour_cos = np.cos(target_series.index.hour * (2 * np.pi / 24))
    target_series_series_dayofyear_cos = np.cos(target_series.index.dayofyear * (2 * np.pi / 365))
    for n_lead_time in lead_time_range:     
        reduced_forecast_error_usability_entry = np.maximum(0, n_lags - target_series.index.hour)
        y_test, y_pred = train_and_test_function(
            target_series=normalized_target.values, 
            forecast_series=normalized_forecast.values, 
            past_forecast_error=difference.values, 
            reduced_forecast_error_usability_entry=reduced_forecast_error_usability_entry, 
            target_series_hour_sin=target_series_hour_sin, 
            target_series_yearly_sin=target_series_series_dayofyear_sin, 
            target_series_hour_cos=target_series_hour_cos, 
            target_series_yearly_cos=target_series_series_dayofyear_cos, 
            n_lags=n_lags, 
            n_lead_time=n_lead_time,
            train_test_split=train_test_split,
            )
        y_test_dict[n_lead_time] = (y_test * scaled_target_std + scaled_target_mean) * scaling_value
        y_pred_dict[n_lead_time] = (y_pred * scaled_target_std + scaled_target_mean) * scaling_value


    make_dir(f"{forecast_pickle_dir}/{zone}/{error_type}")
    if pickle_output_flag:
        save_data(forecast_pickle_dir, zone, error_type, y_test_dict, y_pred_dict, scaling_value, target_series, forecast_series)   


    return y_test_dict, y_pred_dict


def get_forecast_loop(
        config: ConfigSettings, 
        ):
    """_summary_

    Args:
        config (ConfigSettings): _description_
    """
    forecasting_function = get_algorithm(config.forecasting_model)
    for error_type in config.error_types:
        zones = config.zones_error_types[error_type]
        for zone in zones:
            get_forecasts(
                error_type, 
                zone, 
                config.years, 
                train_and_test_function=forecasting_function, 
                n_lags=config.n_lags, 
                lead_time_range=range(1, config.max_lead_time + 1),
                train_test_split=config.train_test_split,
                forecast_pickle_dir=f"data/pickled_forecasts/{config.run_id}",
                )
    return 
