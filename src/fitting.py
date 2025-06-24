
import pickle
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, root_mean_squared_error
from typing import Callable

from .utils import make_dir
from .forecast_data_handling import get_data


def get_fitting_function(fitting_function_name: str) -> Callable:
    """
    Get the fitting function based on the provided name.

    Args:
        fitting_function_name (str): Name of the fitting function to be used.

    Returns:
        Callable: The fitting function.
    """
    if fitting_function_name == "asymptotic":
        return lambda x, a, b: a * x / (x + b)
    else:
        raise NotImplementedError(f"Fitting function '{fitting_function_name}' is not implemented.")
    

def calc_nrmse(
        lead_time_range: range | list[int],
        y_test_dict: dict[int, np.ndarray], 
        y_pred_dict: dict[int, np.ndarray],
        scaling_value: float,
        ) -> pd.Series:
    """Calculate RMSE for each lead time and return as a pandas Series.

    Args:
        lead_time_range (range | list[int]): _description_
        y_test_dict (dict[int, np.ndarray]): _description_
        y_pred_dict (dict[int, np.ndarray]): _description_
        scaling_value (float): _description_

    Returns:
        pd.Series: _description_
    """
    rmse_ser = pd.Series(index=lead_time_range, dtype=float)
    for lead_time in lead_time_range:
        y_test = y_test_dict[lead_time]
        y_pred = y_pred_dict[lead_time]
        rmse_ser.loc[lead_time] = root_mean_squared_error(y_test, y_pred) / scaling_value
    return rmse_ser


def calc_fitting_params(
    lead_time_range,
    rmse_ser: pd.Series,
    fitting_function: Callable,
    ) -> tuple[float, float, float]:

    asymptotic_params, _ = curve_fit(fitting_function, lead_time_range, rmse_ser)
    a, b = asymptotic_params  # hard-coded two fitting parameters! 
    r_squared = r2_score(rmse_ser, fitting_function(lead_time_range, *asymptotic_params))
    return a, b, r_squared 


def calc_fitting_params_loop(
        config,
    ):
    """Calculate RMSE w.r.t. lead time for each zone and fit the asymptotic function to the RMSE values.

    Args:
        config (ConfigSettings): Configuration settings 
    """
    output_param_dir = f"output/{config.run_id}"
    forecast_pickle_dir = f"data/pickled_forecasts/{config.run_id}"
    make_dir(output_param_dir)

    lead_time_range = range(1, config.max_lead_time + 1)
    fitting_function = get_fitting_function(config.fitting_function)

    rmse_dict = {}
    params_dict = {}
    for error_type in config.error_types:
        rmse_dict[error_type] = {}
        zones = config.zones_error_types[error_type]
        param_df = pd.DataFrame(index=zones, columns=['a', 'b'])
        for zone in zones:
            y_test_dict, y_pred_dict, scaling_value, _, _ = get_data(
                forecast_pickle_dir, zone, error_type)
            

            rmse_ser = calc_nrmse(lead_time_range, y_test_dict, y_pred_dict, scaling_value)
            a, b, r_squared = calc_fitting_params(lead_time_range, rmse_ser, fitting_function)
            rmse_dict[error_type][zone] = rmse_ser
            param_df.loc[zone, 'a'] = a
            param_df.loc[zone, 'b'] = b
            param_df.loc[zone, 'r_squared'] = r_squared
            param_df.to_csv(f"{output_param_dir}/rmse_params.csv", index=True)
        params_dict[error_type] = param_df
        make_dir(f"{output_param_dir}/{error_type}")
        param_df.to_csv(f"{output_param_dir}/{error_type}/fitting_params.csv", index=True)
        
    # pickle rmse values
    with open(f"{output_param_dir}/rmse_dict.pkl", "wb") as f:
        pickle.dump(rmse_dict, f)


    # restructured_params = {}
    # for error_type, params_df in params_dict.items():
    #     # if no keys in restructured_params, initialize it
    #     if not restructured_params:
    #         restructured_params = {zone: {} for zone in params_df.index}
    #     restructured_params[zone][error_type] = params_df.loc[zone]
    # for zone, params in restructured_params.items():
    #     make_dir(f"{output_param_dir}/{zone}")
    #     restructured_params[zone] = pd.DataFrame(params).T
    #     restructured_params[zone].to_csv(f"{output_param_dir}/{zone}/fitting_params.csv", index=True)
    return params_dict, rmse_dict


