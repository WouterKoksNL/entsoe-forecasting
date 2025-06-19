import pickle
import numpy as np
import pandas as pd


def save_data(forecast_pickle_dir, zone, error_type, y_test_dict, y_pred_dict, scaling_value, target_series, forecast_series):
    with open(f"{forecast_pickle_dir}/{zone}/{error_type}/y_test_dict.pkl", "wb") as f:
        pickle.dump(y_test_dict, f)
    with open(f"{forecast_pickle_dir}/{zone}/{error_type}/y_pred_dict.pkl", "wb") as f:
        pickle.dump(y_pred_dict, f)
    with open(f"{forecast_pickle_dir}/{zone}/{error_type}/scaling_value_mw.txt", "w") as f:
        f.write(str(scaling_value))
    # save target and forecast series as CSV files
    target_series.to_csv(f"{forecast_pickle_dir}/{zone}/{error_type}/target_series.csv")
    forecast_series.to_csv(f"{forecast_pickle_dir}/{zone}/{error_type}/forecast_series.csv")
        

def get_data(forecast_pickle_dir, zone, error_type):
    with open(f"{forecast_pickle_dir}/{zone}/{error_type}/y_test_dict.pkl", "rb") as f:
        y_test_dict = pickle.load(f)
    with open(f"{forecast_pickle_dir}/{zone}/{error_type}/y_pred_dict.pkl", "rb") as f:
        y_pred_dict = pickle.load(f)
    with open(f"{forecast_pickle_dir}/{zone}/{error_type}/scaling_value_mw.txt", "r") as f:
        scaling_value = float(f.read().strip())
    target_series = pd.read_csv(f"{forecast_pickle_dir}/{zone}/{error_type}/target_series.csv", index_col=0, parse_dates=True)
    forecast_series = pd.read_csv(f"{forecast_pickle_dir}/{zone}/{error_type}/forecast_series.csv", index_col=0, parse_dates=True)
    return y_test_dict, y_pred_dict, scaling_value, target_series, forecast_series