from src.load_config import ConfigSettings
from src.fitting import calc_fitting_params_loop
from src.forecasting import get_forecast_loop
import pandas as pd
from src.utils import select_day
from src.forecasting import get_correct_time_indices
from src.forecast_data_handling import get_data
from src.post_process import restructure_forecast_df
from src.utils import make_dir
from src.plotting.rmse import plot_rmse
    
def main(config_yaml_name: str, day: int, month: int, year: int):
    config = ConfigSettings(config_yaml_name)
    # get_forecast_loop(config)
    param_dict, rmse_dict = calc_fitting_params_loop(config=config)
    plot_rmse(rmse_dict, 'ES')

    for error_type in config.error_types:
        for zone in config.zones_error_types[error_type]:
            y_test_dict, y_pred_dict, scaling_value, target_series, forecast_series = get_data(forecast_pickle_dir=f"data/pickled_forecasts/{config.run_id}", zone=zone, error_type=error_type)
            target_series = pd.Series(target_series.iloc[:, 0])    
            forecast_series = pd.Series(forecast_series.iloc[:, 0])
            pred_dict = get_correct_time_indices(y_pred_dict, target_series.index, config.max_lead_time)


            pred_dict_sel = {lead_time: select_day(pred, day=day, month=month, year=year) for lead_time, pred in pred_dict.items()}
            pred_df = pd.DataFrame(pred_dict_sel).clip(0)

            target_sel = select_day(target_series, day=day, month=month, year=year)
            forecast_sel = select_day(forecast_series, day=day, month=month, year=year)


            forecasts_df = restructure_forecast_df(
                target_sel,
                forecast_sel,
                pred_df,
                n_steps=1,
            )
            make_dir(f"output/{config.run_id}/{zone}/{error_type}")
            forecasts_df.index.name = 'delivery_time'
            forecasts_df.to_csv(f"output/{config.run_id}/{zone}/{error_type}/{day}_{month}_{year}.csv")




if __name__ == "__main__":
    main(config_yaml_name='spain.yaml', day=2, month=12, year=2024)
    main(config_yaml_name='spain.yaml', day=8, month=7, year=2024)
