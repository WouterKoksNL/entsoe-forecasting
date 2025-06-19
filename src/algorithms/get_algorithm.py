from typing import Callable 

from .lstm import train_and_test_lstm

def get_algorithm(
        forecasting_model: str
        ) -> Callable:
    """
    Set the forecasting algorithm based on the provided model name.

    Args:
        forecasting_model (str): Name of the forecasting model to be used.

    Returns:
        Callable: The function that implements the specified forecasting algorithm.
    """
    if forecasting_model == "LSTM":
        return train_and_test_lstm
    else:
        raise NotImplementedError(f"Forecasting model '{forecasting_model}' is not implemented.")


