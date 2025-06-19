import os 
import pandas as pd

def make_dir(path):
    """
    Create a directory if it does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def convert_index_to_datetime(ser):
    ser.index = pd.to_datetime(ser.index, utc=True)
    return ser

def get_hourly(series):
    return series.loc[series.index.minute == 0]