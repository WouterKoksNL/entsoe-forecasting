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

def get_hourly(ser, set_hour_indices=True):
    selection = ser.loc[ser.index.minute == 0]
    if set_hour_indices:
        selection.index = selection.index.hour
    return selection

def select_day(ser, year, month, day):
    return ser.loc[(ser.index.year == year) & (ser.index.month == month) & (ser.index.day == day)]