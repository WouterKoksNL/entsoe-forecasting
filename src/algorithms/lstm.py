import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


def create_multivariate_supervised(target, forecast, past_forecast_error=None, reduced_forecast_error_usability_entry=None, target_series_hour=None, target_series_month=None, n_lags=24, n_lead_time=8):
    X, y = [], []
    for i in range(len(target) - n_lags - n_lead_time + 1):
        # current forecasting time index: i + n_lags 
        # past values: target[i : i + n_lags] (1D array)
        # forecast the value at i + n_lags + n_lead_time - 1

        past_values = target[i : i + n_lags]  # 1D array
        future_forecast = forecast[i + n_lags + n_lead_time - 1]  # scalar
        target_value = target[i + n_lags + n_lead_time - 1]       # scalar

        input_data_list = [past_values, np.array([future_forecast])]  # ensure 1D

        if target_series_hour is not None:
            target_hour = target_series_hour[i + n_lags + n_lead_time - 1]
            input_data_list.append(np.array([target_hour]))  # ensure 1D
        
        if target_series_month is not None:
            target_month = target_series_month[i + n_lags + n_lead_time - 1]
            input_data_list.append(np.array([target_month]))  # ensure 1D

        if past_forecast_error is not None:
            past_forecast_error_values = past_forecast_error[i : i + n_lags]
            input_data_list.append(past_forecast_error_values)
        
        if reduced_forecast_error_usability_entry is not None:
            reduced_forecast_error_values = reduced_forecast_error_usability_entry[i : i + n_lags]
            input_data_list.append(reduced_forecast_error_values)
        features = np.concatenate(input_data_list)
        X.append(features)
        y.append(target_value)
    
    return np.array(X), np.array(y)

def train_and_test_lstm(
        target_series, 
        forecast_series,
        past_forecast_error=None, 
        reduced_forecast_error_usability_entry=None, 
        target_series_hour_sin=None, 
        target_series_yearly_sin=None, 
        n_lags=8, 
        n_lead_time=8, 
        train_test_split=0.8, 
        epochs=20, 
        learning_rate=0.001,
        lstm_activation='tanh',  # Activation function for LSTM
        output_activation=None,  # Activation function for output layer, None for linear output
        lstm_units:int=50,  # Number of LSTM units
        ):
    # Create dataset
    print("Training LSTM model for n_lead_time =", n_lead_time)
    X, y = create_multivariate_supervised(target_series, forecast_series, past_forecast_error, reduced_forecast_error_usability_entry, target_series_hour_sin, target_series_yearly_sin, n_lags, n_lead_time)

    # Scale features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # Reshape for LSTM input: (samples, timesteps, features)
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Train/test split
    split = int(train_test_split * len(X_scaled))
    X_train, X_test = X_scaled[:split], X_scaled[split:]
    y_train, y_test = y_scaled[:split], y_scaled[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(lstm_units, activation=lstm_activation),
        # Dense(32, activation='relu'),
        # Dense(32, activation='relu'),
        Dense(1, activation=output_activation),  
    ])

    optimizer = Adam(learning_rate=learning_rate)  # Typical default is 0.001, can be lowered to improve convergence

    # Re-compile with the customized optimizer
    model.compile(optimizer=optimizer, loss='mse')
    # Train
    model.fit(X_train, y_train, epochs=epochs, verbose=0)

    # Predict and inverse scale
    y_pred_scaled = model.predict(X_test)
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred_scaled)

    return y_test_inv.ravel(), y_pred_inv.ravel()