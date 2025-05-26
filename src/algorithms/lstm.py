import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler


def create_multivariate_supervised(target, forecast, past_forecast_error=None, target_series_hour=None, n_lags=24, n_lead_time=8):
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
        
        if past_forecast_error is not None:
            past_forecast_error_values = past_forecast_error[i : i + n_lags]
            input_data_list.append(past_forecast_error_values)

        features = np.concatenate(input_data_list)
        X.append(features)
        y.append(target_value)
    
    return np.array(X), np.array(y)

def train_and_test_lstm(target_series, forecast_series, past_forecast_error=None, target_series_hour=None, n_lags=24, n_lead_time=8, train_test_split=0.8, epochs=20):
    # Create dataset
    X, y = create_multivariate_supervised(target_series, forecast_series, past_forecast_error, target_series_hour, n_lags, n_lead_time)

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
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Train
    model.fit(X_train, y_train, epochs=epochs, verbose=1)

    # Predict and inverse scale
    y_pred_scaled = model.predict(X_test)
    y_test_inv = scaler_y.inverse_transform(y_test)
    y_pred_inv = scaler_y.inverse_transform(y_pred_scaled)

    return y_test_inv.ravel(), y_pred_inv.ravel()