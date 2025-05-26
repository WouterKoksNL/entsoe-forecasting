
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def create_multivariate_supervised(target, forecast, past_forecast_error: None | np.ndarray, n_lags=24, n_lead_time=8):
    X, y = [], []
    for i in range(len(target) - n_lags - n_lead_time + 1):
        past_values = target[i : i + n_lags]
        future_forecast = forecast[i + n_lags : i + n_lags + n_lead_time]  # still include full future forecast i + n_lags : i + n_lags + n_lead_time
        target_value = target[i + n_lags + n_lead_time - 1]  # pick only one target value

        if past_forecast_error is not None:
            past_forecast_error_values = past_forecast_error[i : i + n_lags]
            X.append(np.concatenate([past_values, future_forecast, past_forecast_error_values]))
        else:
            X.append(np.concatenate([past_values, future_forecast]))
        
        y.append(target_value)
    
    return np.array(X), np.array(y)


def train_and_test_random_forest(target_series, forecast_series, past_forecast_error=None, n_lags=24, n_lead_time=8, train_test_split=0.8):
    X, y = create_multivariate_supervised(target_series, forecast_series, past_forecast_error, n_lags=n_lags, n_lead_time=n_lead_time)
    split = int(train_test_split * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred