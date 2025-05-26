# ENTSO-E forecasting

This code uses ENTSO-E zonal wind and load data, as well as day-ahead forecasts of these quantities, to create updated forecasts, which contain more recently revealed information. 
For instance, for a forecast made at 9.00, regarding the wind generation at 11.00, the model uses the day-ahead forecast for 11.00 as well as the observed wind generation up to 9.00. 
LSTM and random forest regression are implemented as forecasting methods. The data is separated into training and test data, and using the test data we extract the RMSE forecast error as a function of lead-time, and fit the curve. 
