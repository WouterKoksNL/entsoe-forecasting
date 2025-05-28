# ENTSO-E forecasting

This code uses ENTSO-E zonal wind and load data, as well as day-ahead forecasts of these quantities, to create updated forecasts, which contain more recently revealed information. 
For instance, for a forecast made at 9.00, regarding the wind generation at 11.00, the model uses the day-ahead forecast for 11.00 as well as the observed wind generation up to 9.00. 
Currently two forecasting methods are implmented:
 * LSTM
 * Random forest regression

The data is separated into training and test data, and using the test data we extract the RMSE forecast error as a function of lead-time, and fit the curve. 

ToDo:
* Implement SARIMAX as a baseline
* Tune hyperparameters, and make hyperparameters inputs (with default values)
* Make code more flexible
* Return forecasting performance statistics
* Implement unittests
* Automatically check for outliers and handle NaNs better.
