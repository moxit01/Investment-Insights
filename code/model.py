import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from alpha_vantage.timeseries import TimeSeries
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle

# Load data from Alpha Vantage
api_key = '8U8OGK31EO6CESRW'
ts = TimeSeries(key=api_key, output_format='pandas')
data, meta_data = ts.get_daily(symbol='GOOGL', outputsize='full')

# Preprocess data
data.index = pd.to_datetime(data.index)
data = data.sort_values(by='date')

# Split into train and test sets
size = int(len(data) * 0.66)
train, test = data[0:size], data[size:len(data)]
history = [x for x in train['4. close']]
predictions = list()

# Train and test ARIMA model
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test['4. close'].iloc[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
    with open('pickle/arima_model.pkl', 'wb') as f:
        pickle.dump(model_fit, f)

# Evaluate forecasts
rmse = sqrt(mean_squared_error(test['4. close'], predictions))
print('Test RMSE: %.3f' % rmse)
