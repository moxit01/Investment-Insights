from flask import Flask,render_template,request, render_template_string
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import urllib
import base64
from textblob import TextBlob
import json
from tweepy import OAuthHandler, API, Client
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from alpha_vantage.timeseries import TimeSeries
from sklearn.metrics import mean_squared_error
from math import sqrt
import os
import uuid
from flask import send_from_directory

app = Flask(__name__)



@app.route('/')
def home():

    return render_template('index.html')



@app.route('/news_sentiment', methods=['GET', 'POST'])
def news_sentiment():
    if request.method == 'POST':
        symbol = request.form.get('symbol')
        api_key = '8U8OGK31EO6CESRW'
        url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={api_key}'
        r = requests.get(url)
        data = r.json()

        # Extract the feed items from the data
        feeds = data.get('feed', [])


        return render_template('news_sentiment.html', feeds=feeds)

    return render_template('news_sentiment.html')



@app.route('/visuals', methods=['GET', 'POST'])
def visuals():

    if request.method == 'POST':
        symbol = request.form.get('symbol')
        api_key = "8U8OGK31EO6CESRW"
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
        data.index = pd.to_datetime(data.index)
        data = data.sort_values(by='date')

        # Split into train and test sets
        size = int(len(data) * 0.66)
        train, test = data[0:size], data[size:len(data)]
        history = [x for x in train['4. close']]
        predictions = list()

    # Train and test ARIMA model
    # for t in range(len(test)):
    #     model = ARIMA(history, order=(5,1,0))
    #     model_fit = model.fit()
    #     output = model_fit.forecast()
    #     yhat = output[0]
    #     predictions.append(yhat)
    #     obs = test['4. close'].iloc[t]
    #     history.append(obs)
        for t in range(min(20,len(test))):
             model = ARIMA(history, order=(5,1,0))
             model_fit = model.fit()
             output = model_fit.forecast()
             yhat = output[0]
             predictions.append(yhat)
             obs = test['4. close'].iloc[20]
             history.append(obs)

    # Evaluate forecasts
        rmse = sqrt(mean_squared_error(test['4. close'].iloc[:20], predictions))
    
    # Plot the actual observations
        plt.plot(test['4. close'].values[:len(predictions) - 3], label='Actual')
        plt.plot(predictions, color='red', label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.title(f'{symbol} Stock Price Prediction')
        plt.legend()
    
    # Save the plot to a file
        # unique_filename = str(uuid.uuid4()) + '.png'
        # plot_path = os.path.join('static', unique_filename)
        plot_path = os.path.join('static', f'{symbol}_plot.png')
        plt.savefig(plot_path)
        plt.close()


        residuals = [test['4. close'].iloc[i] - predictions[i] for i in range(len(predictions))]
        plt.figure(figsize=(12, 6))
        plt.plot(residuals)
        plt.title('Residual Errors')
        plt.xlabel('Time')
        plt.ylabel('Residual Error')
        plt.axhline(0, color='black', linestyle='--') 

        plot_path2 = os.path.join('static', f'{symbol}_plot2.png')
        plt.savefig(plot_path2)
        plt.close()


    

    return render_template('visuals.html')

@app.route('/model', methods=['GET', 'POST'])
def model():
    if request.method == 'POST':
        symbol = request.form['symbol']
        model = joblib.load('pickle/arima_model.pkl')
        prediction = model.predict(symbol)
        return render_template('model.html', prediction=prediction)
    return render_template('model.html')

# @app.route('/plot/<filename>')
# def serve_plot(filename):
#     return send_from_directory('static', filename)



if __name__ == '__main__':
    app.run(debug=True)
