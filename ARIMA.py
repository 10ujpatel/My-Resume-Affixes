# -*- coding: utf-8 -*-
"""
Created on Fri May 19 12:08:03 2023

@author: Tanuj
"""

#ARIMA - AutoRegression Integrated Moving Average
#Importing Necessary Packages
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
#Import Dataset
os.chdir(r"E:\Loyola College\SEM IV\Multivariate Data Analysis\Full MVA\Unit 4\ts 5 data sets")
df=pd.read_csv("Sovenir.csv",index_col=('month_year'),parse_dates=True)
print(df)
df=df.drop(columns='obsno')
print(df)
print(df.columns)
#Dropping Null Values (if any)
df=df.dropna(axis=0)
df.plot()
df.sales.plot(title='Sales Plot')
#Calculating AutoCorrelation Function
acf=sm.tsa.stattools.acf(df['sales'],nlags=20)
#Plotting ACF
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(df['sales'], lags=20)
#p=2 - Lag for Auto Regression
#Calculating Partial AutoCorrelation Function
pacf=sm.tsa.stattools.pacf(df['sales'],nlags=20,method='ols')
#Plotting PACF
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(df['sales'],lags=20)
#q=2 - Lag for Moving Average
#Checking Stationarity of TSA p<0.05 -> Stationary  kps>0.05 -> Stationary
#Perform ADF test for Stationarity
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['sales'])
#Print the Results
print('ADF Statistic: %f' % result[0])
print('p_value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
# the test result can be interpreted as follows:
if p_value < 0.05:
    print('The time series is stationary around a deterministic trend.')
else:
    print('The time series is not stationary around a deterministic trend.')
#Fit ARIMA Model
model=sm.tsa.ARIMA(df['sales'],order=(2,1,2)).fit()
print(model.summary())
"""
 specifies the order of the ARIMA model, where the first parameter is the 
 autoregressive order (p), the second parameter is the degree of differencing 
 (d), and the third parameter is the moving average order (q).
"""
#Making Forecasts
forecasts=model.forecast(steps=10)
#Time Series Plot
plt.plot(df['sales'], color='blue', label='Original')
plt.plot(forecasts,color="red",label="Forecasted")
plt.legend()
plt.title('Time Series Plot')
plt.show()
