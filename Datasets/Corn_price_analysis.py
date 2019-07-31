import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_finance import candlestick_ohlc
from statsmodels.tsa.seasonal import seasonal_decompose


#Line graph style with bollinger band, using 20 day mean as a basis for those
df = pd.read_csv('/Users/snowwhite/Github/Corn/Datasets/Data.csv', index_col='Date',parse_dates=True)
ohlc = df[['Open', 'High', 'Low','Price']].copy()
ohlc = ohlc.iloc[::-1].tail(500) # To reduce the data shown on the plot
ohlc['Price: 20 Day Mean'] = ohlc['Price'].rolling(window=20).mean()
ohlc['Upper'] = ohlc['Price: 20 Day Mean'] + 2*ohlc['Price'].rolling(window=20).std()
ohlc['Lower'] = ohlc['Price: 20 Day Mean'] - 2*ohlc['Price'].rolling(window=20).std()
ohlc[['Price','Price: 20 Day Mean','Upper','Lower']].plot.line(figsize=(16,6))
#plt.show() #if all the codes are running, no need for double show

# Candlestick style
df1 = pd.read_csv('/Users/snowwhite/Github/Corn/Datasets/Data.csv')

df1['Date'] = pd.to_datetime(df1['Date'])
df1["Date"] = df1["Date"].apply(mdates.date2num)
ohlc = df1[['Date','Open', 'High', 'Low','Price']].copy()
ohlc = ohlc.iloc[::-1].tail(200) # To reduce the data shown on the plot

fig, ax = plt.subplots(figsize = (10,5))
candlestick_ohlc(ax, ohlc.values, width=.6, colorup='green', colordown='red')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
#plt.show() #if all the codes are running, no need for double show

#ETS_analysis of the data
df2 = pd.read_csv('/Users/snowwhite/Github/Corn/Datasets/Data.csv', index_col='Date')
df2.index = pd.to_datetime(df2.index)
#df2 = df2.iloc[::-1].tail(2000) # no need for data cut, as seasonality test is better run on huge dataset. 
#print(df2.head())
ETS_model = seasonal_decompose(df2['Price'], model='additive', freq=365) # seasonality freq set as 1 year, seems reasonable for Corn production
ETS_model.plot()

plt.show()