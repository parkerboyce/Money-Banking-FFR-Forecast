import pandas as pd
import matplotlib.pyplot as plt
from fredapi import Fred
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
fred = Fred(api_key='770c577025475feebad2a471e8985218')

ffr_data = fred.get_series('FEDFUNDS', start_date = '2014-01-01')
ffr_filtered = ffr_data[ffr_data.index >= pd.to_datetime('2014-01-01')]
ffr_data.dropna(inplace = True)
ffr_data.index = pd.to_datetime(ffr_data.index)

adf_result = adfuller(ffr_data)
if adf_result[1] > 0.05:
    ffr_data = ffr_data.diff().dropna()

model = ARIMA(ffr_data, order = (1, 1, 1))
model_fit = model.fit()
#print(model_fit.summary())

forecast = model_fit.forecast(steps = 12)
forecast.index = pd.date_range(ffr_filtered.index[-1] + pd.Timedelta('30D'), periods=12, freq = 'M')
plt.plot(ffr_filtered, label = "Historical FFR")
plt.plot(forecast, label = "12-Month Forecast", color = 'red')
plt.title("12-Month Forecast of the FFR")
plt.xlabel("Date")
plt.ylabel("FFR (%)")
plt.legend()
plt.show()