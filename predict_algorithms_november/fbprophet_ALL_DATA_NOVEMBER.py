import pandas as pd
import logging
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# read csv
# november data
data = pd.read_csv("https://raw.githubusercontent.com/iulianastroia/csv_data/master/final_dataframe.csv")

# drop Nan columns and indexes
data.dropna(axis='columns', how='all', inplace=True)
data.dropna(axis='index', how='all', inplace=True)

# convert to date format
data['day'] = pd.to_datetime(data['day'], dayfirst=True)

# sort dates by readable time
data = data.sort_values(by=['readable time'])
print("sorted days", data.day)

# modify name with any sensor name from df
sensor_name = 'pm25'

# drop unnecessary columns(all but day and sensor_name)
print("DROP", data.columns)
cols_to_drop = ['time', 'latitude', 'longitude', 'altitude', 'pm10', 'co2', 'pressure', 'temperature', 'pm1', 'o3',
                'ch2o', 'readable time']

data = data.drop(cols_to_drop, axis=1)

df = data.copy()
print("df_new", df.head())
print("df_col", df.columns)

# start using prophet
logging.getLogger().setLevel(logging.ERROR)

# create df for prophet
print("columns of initial df are:", df.columns)
df.columns = ['y', 'ds']
print("df columns created for prophet are: ", df.columns)

m = Prophet()
# fit dataframe to prophet algorithm
m.fit(df)

# make predictions for november, periods=0 because we don't want predictions for another month
future = m.make_future_dataframe(periods=0)
forecast = m.predict(future)

# print only values of interest, ds=date, yhat=forecasted value
# yhat_lower=low forecast value, yhat_upper=high forecast value
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# plot predictions
fig = plot_plotly(m, forecast)
fig.update_layout(
    title=sensor_name + ' forecast for November-December 2019',
    xaxis_title="Day",
    yaxis_title=sensor_name)
fig.show()

# check if there is seasonality+trend
fig = plot_components_plotly(m, forecast)
fig.update_layout(
    title=sensor_name + ' characteristics-seasonality'
)
fig.show()


# define a function to make a df containing the prediction and the actual values
def make_comparison_dataframe(historical, forecast):
    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))


cmp_df = make_comparison_dataframe(df, forecast)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# don't take january values into consideration at MAPE and MAE
cmp_df = cmp_df.dropna()
print("MAPE ", mean_absolute_percentage_error(cmp_df['y'], cmp_df['yhat']))
print("MAE", mean_absolute_error(cmp_df['y'], cmp_df['yhat']))

forecast_errors = [abs(cmp_df['y'][i] - cmp_df['yhat'][i]) for i in range(len(cmp_df))]
print('Forecast Errors: ', forecast_errors)
print('MAX Forecast Error: %s' % max(forecast_errors))
print('MIN Forecast Error: %s' % min(forecast_errors))
print("MSE(mean squared error)", mean_squared_error(cmp_df['y'], cmp_df['yhat']))
