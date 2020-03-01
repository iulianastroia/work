import pandas as pd
import logging
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go
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

# sort dates by day
data = data.sort_values(by=['day'])
print("sorted days", data.day)

group_by_df = pd.DataFrame(
    [name, group.mean().pm25] for name, group in data.groupby('day')
)
group_by_df.columns = ['day', 'pm25']

# plot groupby df
fig = go.Figure(data=go.Scatter(x=group_by_df['day'], y=group_by_df['pm25']))
fig.update_layout(
    title='Pm2.5 REAL mean values for November',
    xaxis_title="Day",
    yaxis_title="Pm2.5")
fig.show()

# group df by day
grp_date = data.groupby('day')

# calculate mean value  for every given day
data = pd.DataFrame(grp_date.mean())
print("MEAN pm25 values by day\n", data.pm25)

# drop unnecessary columns(all but day and pm25)
cols_to_drop = ['time', 'latitude', 'longitude', 'altitude', 'o3', 'pressure', 'temperature', 'pm1', 'pm10', 'ch2o',
                'co2']
data = data.drop(cols_to_drop, axis=1)
print(data.head())

# start using prophet
logging.getLogger().setLevel(logging.ERROR)

# create df for prophet
df = data.reset_index()
# ds=date, y=pm2.5 values
df.columns = ['ds', 'y']
print(df)

m = Prophet()
m.fit(df)

# make predictions for november, periods=0 because we don't want predictions for another month
future = m.make_future_dataframe(periods=0)
forecast = m.predict(future)
# print only values of interest
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# plot predictions
fig = plot_plotly(m, forecast)
fig.update_layout(
    title='Pm2.5 forecast for November-December 2019',
    xaxis_title="Day",
    yaxis_title="Pm2.5")
fig.show()

# check if there is seasonality+trend
fig = plot_components_plotly(m, forecast)
fig.update_layout(
    title='Pm2.5 characteristics-seasonality'
)
fig.show()


# define a function to make a df containing the prediction and the actual values
def make_comparison_dataframe(historical, forecast):
    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))


cmp_df = make_comparison_dataframe(df, forecast)
print("COMPARED DF of predicted and real values\n", cmp_df)

# plot forecast with upper and lower bound
fig = go.Figure()

# predicted value
fig.add_trace(go.Scatter(
    x=group_by_df['day'],
    y=cmp_df['yhat'],
    name='yhat(predicted value)',
    mode='lines+markers'
))

# lower bound of predicted value
fig.add_trace(go.Scatter(
    x=group_by_df['day'],
    y=cmp_df['yhat_lower'],
    name='yhat_lower',
    mode='lines+markers'

))

# upper bound of predicted value
fig.add_trace(go.Scatter(
    x=group_by_df['day'],
    y=cmp_df['yhat_upper'],
    name='yhat_upper',
    mode='lines+markers'

))

# actual value
fig.add_trace(go.Scatter(
    x=group_by_df['day'],
    y=cmp_df['y'],
    name='y(actual value)',
    mode='lines+markers'

))
fig.update_layout(title='Comparison between predicted values and real ones', yaxis_title='Pm2.5', xaxis_title='Day',
                  showlegend=True)
fig.show()


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
