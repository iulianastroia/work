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
print("LEN", len(data))
# drop Nan columns and indexes
data.dropna(axis='columns', how='all', inplace=True)
data.dropna(axis='index', how='all', inplace=True)

# convert to date format
data['day'] = pd.to_datetime(data['day'], dayfirst=True)

# modify name with any sensor name from df
sensor_name = 'pm25'

# sort dates by day
data = data.sort_values(by=['day'])
print("sorted days", data.day)

group_by_df = pd.DataFrame(
    [name, group.mean()[sensor_name]] for name, group in data.groupby('day')
)
group_by_df.columns = ['day', sensor_name]

# plot groupby df
fig = go.Figure(data=go.Scatter(x=group_by_df['day'], y=group_by_df[sensor_name]))
fig.update_layout(
    title=sensor_name + ' REAL mean values for November',
    xaxis_title="Day",
    yaxis_title=sensor_name)
fig.show()

# group df by day
grp_date = data.groupby('day')

# calculate mean value  for every given day
data = pd.DataFrame(grp_date.mean())
print("MEAN " + sensor_name + " values by day\n", data[sensor_name])

# select needed data
data = data[[sensor_name]]

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
    title=sensor_name + ' forecast for November-December 2019',
    xaxis_title="Day",
    yaxis_title=sensor_name)
fig.show()

# check if there is seasonality+trend
fig = plot_components_plotly(m, forecast)
fig.update_layout(
    title=sensor_name + " characteristics-seasonality"
)
fig.show()


# define a function to make a df containing the prediction and the actual values
def make_comparison_dataframe(historical, forecast):
    return forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']].join(historical.set_index('ds'))


cmp_df = make_comparison_dataframe(df, forecast)
print("COMPARED DF of predicted and real values\n", cmp_df)

# add new column with default value
cmp_df['outlier_detected'] = 0
for i in range(len(cmp_df)):
    # detect outliers
    if (cmp_df['y'][i] > cmp_df['yhat_upper'][i] or cmp_df['y'][i] < cmp_df['yhat_lower'][i]):
        cmp_df['outlier_detected'][i] = 1
    else:
        cmp_df['outlier_detected'][i] = 0
print("DF of outlier", cmp_df)
print(["outlier DET" for i in range(len(cmp_df)) if cmp_df['outlier_detected'][i] == 1])

# plot forecast with upper and lower bound
fig = go.Figure()

# predicted value
fig.add_trace(go.Scatter(
    x=group_by_df['day'],
    y=cmp_df['yhat'],
    name='yhat(predicted value)',
    mode='lines+markers',
    line=dict(
        color='rgb(95,158,160)'),
    marker=dict(
        color='rgb(95,158,160)')
))

# actual value
fig.add_trace(go.Scatter(
    x=group_by_df['day'],
    y=cmp_df['y'],
    name='y(actual value)',
    mode='lines+markers',
    line=dict(
        color='rgb(75,0,130)'),
    marker=dict(color=np.where(cmp_df['outlier_detected'] == 1, 'red', 'rgb(75,0,130)'))))

# lower bound of predicted value
fig.add_trace(go.Scatter(
    x=group_by_df['day'],
    y=cmp_df['yhat_lower'],
    name='yhat_lower',
    mode='lines+markers',
    line=dict(
        color='rgb(205,92,92)'),
    marker=dict(
        color='rgb(205,92,92)')

))

# upper bound of predicted value
fig.add_trace(go.Scatter(
    x=group_by_df['day'],
    y=cmp_df['yhat_upper'],
    name='yhat_upper',
    mode='lines+markers',
    line=dict(
        color='rgb(65,105,225)'),
    marker=dict(
        color='rgb(65,105,225)')
))

fig.update_layout(title='Comparison between predicted values and real ones', yaxis_title=sensor_name, xaxis_title='Day',
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
