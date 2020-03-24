import logging

import numpy as np
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None

# data = pd.read_csv("final_dataframe.csv")

data = pd.read_csv("https://raw.githubusercontent.com/iulianastroia/csv_data/master/march_data.csv")

# drop Nan columns and indexes
data.dropna(axis='columns', how='all', inplace=True)
data.dropna(axis='index', how='all', inplace=True)

# convert to date format
data['day'] = pd.to_datetime(data['day'], dayfirst=True)

# modify name with any sensor name from df
sensor_name = 'ch2o'

# sort dates by day
data = data.sort_values(by=['day'])
print("sorted days", data.day)

group_by_df = pd.DataFrame(
    [name, group.mean()[sensor_name]] for name, group in data.groupby('day')
)

group_by_df.columns = ['day', sensor_name]

# group df by day
grp_date = data.groupby('day')
# calculate mean value  for every given day
data = pd.DataFrame(grp_date.mean())
print("MEAN " + sensor_name + " values by day\n", data[sensor_name])

# select needed data
data = data[[sensor_name]]

# boxplot values to eliminate outliers
upper_quartile = np.percentile(data[sensor_name], 75)
lower_quartile = np.percentile(data[sensor_name], 25)

iqr = upper_quartile - lower_quartile
upper_whisker = data[sensor_name][data[sensor_name] <= upper_quartile + 1.5 * iqr].max()
lower_whisker = data[sensor_name][data[sensor_name] >= lower_quartile - 1.5 * iqr].min()

# todo eliminate outliers detected by boxplot
# data = data.loc[
#     (data[sensor_name] >= lower_whisker) & (data[sensor_name] <= upper_whisker)]
# print("new df issss", data)
# todo eliminate outliers detected by boxplot
# group_by_df = group_by_df.loc[
#     (group_by_df[sensor_name] >= lower_whisker) & (group_by_df[sensor_name] <= upper_whisker)]
# print("new df is", group_by_df)


# start using prophet
logging.getLogger().setLevel(logging.ERROR)

# create df for prophet
df = data.reset_index()

df.columns = ['ds', 'y']

X = group_by_df[['day']].values
y = group_by_df[[sensor_name]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# todo use X_train and y_test for march(test values from 11 March)
# X_train = X[:10]
# y_train = y[:10]
# X_test = X[10:]
# y_test = y[10:]

# create dataframe containing only train values
dff = pd.DataFrame(index=range(0, len(y_train)))

dff['ds'] = group_by_df['day'][:len(y_train)]
dff['y'] = group_by_df[sensor_name][:len(y_train)]

m = Prophet()
# fit train values to prophet
m.fit(dff)

# predict whole month
future = m.make_future_dataframe(periods=len(y_test))
forecast = m.predict(future)
print('forecast', forecast)
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


# modify dff so that mse can be calculated for each value of the dataframe
dff['ds'] = group_by_df['day']
dff['y'] = group_by_df[sensor_name]
# cmp_df = make_comparison_dataframe(df, forecast)
cmp_df = make_comparison_dataframe(df, forecast)

# add new column with default value
cmp_df['outlier_detected'] = 0
for i in range(len(cmp_df)):
    # detect outliers
    if (cmp_df['y'][i] > cmp_df['yhat_upper'][i] or cmp_df['y'][i] < cmp_df['yhat_lower'][i]):
        cmp_df['outlier_detected'][i] = 1
    else:
        cmp_df['outlier_detected'][i] = 0

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


cmp_df = cmp_df.dropna()

forecast_errors = [abs(cmp_df['y'][i] - cmp_df['yhat'][i]) for i in range(len(cmp_df))]
print('Forecast Errors: ', forecast_errors)
print('MAX Forecast Error: %s' % max(forecast_errors))
print('MIN Forecast Error: %s' % min(forecast_errors))

rmse = np.sqrt(mean_squared_error(cmp_df['y'], cmp_df['yhat']))
print("MSE is ", mean_squared_error(cmp_df['y'], cmp_df['yhat']))
print("rmse is ", rmse)
print("r2 score ", r2_score(cmp_df['y'], cmp_df['yhat']))  # around 1


def correlation_line(df, x, y):
    scatter_data = go.Scattergl(
        x=df[x],
        y=df[y],
        mode='markers',
        name=x + ' and ' + y + ' correlation'
    )

    layout = go.Layout(
        xaxis=dict(
            title=x
        ),
        yaxis=dict(
            title=y)
    )

    # calculate best fit line
    denominator = (df[x] ** 2).sum() - df[x].mean() * df[x].sum()
    print('denominator', denominator)
    m = ((df[y] * df[x]).sum() - df[y].mean() * df[x].sum()) / denominator
    b = ((df[y].mean() * ((df[x] ** 2).sum())) - df[x].mean() * ((df[y] * df[x]).sum())) / denominator
    best_fit_line = m * df[x] + b

    best_fit_line = go.Scattergl(
        x=df[x],
        y=best_fit_line,
        name='Line of best fit',
        line=dict(
            color='red'
        )
    )

    data = [scatter_data, best_fit_line]
    figure = go.Figure(data=data, layout=layout)

    figure.show()


# yhat and y
correlation_line(cmp_df, cmp_df.columns[0], cmp_df.columns[3])
