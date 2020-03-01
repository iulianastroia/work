import pandas as pd
from datetime import datetime
import matplotlib.pylab as plt
from statsmodels.tsa.ar_model import AR, AR_DEPRECATION_WARN
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
import plotly.graph_objs as go
import warnings
import itertools

warnings.filterwarnings('ignore')


# predictions for 9 days(using MEAN values of November)
# convert date to datetime format
def parser(x):
    return datetime.strptime(x, '%d/%m/%Y')


# read csv file
data = pd.read_csv('https://raw.githubusercontent.com/iulianastroia/csv_data/master/final_dataframe.csv',
                   parse_dates=['day'], index_col='day', date_parser=parser)

# drop unnecessary columns
cols_to_drop = ['time', 'latitude', 'longitude', 'altitude', 'o3', 'co2', 'temperature', 'pm1', 'pm10', 'ch2o',
                'pressure', 'readable time']
data = data.drop(cols_to_drop, axis=1)
# print day and pm 2.5 values
print(data.head())

# group df by day
grp_date = data.groupby('day')

# calculate mean value of pm2.5  for every given day
data = pd.DataFrame(grp_date.mean())
print("MEAN pm25 values by day\n", data.pm25)
data.plot()
plt.title('Initial mean values for November')
plt.show()

# begin training
X = data.values
print("length of input values", len(X))
# ~70% of data->training
train = X[0:21]  # 21 data as train
print("length of train values", len(train))
# ~30% to test, 9 data as test
test = X[21:]
print("length of test values", len(test))
predictions = []

# train forecasting model
model_ar = AR(train)
model_ar_fit = model_ar.fit()

# predict
predictions = model_ar_fit.predict(start=21, end=30)
print("length of predictions", len(predictions))

# plot test data against predicted data
plt.plot(test, label="test data")
plt.plot(predictions, color='red', label='predicted data')
plt.legend(loc="upper left")
plt.show()

# ARIMA model
# p,d,q
# p=periods taken for autoregressive model:may=1; april, may=2
# d=integrated order, how many times difference is done
# q=no of periods in moving average model
model_arima = ARIMA(train, order=(1, 0, 0))
# model_arima = ARIMA(train, order=(4, 2, 0))
model_arima_fit = model_arima.fit()
# CHANGE VALUES (p,d,q) until AIC is MINIMUM
print("model_arima_fit.aic SHOULD HAVE minimum value", model_arima_fit.aic)

# predict 7 values
predictions = model_arima_fit.forecast(steps=9)[0]

p = d = q = range(0, 5)
# all combinations of p,d,q
pdq = list(itertools.product(p, d, q))
print('all pdq possible combinations', pdq)

for param in pdq:
    try:
        model_arima = ARIMA(train, order=param)
        model_arima_fit = model_arima.fit()
        # CHANGE VALUES (p,d,q) until AIC is MINIMUM
        print("param (p,d,q) and values of aic", param, model_arima_fit.aic)
    except:
        continue

print("test data ", test)
print("test data lenght", len(test))
print("predicted", predictions)
print("predicted", predictions)
print("length of predicted", len(predictions))

difference_true_pred = []
for i in range(0, len(test)):
    difference_true_pred.append(abs((test[i] - predictions[i])))
    min_Val = min(difference_true_pred)
    max_Val = max(difference_true_pred)
print("min is ", min_Val)
print("max is ", max_Val)

print(test)
print(type(test))

import numpy as np

group_by_df = pd.DataFrame(
    [name, group.mean().pm25] for name, group in data.groupby('day'))
group_by_df.columns = ['day', 'pm25']

prediction_df = group_by_df.copy()
prediction_df[:21] = np.nan
prediction_df.columns = ['day', 'pm25']

prediction_df.dropna(axis='columns', how='all', inplace=True)
prediction_df.dropna(axis='index', how='all', inplace=True)

prediction_df['pm25'] = predictions

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=group_by_df['day'],
    y=group_by_df['pm25'],
    name='Real value',
    mode='lines+markers'
))

fig.add_trace(go.Scatter(
    x=prediction_df['day'],
    y=prediction_df['pm25'],
    name='Predicted value'
))

fig.show()
print("MSE(mean squared error)", mean_squared_error(test, predictions))
