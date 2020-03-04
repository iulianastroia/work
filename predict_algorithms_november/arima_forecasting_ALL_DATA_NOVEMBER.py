import pandas as pd
from datetime import datetime
import matplotlib.pylab as plt
from pmdarima.model_selection import train_test_split
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
import warnings
import numpy as np

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

# calculate mean value of pm2.5  for every given day
print("MEAN pm25 values by day\n", data.pm25)
data.plot()
plt.title('Initial mean values for November')
plt.show()
#
# # begin training
X = data.values
print("length of input values", len(X))

y_train, y_test = train_test_split(X, test_size=0.3
                                   )

print("length of train values", len(y_train))

print("length of test values", len(y_test))
predictions = []

model_ar = AR(y_train)
model_ar_fit = model_ar.fit()

predictions = model_ar_fit.predict(start=len(y_train), end=len(data))
print("length of predictions", len(predictions))

# TODO try all possibilities of (p,d,q)
model_arima = ARIMA(y_train, order=(1, 0, 4))
model_arima_fit = model_arima.fit()

predictions = model_arima_fit.forecast(steps=len(y_test))[0]
print("DATA", data.pm25)

prediction_df = data.copy()
prediction_df[:len(y_train)] = np.nan
prediction_df.columns = ['pm25']

prediction_df.dropna(axis='columns', how='all', inplace=True)
prediction_df.dropna(axis='index', how='all', inplace=True)

prediction_df['pm25'] = predictions

print("MSE(mean squared error)", mean_squared_error(y_test, predictions))
