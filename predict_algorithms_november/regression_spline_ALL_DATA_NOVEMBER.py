import datetime as dt
import math
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
from distributed.deploy.ssh import bcolors
from patsy.highlevel import dmatrix
from plotly import graph_objs as go
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def parser(x):
    return datetime.strptime(x, '%d/%m/%Y')


data = pd.read_csv('final_dataframe.csv',
                   date_parser=parser, parse_dates=['day'])
# data = pd.read_csv('final_dataframe.csv')
data = data.sort_values(by=['readable time'])

# modify name with any sensor name from df
sensor_name = 'pm25'

# give values of day(1->30), otherwise->overflow
X = data['day'].dt.day.values.reshape(-1, 1)
X = np.asmatrix(X)
y = data[sensor_name].values.reshape(-1, 1)
y = np.asmatrix(y)

# Splitting the dataset into training(70%) and test(30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    shuffle=False)


def analyse_forecast(dataframe_name, predicted_list, regression_type):
    # predicted_list = pol_reg.predict(poly_reg.fit_transform(X))
    predicted_list = [arr.tolist() for arr in predicted_list]
    print(bcolors.OKBLUE + "MSE " + regression_type + " regression(mean squared error)",
          mean_squared_error(dataframe_name[sensor_name], predicted_list), bcolors.ENDC)
    print("r2 score ", r2_score(dataframe_name[sensor_name], predicted_list))
    rmse = np.sqrt(mean_squared_error(dataframe_name[sensor_name], predicted_list))
    print(bcolors.WARNING + "RMS for " + regression_type + " regression=", rmse, bcolors.ENDC)
    return mean_squared_error(dataframe_name[sensor_name], predicted_list)


# create dataframe with mse values and corresponding polynomial grade
def mse_minumum(regression_type, mse_list_regression, max_grade_regression):
    mse_df = pd.DataFrame(mse_list_regression)
    mse_df.columns = ['mse_values']
    mse_df[regression_type + '_grade'] = [i + 1 for i in range(0, max_grade_regression)]
    print(bcolors.OKBLUE + "minimum MSE for given " + regression_type + " grades:",
          mse_df[mse_df['mse_values'] == mse_df['mse_values'].min()], bcolors.ENDC)


# calculate maximum polynomial grade
max_grade = math.floor(math.sqrt(len(data)))
# data['day'] = data['day'].dt.day.values

# create list to store mse for every given polynomial grade
data['day'] = data['day'].map(dt.datetime.toordinal)

# calculate and plot spline regression
# calculate 25%,50% and 75% percentiles
percentile_25 = np.percentile(data['day'], 25)
percentile_50 = np.percentile(data['day'], 50)
percentile_75 = np.percentile(data['day'], 75)

fig_spline = go.Figure()
mse_list_spline = []
for count, degree in enumerate([i + 1 for i in range(0, max_grade)]):
    # Specifying 3 knots for regression spline
    transformed_x1 = dmatrix(
        "bs(data.day, knots=(percentile_25,percentile_50,percentile_75), degree=degree, include_intercept=False)",
        {"data.day": data.day}, return_type='dataframe')

    # build a regular linear model from the splines
    fit_spline = sm.GLM(data[sensor_name], transformed_x1).fit()

    # make predictions
    pred_spline = fit_spline.predict(transformed_x1)

    print('\ngrade for regression spline: ', degree)
    mse_list_spline.append(analyse_forecast(data, pred_spline.values, "spline"))

    fig_spline.add_trace(go.Scatter(
        x=data['day'].map(dt.datetime.fromordinal),
        y=pred_spline,
        name="Predicted values grade " + str(degree),
        mode='lines+markers'
    ))

fig_spline.add_trace(go.Scatter(
    x=data['day'].map(dt.datetime.fromordinal),
    y=data[sensor_name],
    name='Actual values',
    mode='lines+markers'))

fig_spline.update_layout(
    title="Regression Spline for " + sensor_name,
    yaxis_title=sensor_name,
    xaxis_title='Day(time)',
    showlegend=True)
fig_spline.show()
mse_minumum("spline", mse_list_spline, max_grade)
