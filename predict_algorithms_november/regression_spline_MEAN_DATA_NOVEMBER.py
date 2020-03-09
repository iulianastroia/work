import datetime as dt
import math
from math import sqrt

import numpy as np
import pandas as pd
import statsmodels.api as sm
from distributed.deploy.ssh import bcolors
from pandas.plotting import register_matplotlib_converters
from patsy import dmatrix
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

register_matplotlib_converters()

# read csv file
data = pd.read_csv("final_dataframe.csv")

# convert day to pandas datetime format
data['day'] = pd.to_datetime(data['day'], dayfirst=True)

# modify name with any sensor name from df
sensor_name = 'pm25'

# sort values by day
data = data.sort_values(by=['readable time'])

# create mean of values by days
group_by_df = pd.DataFrame([name, group.mean()[sensor_name]] for name, group in data.groupby('day'))
group_by_df.columns = ['day', sensor_name]

group_by_df['day'] = pd.to_datetime(group_by_df['day'])

# convert day column to needed(supported) date format
group_by_df['day'] = group_by_df['day'].map(dt.datetime.toordinal)


# eliminate outliers that aren't in the quartiles(optional; if we want to remove outliers or not)
def remove_outliers(dataframe_name):
    upper_quartile = np.percentile(dataframe_name[sensor_name], 75)
    lower_quartile = np.percentile(dataframe_name[sensor_name], 25)

    iqr = upper_quartile - lower_quartile
    upper_whisker = dataframe_name[sensor_name][dataframe_name[sensor_name] <= upper_quartile + 1.5 * iqr].max()
    lower_whisker = dataframe_name[sensor_name][dataframe_name[sensor_name] >= lower_quartile - 1.5 * iqr].min()

    # eliminate outliers detected by boxplot
    dataframe_name = dataframe_name.loc[
        (dataframe_name[sensor_name] >= lower_whisker) & (dataframe_name[sensor_name] <= upper_whisker)]
    return dataframe_name


# todo optional(remove outliers or not from dataframe)
# group_by_df = remove_outliers(group_by_df)

# data for x label->independent
data_x = group_by_df['day']
# data for y label->dependent by data_x
data_y = group_by_df[sensor_name]

# divide data into train and test, test data= 30%
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.3, shuffle=False)

# plot train values(actual values)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=train_x,
    y=train_y,
    name='Train values',
    mode='markers'))

fig1.update_layout(
    title='Train values for ' + sensor_name,
    yaxis_title=sensor_name,
    xaxis_title='Day(time)',
    showlegend=True)
fig1.show()

# linear regression
model = LinearRegression()
model.fit(train_x.values.reshape(-1, 1), train_y)
# Prediction on validation dataset
test_x = test_x.values.reshape(-1, 1)
pred_linear = model.predict(test_x)


# function used to plot regressions
def plot_regression(first_trace_name, second_trace_name, prediction, title):
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=group_by_df['day'][len(train_x):],
        y=group_by_df[sensor_name][len(train_x):],
        name=first_trace_name,
        mode='markers'))

    fig2.add_trace(go.Scatter(
        x=group_by_df['day'][len(train_x):],
        y=prediction,
        name=second_trace_name
    ))

    fig2.update_layout(
        title=title,
        yaxis_title=sensor_name,
        xaxis_title='Day(time)',
        showlegend=True)
    fig2.show()


# plot linear regression
plot_regression('Actual test values', 'Predicted values', pred_linear,
                'Linear regression for test values for ' + sensor_name)

# calculate linear regression accuracy
print(bcolors.UNDERLINE + "LINEAR REGRESSION ACCURACY:\n" + bcolors.ENDC)
mse1 = mean_squared_error(test_y, pred_linear)
print(bcolors.OKBLUE + "MSE linear regression(mean squared error)", mse1, bcolors.ENDC)

rms_linear = sqrt(mean_squared_error(test_y, pred_linear))
print(bcolors.WARNING + "RMS for linear regression=", rms_linear, bcolors.ENDC)

# reshape dataframe for training+testing for polynomial regression
X = group_by_df[['day']].values
y = group_by_df[[sensor_name]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

fig = go.Figure()


# create list of real values(actual) and forecasted values
# also, calculate the difference between them for every point in dataframe
# return the MSE for each grade used for polynomial forecasting
def analyse_forecast():
    real_data_list = []
    difference_list = []

    predicted_list = pol_reg.predict(poly_reg.fit_transform(X))
    predicted_list = [arr.tolist() for arr in predicted_list]
    print(bcolors.OKBLUE + "MSE polynomial regression(mean squared error)",
          mean_squared_error(group_by_df[sensor_name], predicted_list), bcolors.ENDC)
    print("r2 score ", r2_score(group_by_df[sensor_name], predicted_list))
    print("explained variance score", explained_variance_score(group_by_df[sensor_name], predicted_list))
    rmse = np.sqrt(mean_squared_error(group_by_df[sensor_name], predicted_list))
    print(bcolors.WARNING + "RMS for polynomial regression=", rmse, bcolors.ENDC)
    return mean_squared_error(group_by_df[sensor_name], predicted_list)


# calculate maximum polynomial grade
max_grade = math.floor(math.sqrt(len(group_by_df)))

# create list to store mse for every given polynomial grade(1->sqrt(n))
mse_list = []
group_by_df.reset_index(inplace=True)

print(bcolors.UNDERLINE + "\nPOLYNOMIAL REGRESSION ACCURACY:\n" + bcolors.ENDC)

for count, degree in enumerate([i + 1 for i in range(0, max_grade)]):
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(X_train)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y_train)
    mse_list.append(analyse_forecast())
    # create dataframe with predicted values for given month(30 values)
    group_by_df['predicted'] = pol_reg.predict(poly_reg.fit_transform(X))

    forecast_errors = [abs(group_by_df[sensor_name][i] - group_by_df['predicted'][i]) for i in
                       range(len(group_by_df))]
    print('MAX Forecast Error(degree ', degree, ') is: ', max(forecast_errors))
    print('MIN Forecast Error(degree ', degree, ') is: ', min(forecast_errors))

    # plot predicted values
    fig.add_trace(go.Scatter(
        x=group_by_df['day'],
        y=group_by_df['predicted'],
        name="polynomial grade %d" % degree,
        mode='lines+markers'
    ))

# plot actual values
fig.add_trace(go.Scatter(
    x=group_by_df['day'],
    y=group_by_df[sensor_name],
    name='ACTUAL values',
    mode='lines+markers'))

fig.update_layout(
    title='Polynomial regression for ' + sensor_name,
    yaxis_title=sensor_name,
    xaxis_title='Day',
    showlegend=True)
fig.show()

# create dataframe with mse values and corresponding polynomial grade
mse_df = pd.DataFrame(mse_list)
mse_df.columns = ['mse_values']
mse_df['polynomial_grade'] = [i + 1 for i in range(0, max_grade)]
print(bcolors.OKBLUE + "minimum MSE for given polynomial grades:",
      mse_df[mse_df['mse_values'] == mse_df['mse_values'].min()], bcolors.ENDC)

# calculate and plot spline regression
# calculate 25%,50% and 75% percentiles
percentile_25 = np.percentile(group_by_df['day'], 25)
percentile_50 = np.percentile(group_by_df['day'], 50)
percentile_75 = np.percentile(group_by_df['day'], 75)

# Specifying 3 knots for regression spline
transformed_x1 = dmatrix(
    "bs(group_by_df.day, knots=(percentile_25,percentile_50,percentile_75), degree=20, include_intercept=False)",
    {"group_by_df.day": group_by_df.day}, return_type='dataframe')

# build a regular linear model from the splines
fit_spline = sm.GLM(group_by_df[sensor_name], transformed_x1).fit()

# make predictions
pred_spline = fit_spline.predict(transformed_x1)

# plot regression spline
fig3 = go.Figure()
group_by_df['day'] = group_by_df['day'].map(dt.datetime.fromordinal)
fig3.add_trace(go.Scatter(
    x=group_by_df['day'],
    y=group_by_df[sensor_name],
    name='Actual values',
    mode='lines+markers'))

fig3.add_trace(go.Scatter(
    x=group_by_df['day'],
    y=pred_spline,
    name="Predicted values",
    mode='lines+markers'
))

fig3.update_layout(
    title="Regression Spline for " + sensor_name,
    yaxis_title=sensor_name,
    xaxis_title='Day(time)',
    showlegend=True)
fig3.show()

# calculate regression accuracy
print(bcolors.UNDERLINE + "\nSPLINE REGRESSION ACCURACY:\n" + bcolors.ENDC)
mse1 = mean_squared_error(group_by_df[sensor_name], pred_spline)
print(bcolors.OKBLUE + "MSE spline regression(mean squared error)", mse1, bcolors.ENDC)
rms1 = sqrt(mean_squared_error(group_by_df[sensor_name], pred_spline))
print(bcolors.WARNING + "RMS for regression spline=", rms1, bcolors.ENDC)
print("r2 score for regression spline", r2_score(group_by_df[sensor_name], pred_spline))
print(pred_spline)
