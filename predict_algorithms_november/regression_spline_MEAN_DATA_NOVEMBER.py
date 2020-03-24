import datetime as dt
from math import sqrt

import numpy as np
import pandas as pd
import statsmodels.api as sm
from distributed.deploy.ssh import bcolors
from pandas.plotting import register_matplotlib_converters
from patsy import dmatrix
from plotly import graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

register_matplotlib_converters()

# read csv file
data = pd.read_csv("https://raw.githubusercontent.com/iulianastroia/csv_data/master/final_dataframe.csv")
# data = pd.read_csv("https://raw.githubusercontent.com/iulianastroia/csv_data/master/march_data.csv")

# convert day to pandas datetime format
data['day'] = pd.to_datetime(data['day'], dayfirst=True)

# modify name with any sensor name from df
sensor_name = 'ch2o'

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
        x=group_by_df['day'],
        y=group_by_df[sensor_name],
        name=first_trace_name,
        mode='lines+markers'))

    fig2.add_trace(go.Scatter(
        x=group_by_df['day'][len(train_x):],
        y=prediction,
        name=second_trace_name,
        marker=dict(
            color=np.where(group_by_df['day'].index < len(train_x), 'green', 'red'))))

    fig2.update_layout(
        title=title,
        yaxis_title=sensor_name,
        xaxis_title='Day(time)',
        showlegend=True)
    fig2.show()


# plot linear regression
plot_regression('Actual values', 'Predicted values', pred_linear,
                'Linear regression for test values for ' + sensor_name)

# calculate linear regression accuracy
print(bcolors.UNDERLINE + "LINEAR REGRESSION ACCURACY:\n" + bcolors.ENDC)
mse1 = mean_squared_error(test_y, pred_linear)
print(bcolors.OKBLUE + "MSE linear regression(mean squared error)", mse1, bcolors.ENDC)

rms_linear = sqrt(mean_squared_error(test_y, pred_linear))
print(bcolors.WARNING + "RMSE for linear regression=", rms_linear, bcolors.ENDC)
print("r2 score for LINEAR regression", r2_score(test_y, pred_linear))

# reshape dataframe for training+testing for polynomial regression
X = group_by_df[['day']].values
y = group_by_df[[sensor_name]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

fig = go.Figure()


# create list of real values(actual) and forecasted values
# also, calculate the difference between them for every point in dataframe
# return the MSE for each grade used for regression forecasting
def analyse_forecast(dataframe_name, predicted_list, regression_type):
    # predicted_list = pol_reg.predict(poly_reg.fit_transform(X))
    # predicted_list = [arr.tolist() for arr in predicted_list]
    print("\n Grade: ", degree)
    print(bcolors.OKBLUE + "MSE " + regression_type + " regression(mean squared error)",
          mean_squared_error(dataframe_name[sensor_name], predicted_list), bcolors.ENDC)
    print("r2 score ", r2_score(dataframe_name[sensor_name], predicted_list))
    rmse = np.sqrt(mean_squared_error(dataframe_name[sensor_name], predicted_list))
    print(bcolors.WARNING + "RMSE for " + regression_type + " regression=", rmse, bcolors.ENDC)
    return mean_squared_error(dataframe_name[sensor_name], predicted_list)


# calculate maximum polynomial grade
# max_grade = np.math.floor(np.math.sqrt(len(group_by_df)))
max_grade = 16
# max_grade = 11

# create list to store mse for every given polynomial grade(1->sqrt(n))
mse_list = []
group_by_df.reset_index(inplace=True)

print(bcolors.UNDERLINE + "\nPOLYNOMIAL REGRESSION ACCURACY:\n" + bcolors.ENDC)

for count, degree in enumerate([i + 1 for i in range(0, max_grade)]):
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(X_train)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y_train)
    mse_list.append(analyse_forecast(group_by_df, pol_reg.predict(poly_reg.fit_transform(X)), "polynomial"))

    # create dataframe with predicted values for given month(30 values)
    group_by_df['predicted'] = pol_reg.predict(poly_reg.fit_transform(X))

    forecast_errors = [abs(group_by_df[sensor_name][i] - group_by_df['predicted'][i]) for i in
                       range(len(group_by_df))]
    print('MAX Forecast Error(degree ', degree, ') is: ', max(forecast_errors))
    print('MIN Forecast Error(degree ', degree, ') is: ', min(forecast_errors))

    print(bcolors.OKBLUE + "MSE TRAIN",
          mean_squared_error(train_y, group_by_df['predicted'][:len(train_x)]), bcolors.ENDC)
    print("r2 score TRAIN", r2_score(train_y, group_by_df['predicted'][:len(train_x)]))

    print(bcolors.OKBLUE + "MSE TEST ",
          mean_squared_error(test_y, group_by_df['predicted'][len(train_x):]), bcolors.ENDC)
    print("r2 score TEST", r2_score(test_y, group_by_df['predicted'][len(train_x):]))

    # plot predicted values
    fig.add_trace(go.Scatter(
        x=group_by_df['day'].map(dt.datetime.fromordinal),
        y=group_by_df['predicted'],
        name="polynomial grade %d" % degree,
        mode='lines+markers',
        marker=dict(
            color=np.where(group_by_df['day'].index < len(train_x), 'red', 'green'))))

# plot actual values
fig.add_trace(go.Scatter(
    x=group_by_df['day'].map(dt.datetime.fromordinal),
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
def mse_minumum(regression_type, mse_list_regression, max_grade_regression):
    mse_df = pd.DataFrame(mse_list_regression)
    mse_df.columns = ['mse_values']
    mse_df[regression_type + '_grade'] = [i + 1 for i in range(0, max_grade_regression)]
    print(bcolors.OKBLUE + "minimum MSE for given " + regression_type + " grades:",
          mse_df[mse_df['mse_values'] == mse_df['mse_values'].min()], bcolors.ENDC)


mse_minumum("polynomial", mse_list, max_grade)

# calculate and plot spline regression
# calculate 25%,50% and 75% percentiles
# percentiles for train data
percentile_25_train = np.percentile(group_by_df['day'][:len(X_train)], 25)
percentile_50_train = np.percentile(group_by_df['day'][:len(X_train)], 50)
percentile_75_train = np.percentile(group_by_df['day'][:len(X_train)], 75)

# percentiles for test data
percentile_25_test = np.percentile(group_by_df['day'][len(X_train):], 25)
percentile_50_test = np.percentile(group_by_df['day'][len(X_train):], 50)
percentile_75_test = np.percentile(group_by_df['day'][len(X_train):], 75)

# plot regression spline
print(bcolors.UNDERLINE + "\nSPLINE REGRESSION ACCURACY:\n" + bcolors.ENDC)

fig3 = go.Figure()
mse_list_spline = []
mse_list_train_spline = []
mse_list_test_spline = []


def correlation_line(df, x, y, degree):
    # calculate best fit line
    denominator = (df[x] ** 2).sum() - df[x].mean() * df[x].sum()
    m = ((df[y] * df[x]).sum() - df[y].mean() * df[x].sum()) / denominator
    b = ((df[y].mean() * ((df[x] ** 2).sum())) - df[x].mean() * ((df[y] * df[x]).sum())) / denominator
    # best_fit_line = m * df.pm1 + b
    best_fit_line = m * df[x] + b
    best_fit_fig = go.Figure()

    best_fit_fig.add_trace(go.Scatter(
        x=df[x],
        y=best_fit_line,
        name='Line for Best Fit for grade ' + str(degree),
        mode='lines'))
    best_fit_fig.add_trace(go.Scatter(
        x=df[x],
        y=df[y],
        name=x + ' and ' + y + ' correlation for degree ' + str(degree),
        mode='markers'))
    best_fit_fig.update_layout(
        yaxis_title="forecast",
        xaxis_title='actual')
    best_fit_fig.show()


# regression spline
for count, degree in enumerate([i + 1 for i in range(0, max_grade)]):
    # Specifying 3 knots for regression spline
    transformed_x1 = dmatrix(
        "bs(X_train, knots=(percentile_25_train,percentile_50_train,percentile_75_train), degree=degree,"
        " include_intercept=False)",
        {"X_train": X_train}, return_type='dataframe')
    fit_spline = sm.GLM(y_train, transformed_x1).fit()
    # predict test values
    pred_spline_test = fit_spline.predict(
        dmatrix(
            "bs(X_test, knots=(percentile_25_test,percentile_50_test,percentile_75_test),degree=degree, "
            "include_intercept=False)",
            {"X_test": X_test}, return_type='dataframe'))

    # predict train values
    pred_spline_train = fit_spline.predict(
        dmatrix(
            "bs(X_train, knots=(percentile_25_train,percentile_50_train,percentile_75_train), degree=degree,"
            " include_intercept=False)",
            {"X_train": X_train}, return_type='dataframe'))
    # list of train predicted values
    pred_spline_train = pred_spline_train.tolist()
    # list of test predicted values
    pred_spline_test = pred_spline_test.tolist()
    # holds all predicted values(train and test)
    predicted_val = pred_spline_train + pred_spline_test
    mse_list_spline.append(analyse_forecast(group_by_df, predicted_val, "spline"))
    # list for train MSE
    mse_list_train_spline.append(mean_squared_error(train_y, pred_spline_train))
    # list for test MSE
    mse_list_test_spline.append(mean_squared_error(test_y, pred_spline_test))

    fig3.add_trace(go.Scatter(
        x=group_by_df['day'].map(dt.datetime.fromordinal),
        y=predicted_val,
        name="Predicted values grade " + str(degree),
        mode='lines+markers',
        marker=dict(
            color=np.where(group_by_df['day'].index < len(y_train), 'red', 'green'))))

    # create best line between actual values and predicted ones
    data = pd.DataFrame(columns=['actual', 'forecast'])
    data.actual = group_by_df[sensor_name]
    data.forecast = predicted_val
    correlation_line(data, data.columns[0], data.columns[1], degree)

fig3.add_trace(go.Scatter(
    x=group_by_df['day'].map(dt.datetime.fromordinal),
    y=group_by_df[sensor_name],
    name='Actual values',
    mode='lines+markers'))

fig3.update_layout(
    title="Regression Spline for " + sensor_name,
    yaxis_title=sensor_name,
    xaxis_title='Day(time)',
    showlegend=True)
fig3.show()

mse_minumum("spline", mse_list_spline, max_grade)

# print MSE
fig_mse = go.Figure()
fig_mse.add_trace(go.Scatter(
    x=[i + 1 for i in range(0, max_grade)],
    y=mse_list_spline,
    name='MSE',
    mode='lines+markers'
))

fig_mse.add_trace(go.Scatter(
    x=[i + 1 for i in range(0, max_grade)],
    y=mse_list_train_spline,
    name='MSE train',
    mode='lines+markers'
))

fig_mse.add_trace(go.Scatter(
    x=[i + 1 for i in range(0, max_grade)],
    y=mse_list_test_spline,
    name='MSE test',
    mode='lines+markers'
))
fig_mse.update_layout(
    title='MSE values for different grades',
    xaxis_title='grade',
    yaxis_title='MSE',
)
fig_mse.show()
