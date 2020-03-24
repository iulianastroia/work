import datetime as dt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from distributed.deploy.ssh import bcolors
from pandas.plotting import register_matplotlib_converters
from patsy import dmatrix
from plotly import graph_objs as go
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# fct to create correlation graph and best fit line for given degree

register_matplotlib_converters()

# read csv file
data = pd.read_csv("https://raw.githubusercontent.com/iulianastroia/csv_data/master/march_data.csv")

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

# data for x label->independent
data_x = group_by_df['day']
# data for y label->dependent by data_x
data_y = group_by_df[sensor_name]

# divide data into train and test, test data= 30%
X = group_by_df[['day']].values
y = group_by_df[[sensor_name]].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)


# todo can use only first 10 values as train and the other ones as test(so that 11 March is test data)
# X_train = X[:10]
# y_train = y[:10]
# X_test = X[10:]
# y_test = y[10:]

# create list of real values(actual) and forecasted values
# return the MSE for each grade used for regression forecasting
def analyse_forecast(dataframe_name, predicted_list, regression_type):
    print("\n Grade: ", degree)
    print(bcolors.OKBLUE + "MSE " + regression_type + " regression(mean squared error)",
          mean_squared_error(dataframe_name[sensor_name], predicted_list), bcolors.ENDC)
    print("r2 score ", r2_score(dataframe_name[sensor_name], predicted_list))
    rmse = np.sqrt(mean_squared_error(dataframe_name[sensor_name], predicted_list))
    print(bcolors.WARNING + "RMSE for " + regression_type + " regression=", rmse, bcolors.ENDC)
    return mean_squared_error(dataframe_name[sensor_name], predicted_list)


# decide maximum regression grade
# grade for dec+nov
# max_grade = 16
# grade for march 10
# max_grade = 5 #(for 11 march as test)
max_grade = 10  # (for 30%test data)

group_by_df.reset_index(inplace=True)


# create dataframe with mse values and corresponding regression grade
def mse_minumum(regression_type, mse_list_regression, max_grade_regression):
    mse_df = pd.DataFrame(mse_list_regression)
    mse_df.columns = ['mse_values']
    mse_df[regression_type + '_grade'] = [i + 1 for i in range(0, max_grade_regression)]
    print(bcolors.OKBLUE + "minimum MSE for given " + regression_type + " grades:",
          mse_df[mse_df['mse_values'] == mse_df['mse_values'].min()], bcolors.ENDC)


# percentiles for train data
percentile_25_train = np.percentile(group_by_df['day'][:len(X_train)], 25)
percentile_50_train = np.percentile(group_by_df['day'][:len(X_train)], 50)
percentile_75_train = np.percentile(group_by_df['day'][:len(X_train)], 75)

# percentiles for test data
percentile_25_test = np.percentile(group_by_df['day'][len(X_train):], 25)
percentile_50_test = np.percentile(group_by_df['day'][len(X_train):], 50)
percentile_75_test = np.percentile(group_by_df['day'][len(X_train):], 75)

fig3 = go.Figure()
mse_list_spline = []
mse_list_train_spline = []
mse_list_test_spline = []


def correlation_line(df, x, y, degree):
    scatter_data = go.Scattergl(
        # value for x axis
        x=df[x],
        # value for y axis
        y=df[y],
        # scatter plot
        mode='markers',
        # legend name
        name=x + ' and ' + y + ' correlation for degree ' + str(degree)
    )
    layout = go.Layout(
        # set title of axis
        xaxis=dict(
            title=x
        ),
        yaxis=dict(
            title=y)
    )

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
    # best_fit_fig.show()


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
    pred_spline_train = pred_spline_train.tolist()
    pred_spline_test = pred_spline_test.tolist()
    # holds all predicted values(train and test)
    predicted_val = pred_spline_train + pred_spline_test
    mse_list_spline.append(analyse_forecast(group_by_df, predicted_val, "spline"))

    fig3.add_trace(go.Scatter(
        x=group_by_df['day'].map(dt.datetime.fromordinal),
        y=predicted_val,
        name="Predicted values grade " + str(degree),
        mode='lines+markers',
        marker=dict(
            color=np.where(group_by_df['day'].index < len(y_train), 'red', 'green'))))
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
