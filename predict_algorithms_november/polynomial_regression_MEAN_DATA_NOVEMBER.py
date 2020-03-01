import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime as dt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from pandas.plotting import register_matplotlib_converters
import math
from plotly import graph_objs as go

register_matplotlib_converters()

# this script uses 30 values for november(mean of all 43.000)
# read csv
data = pd.read_csv("https://raw.githubusercontent.com/iulianastroia/csv_data/master/final_dataframe.csv")

# convert day to pandas datetime format
data['day'] = pd.to_datetime(data['day'], dayfirst=True)

# sort values by day
data = data.sort_values(by=['day'])
print("sorted days", data.day)

# create mean of values by days
group_by_df = pd.DataFrame([name, group.mean().pm25] for name, group in data.groupby('day'))
group_by_df.columns = ['day', 'pm25']

group_by_df['day'] = pd.to_datetime(group_by_df['day'])
# convert day column to needed(supported) date format
group_by_df['day'] = group_by_df['day'].map(dt.datetime.toordinal)
print(group_by_df)

# reshape dataframe for training+testing
X = group_by_df['day'].values.reshape(-1, 1)
y = group_by_df['pm25'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# convert to original date format
group_by_df['day'] = group_by_df['day'].map(dt.datetime.fromordinal)


# create list of real values(actual) and forecasted values
# also, calculate the difference between them for every point in dataframe
# return the MSE for each grade used for polynomial forecasting
def analyse_forecast():
    real_data_list = []
    difference_list = []
    for i in range(len(group_by_df)):
        real_data_list.append(group_by_df['pm25'][i])

    predicted_list = pol_reg.predict(poly_reg.fit_transform(X))
    predicted_list = [arr.tolist() for arr in predicted_list]

    for i in range(len(group_by_df)):
        difference_list.append(abs(real_data_list[i] - predicted_list[i]))

    print("max forecasting DIFF", max(difference_list))
    print("min forecasting DIFF", min(difference_list))
    print("MSE(mean squared error)", mean_squared_error(group_by_df['pm25'], predicted_list))
    return mean_squared_error(group_by_df['pm25'], predicted_list)


# calculate maximum polynomial grade
max_grade = math.floor(math.sqrt(len(group_by_df)))
print([i for i in range(0, max_grade)])
# create list to store mse for every given polynomial grade(1->sqrt(n))
mse_list = []
fig = go.Figure()

for count, degree in enumerate([i + 1 for i in range(0, max_grade)]):
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(X_train)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y_train)
    mse_list.append(analyse_forecast())
    # create dataframe with predicted values for given month(30 values)
    group_by_df['predicted'] = pol_reg.predict(poly_reg.fit_transform(X))
    # print polynomial degree and predicted values
    print("degree:", degree, "predicted values", group_by_df['predicted'])
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
    y=group_by_df['pm25'],
    name='ACTUAL values',
    mode='lines+markers'))

fig.update_layout(title='Comparison between predicted values and real ones for November 2019', yaxis_title='Pm2.5',
                  xaxis_title='Day',
                  showlegend=True)
fig.show()

# create dataframe with mse values and corresponding polynomial grade
mse_df = pd.DataFrame(mse_list)
mse_df.columns = ['mse_values']
mse_df['polynomial_grade'] = [i + 1 for i in range(0, max_grade)]
print("Dataframe with MSE values and polynomial grades", mse_df)
print("minimum MSE for given polynomial grades: ", mse_df[mse_df['mse_values'] == mse_df['mse_values'].min()])
