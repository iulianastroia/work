import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
from sklearn.metrics import mean_squared_error
from plotly import graph_objs as go
import math
import numpy as np


# predict data for November using all values
def parser(x):
    return datetime.strptime(x, '%d/%m/%Y')


data = pd.read_csv('https://raw.githubusercontent.com/iulianastroia/csv_data/master/final_dataframe.csv',
                   date_parser=parser, parse_dates=['day'])

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
                                                    # shuffle=False)
                                                    random_state=0)


def analyse_forecast():
    real_data_list = []
    difference_list = []
    for i in range(len(data)):
        real_data_list.append(data[sensor_name][i])

    predicted_list = pol_reg.predict(poly_reg.fit_transform(X))
    predicted_list = [arr.tolist() for arr in predicted_list]

    for i in range(len(data)):
        difference_list.append(abs(real_data_list[i] - predicted_list[i]))

    print("max forecasting DIFF", max(difference_list))
    print("min forecasting DIFF", min(difference_list))
    print("MSE(mean squared error)", mean_squared_error(data[sensor_name], predicted_list))
    return mean_squared_error(data[sensor_name], predicted_list)


# calculate maximum polynomial grade
# square root
max_grade = math.floor(math.sqrt(len(data)))
print("max_grade", max_grade)
# max grade=207
# algorithm works until polynomial grade=205

# todo can also use cubic root
# max_grade = math.floor(math.pow(len(data), 1 / 3))
# print("MAX GRADE", max_grade)

# todo can also use sqrt grade 4
# max_grade = math.floor(math.pow(len(data), 1 / 4))
# print("MAX GRADE", max_grade)

# create list to store mse for every given polynomial grade
mse_list = []
fig = go.Figure()

# plot polynomial regression from 1->205(maximum value supported) grade
for count, degree in enumerate([i + 1 for i in range(0, max_grade - 2)]):
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(X_train)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y_train)
    mse_list.append(analyse_forecast())
    # create dataframe with predicted values for given month
    data['predicted'] = pol_reg.predict(poly_reg.fit_transform(X))
    # print polynomial degree and predicted values
    print("degree:", degree, "predicted values", data['predicted'])

    # plot predicted values
    fig.add_trace(go.Scatter(
        x=data['readable time'],
        y=data['predicted'],
        name="polynomial grade %d" % degree))

# plot actual values
fig.add_trace(go.Scatter(
    x=data['readable time'],
    y=data['pm25'],
    name='ACTUAL values'))

fig.update_layout(title='Comparison between predicted values and real ones for November 2019', yaxis_title='Pm2.5',
                  xaxis_title='Day',
                  showlegend=True)
fig.show()

# create dataframe with mse values and corresponding polynomial grade
mse_df = pd.DataFrame(mse_list)
mse_df.columns = ['mse_values']
mse_df['polynomial_grade'] = [i + 1 for i in range(0, max_grade - 2)]
print("Dataframe with MSE values and polynomial grades", mse_df)
print("minimum MSE for given polynomial grades: ", mse_df[mse_df['mse_values'] == mse_df['mse_values'].min()])
