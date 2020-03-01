import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
from sklearn.metrics import mean_squared_error
from plotly import graph_objs as go
import math


# predict data for November using all values
def parser(x):
    return datetime.strptime(x, '%d/%m/%Y')


data = pd.read_csv("https://raw.githubusercontent.com/iulianastroia/csv_data/master/final_dataframe.csv")
print(dict(data.dtypes))

data['day'] = pd.to_datetime(data['day'], dayfirst=True)
data = data.sort_values(by=['day'])

X = data['day'].values.reshape(-1, 1)
y = data['pm25'].values.reshape(-1, 1)

# Splitting the dataset into training(70%) and test(30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    # shuffle=False)
                                                    random_state=0)

# Fitting Polynomial Regression to the dataset
poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X_train)
print("x poly", X_poly)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y_train)

print("LEN of test data", len(X_test))
print("LEN of actual data", len(data))


def analyse_forecast():
    real_data_list = []
    difference_list = []
    for i in range(len(data)):
        real_data_list.append(data['pm25'][i])

    predicted_list = pol_reg.predict(poly_reg.fit_transform(X))
    predicted_list = [arr.tolist() for arr in predicted_list]

    for i in range(len(data)):
        difference_list.append(abs(real_data_list[i] - predicted_list[i]))

    print("max forecasting DIFF", max(difference_list))
    print("min forecasting DIFF", min(difference_list))
    print("MSE(mean squared error)", mean_squared_error(data['pm25'], predicted_list))
    return mean_squared_error(data['pm25'], predicted_list)


# calculate maximum polynomial grade
max_grade = math.floor(math.sqrt(len(data)))

# create list to store mse for every given polynomial grade
mse_list = []
fig = go.Figure()

# plot polynomial regression from 1->15 grade
for count, degree in enumerate([i + 1 for i in range(0, 15)]):
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(X_train)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y_train)
    mse_list.append(analyse_forecast())
    # create dataframe with predicted values for given month(30 values)
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
mse_df['polynomial_grade'] = [i + 1 for i in range(0, 15)]
print("Dataframe with MSE values and polynomial grades", mse_df)
print("minimum MSE for given polynomial grades: ", mse_df[mse_df['mse_values'] == mse_df['mse_values'].min()])
