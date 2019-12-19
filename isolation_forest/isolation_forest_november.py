import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
import time

pd.options.mode.chained_assignment = None  # NO WARNING

csv_path = 'https://raw.githubusercontent.com/iulianastroia/csv_data/master/final_dataframe.csv'
full_df = pd.read_csv(csv_path)

fig = make_subplots(
    rows=2, cols=1, subplot_titles=("co2 data for november", "sample data")
)

# columns[8]=ch2o values
fig.add_trace(go.Scatter(x=full_df["readable time"], y=full_df[full_df.columns[8]], name=full_df.columns[8]), row=1,
              col=1)
fig.update_xaxes(title_text="full november", row=1, col=1)
fig.update_xaxes(title_text="16 november", row=2, col=1)
fig.update_yaxes(title_text="ch2o", row=1, col=1)
fig.update_yaxes(title_text="ch2o", row=2, col=1)

# visualize 16 november(spike)
df = full_df.loc[
    (full_df['readable time'] > '16/11/2019 00:00:00') & (full_df['readable time'] < '16/11/2019 23:59:59')]
df['readable time'] = df['readable time']
df['ch2o'] = df['ch2o']
fig.add_trace(go.Scatter(x=df['readable time'], y=df['ch2o'], name="values 16 november data"), row=2, col=1)

fig.update_layout(title="November ch2o analysis", height=1800,
                  xaxis=dict(
                      tickmode='linear',
                      tick0=0,  # starting position
                      dtick=1000  # tick step
                  )
                  )
plot(fig)

# MODEL COLUMN VALUE
# 0.04->4% contamination parameter
clf = IsolationForest(n_estimators=10, max_samples='auto', contamination=float(.04),
                      max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0, behaviour='new')

clf.fit(df[['ch2o']])
# average anomaly score
df['scores'] = clf.decision_function(df[['ch2o']])

# predict if a given value is an outlier or not; 1=inliner; -1=outlier
df['anomaly'] = clf.predict(df[['ch2o']])
print("added anomaly column:", df)
df.loc[df['anomaly'] == 1, 'anomaly'] = 0  # if anomaly column value is 1, replace with 0
df.loc[df['anomaly'] == -1, 'anomaly'] = 1  # if anomaly column value is -1, replace with 1
print(df['anomaly'].value_counts())  # how many values of 0(not an anomaly) are in df and how many values of 1(anomaly)


# *********************************
# PLOT
def plot_anomaly(df, metric_name):
    df['readable time'] = pd.to_datetime(df['readable time'])
    dates = df['readable time']
    bool_array = (abs(df['anomaly']) > 0)  # True->anomaly is present
    actuals = df["ch2o"][-len(bool_array):]
    print("actual values ", actuals)
    anomaly_points = bool_array * actuals
    anomaly_points[anomaly_points == 0] = np.nan  # ignore 0 values of df anomaly points
    # table color values
    color_map = {0: "green", 1: "red"}

    table = go.Table(
        domain=dict(x=[0, 1],
                    y=[0, 0.3]),
        columnwidth=[1, 2],
        header=dict(values=['time', 'values']),
        cells=dict(values=[df.round(3)[k].tolist() for k in ['readable time', 'ch2o']],
                   fill=dict(color=[df['anomaly'].map(color_map)]))
    )

    # Plot the actuals points
    actuals = go.Scatter(name='Actual Values',
                         x=dates,
                         y=df['ch2o'],
                         xaxis='x1', yaxis='y1',
                         marker=dict(size=12,
                                     line=dict(width=1),
                                     color="blue"))

    # Highlight the anomaly points
    anomalies_map = go.Scatter(name="Anomaly",
                               showlegend=True,
                               x=dates,
                               y=anomaly_points,
                               mode='markers',
                               xaxis='x1',
                               yaxis='y1',
                               marker=dict(color="red",
                                           size=11,
                                           line=dict(
                                               color="red",
                                               width=2)))
    axis = dict(
        showline=True,
        zeroline=False,
        showgrid=True,
        mirror=True,
        ticklen=4,
        gridcolor='#ffffff',
        tickfont=dict(size=10))
    layout = dict(
        width=1000,
        height=865,
        autosize=False,
        title=metric_name,
        margin=dict(t=75),
        showlegend=True,
        xaxis1=dict(axis, **dict(domain=[0, 1], anchor='y1', showticklabels=True)),
        yaxis1=dict(axis, **dict(domain=[2 * 0.21 + 0.20, 1], anchor='x1', hoverformat='.2f')))

    fig = go.Figure(data=[table, anomalies_map, actuals], layout=layout)

    # print OUTLIERS
    fig_outliers = go.Figure(
        data=go.Scatter(x=dates, y=df['ch2o'], mode='markers',
                        marker=dict(color=np.where(df['anomaly'] == 1, 'red', 'blue'))))

    plot(fig)
    time.sleep(4)
    plot(fig_outliers)


plt.show()
plot_anomaly(df, 'visualize anomalies in 16 november data')
