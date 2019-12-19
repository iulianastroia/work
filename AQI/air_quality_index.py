import pandas as pd
from datetime import datetime
import plotly.figure_factory as ff
from plotly.offline import plot

data = pd.read_csv('https://raw.githubusercontent.com/iulianastroia/csv_data/master/final_dataframe.csv')


def calculate_aqi(pollutant_name, pollutant_concentration):
    # convert from concentration to AQI:
    # source: https://en.wikipedia.org/wiki/Air_quality_index
    c = pollutant_concentration
    try:
        if pollutant_name == 'pm25':
            #         24 h average
            if 0 <= pollutant_concentration <= 12:
                c_low = 0
                c_high = 12
                i_low = 0
                i_high = 50
            if 12.1 <= pollutant_concentration <= 35.4:
                c_low = 12.1
                c_high = 35.4
                i_low = 51
                i_high = 100
            if 35.5 <= pollutant_concentration <= 55.4:
                c_low = 35.5
                c_high = 55.4
                i_low = 101
                i_high = 150
            if 55.5 <= pollutant_concentration <= 150.4:
                c_low = 55.5
                c_high = 150.4
                i_low = 151
                i_high = 200
            if 150.5 <= pollutant_concentration <= 250.4:
                c_low = 150.5
                c_high = 250.4
                i_low = 201
                i_high = 300
            if 250.5 <= pollutant_concentration <= 350.4:
                c_low = 250.5
                c_high = 350.4
                i_low = 301
                i_high = 400
            if 350.5 <= pollutant_concentration <= 500.4:
                c_low = 350.5
                c_high = 500.4
                i_low = 401
                i_high = 500

        if pollutant_name == 'pm10':
            #         24 h average
            if 0 <= pollutant_concentration <= 54:
                c_low = 0
                c_high = 54
                i_low = 0
                i_high = 50
            if 55 <= pollutant_concentration <= 154:
                c_low = 55
                c_high = 154
                i_low = 51
                i_high = 100
            if 155 <= pollutant_concentration <= 254:
                c_low = 155
                c_high = 254
                i_low = 101
                i_high = 150
            if 255 <= pollutant_concentration <= 354:
                c_low = 255
                c_high = 354
                i_low = 151
                i_high = 200
            if 355 <= pollutant_concentration <= 424:
                c_low = 355
                c_high = 424
                i_low = 201
                i_high = 300
            if 425 <= pollutant_concentration <= 504:
                c_low = 425
                c_high = 504
                i_low = 301
                i_high = 400
            if 505 <= pollutant_concentration <= 604:
                c_low = 505
                c_high = 604
                i_low = 401
                i_high = 500
        # calculate AQI
        i = (i_high - i_low) / (c_high - c_low) * (c - c_low) + i_low
        return round(i)

    except:
        print("Exceeded Range")


data['day'] = pd.to_datetime(data['day'], dayfirst=True)  # convert to date format
data = data.sort_values(by=['day'])  # sort dates by day
print("sorted days", data.day)

grp_date = data.groupby('day')
average_df = pd.DataFrame(grp_date.mean())  # calculate mean value  for every given day

# calculate AQI for average of pm2.5 and pm10(one value per day)
aqi_df = pd.DataFrame(columns=['AQI'])
aqi_pm10_df = pd.DataFrame(columns=['AQI_pm10'])
for i in range(len(average_df)):
    aqi_df = aqi_df.append({'AQI': calculate_aqi("pm25", average_df.pm25[i])}, ignore_index=True)
    aqi_pm10_df = aqi_pm10_df.append({'AQI_pm10': calculate_aqi("pm10", average_df.pm10[i])}, ignore_index=True)

print("dataframe AQI pm25", aqi_df)

data = data.drop_duplicates(subset='day', keep='first')

data = data.reset_index(drop=True)
data['day'] = [datetime.date(d) for d in data['day']]

print('eliminate duplicates of dates', data.day)


# source: https://plot.ly/~empet/15229/heatmap-with-a-discrete-colorscale/#/
def discrete_colorscale(interval_values, color_codes):
    """
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """
    if len(interval_values) != len(color_codes) + 1:
        raise ValueError('len(boundary values) should be equal to  len(colors)+1')
    interval_values = sorted(interval_values)
    nvals = [(v - interval_values[0]) / (interval_values[-1] - interval_values[0]) for v in
             interval_values]  # normalized values

    color_scale = []  # discrete colorscale
    for k in range(len(color_codes)):
        color_scale.extend([[nvals[k], color_codes[k]], [nvals[k + 1], color_codes[k]]])
    return color_scale


# interval values for AQI
interval_values = [2, 50, 100, 150, 200, 300, 500]

# color codes for AQI
color_codes = ['#0e7a04', '#ffbf00', '#df8719', '#df8719', '#641b6d', '#810808']
color_scale = discrete_colorscale(interval_values, color_codes)


def create_heatmap(title_name, z_values, z_text):
    font_colors = ['black']
    fig = ff.create_annotated_heatmap(z=z_values,
                                      annotation_text=z_text, colorscale=color_scale, zmin=0, zmax=500,
                                      font_colors=font_colors)

    # show colorbar
    fig['data'][0]['showscale'] = True

    fig.update_layout(
        autosize=True,
        title=title_name
    )
    plot(fig)
    fig.show()

    fig.write_image(title_name + ".png", width=8000, height=4500)


z_text = [[data.day[2 * i] for i in range(int(len(data) / 2))],
          [data.day[2 * i + 1] for i in range(int(len(data) / 2))]]

z_values = [[aqi_df['AQI'][2 * i] for i in range(int(len(data) / 2))],
            [aqi_df['AQI'][2 * i + 1] for i in range(int(len(data) / 2))]
            ]
create_heatmap("AQI for pm 2.5 for November", z_values, z_text)

z_values = [[aqi_pm10_df['AQI_pm10'][2 * i] for i in range(int(len(data) / 2))],
            [aqi_pm10_df['AQI_pm10'][2 * i + 1] for i in range(int(len(data) / 2))]
            ]

create_heatmap("AQI for pm 10 for November", z_values, z_text)
