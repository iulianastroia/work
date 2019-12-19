import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.cluster import KMeans
from k_means import point_class

df = pd.read_csv('https://raw.githubusercontent.com/iulianastroia/csv_data/master/final_dataframe.csv')

# get only data from 16 november
# df = df.loc[
#     (df['readable time'] > '16/11/2019 00:00:00') & (df['readable time'] < '16/11/2019 23:59:59')]

X = df.iloc[:, [0, 8]].values  # time and ch2o

wcss = []
# 10 wcss
for i in range(1, 11):
    # fit kmeans to data x
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # computes within clusters sum of squares

distances = []
# distance from every point on elbow(max distance represents number of clusters)
for i in range(0, 10):
    p1 = point_class.Point(initx=1, inity=wcss[0])
    p2 = point_class.Point(initx=10, inity=wcss[9])
    p = point_class.Point(initx=i + 1, inity=wcss[i])
    distances.append(p.distance_to_line(p1, p2))
print("max dist=", max(distances))
index_wcss = distances.index(max(distances))
print("index of max dist", distances.index(max(distances)))
print("wcss[", index_wcss, "]=", wcss[index_wcss])  # find equivalent cluster, y=2675967629744614.5->x=3 clusters
cluster_number = index_wcss + 1
print("Number of clusters is: ", cluster_number)

elbow_figure = go.Figure()

elbow_figure.add_trace(go.Scatter(x=[i for i in range(1, 11)], y=wcss, mode='lines',
                                  marker=dict(color='red'), name='Elbow method'))
elbow_figure.update_layout(
    title="The Elbow Method",
    xaxis_title="Number of Clusters",
    yaxis_title="WCSS")

plot(elbow_figure)

# applying k-means to the dataset
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)  # tell us which cluster the data belongs to

cluster_figure = go.Figure()

# visualizing the cluster(scatter plot)
cluster_figure.add_trace(go.Scatter(x=X[y_kmeans == 0, 0], y=X[y_kmeans == 0, 1], mode='markers',
                                    marker=dict(color='red'), name='Cluster 1'))

cluster_figure.add_trace(go.Scatter(x=X[y_kmeans == 1, 0], y=X[y_kmeans == 1, 1], mode='markers',
                                    marker=dict(color='blue'), name='Cluster 2'))

cluster_figure.add_trace(go.Scatter(x=X[y_kmeans == 2, 0], y=X[y_kmeans == 2, 1], mode='markers',
                                    marker=dict(color='green'), name='Cluster 3'))
cluster_figure.add_trace(go.Scatter(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1], mode='markers',
                                    marker=dict(color='yellow'), name='Centroids'))

cluster_figure.update_layout(
    title="Cluster of ch2o",
    xaxis_title="Date",
    yaxis_title="Ch2o"
)
plot(cluster_figure)
