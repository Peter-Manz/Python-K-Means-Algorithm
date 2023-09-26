import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

#*** Read in csv file, and set some more human readable properties for 'pd' ***#
df = pd.read_csv('Wholesale customers data.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 250)

#*** Remove Null and Certain Columns from Dataframe, return DataFrame  ***#
def dropNullAndDuplicates(dataFrame):
    dataFrame = dataFrame.dropna()
    dataFrame = dataFrame.drop(columns=['Region'])
    dataFrame = dataFrame.drop_duplicates(subset=['Channel','Fresh', 'Milk','Grocery','Frozen','Detergents_Paper','Delicassen'], keep='first')
    return dataFrame

#*** Print dataframe info, len, describe, nulls, etc.   ***#
def describeDataframe(dataFrame):
    print(len(dataFrame))
    print(dataFrame.head(10))
    print(dataFrame.tail(10))
    print(dataFrame.info())
    print(dataFrame.describe())
    print ("Missing : \n", dataFrame.isnull().sum())
    print(dataFrame.Channel.value_counts())

#*** Keep only certain useable Columns ***#
def dropAndModifyColumns(dataFrame):
    dataFrame = dataFrame.drop(columns=['Region'])
    return dataFrame

#*** Display heatmap correlation using seaborn   ***#
def dataCorrelationHeatmap(dataFrame,title):
    corr_matrix = dataFrame.corr()
    plt.figure(figsize=(8,8))
    ax = plt.axes()
    sns.heatmap(data=corr_matrix, cmap='BrBG',annot=True,linewidths=0.2, ax = ax)
    ax.set_title(title)
    plt.show()

#*** Filter through full dataframe and separate into smaller dataframe based on Channel  ***#
def filterApps(dataFrame, intValue):
    filterDataFrame = dataFrame[(dataFrame['Channel'] == intValue)]
    return filterDataFrame

#*** Visualize using a bar graph the Different Channels and the Grocery Quantities  ***#
def dataVisualization(dataFrame,title):
    fig = dataFrame.Channel.value_counts().sort_index().plot(kind='bar',title=title)
    fig.set_xlabel('Channel')
    fig.set_ylabel('Quantity')
    plt.show()

#*** Display the Rating Scores to discover the distribution   ***#
def dataProportions(dataFrame,title):
    plt.figure(1, figsize = (15,6))
    n = 0 
    for x in ['Milk','Grocery','Delicassen','Detergents_Paper']:
        n += 1
        plt.subplot(1,4,n)
        plt.subplots_adjust(hspace=0.5,wspace = 0.5)
        sns.distplot(dataFrame[x], bins = 15)
    plt.title('Distplot of {}'.format(title))
    plt.show()

def dataCorrelations(dataFrame, title):
    sns.pairplot(dataFrame, vars = ['Milk','Grocery','Detergents_Paper','Delicassen'], hue = "Channel")
    plt.show()

#*** Determine the appropriate amount of clusters using the Kmeans Silhouette , outputing cluster size followed by silhouette score ***#
def elbow_kmeans_silhouette(X, max_clusters):
    inertia = []
    min_clusters = 2
    for n in range(min_clusters , max_clusters):
        kmeans = KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, random_state= 111)
        kmeans.fit(X)
        y_kmeans = kmeans.fit_predict(X)
        if n > 1:
            print(n, 'cluster silhouette score = ', silhouette_score(X, y_kmeans))
            inertia.append(kmeans.inertia_)

#*** Determine the appropriate amount of clusters using the Kmeans elbow method displaying results in graph  ***#
def elbow_kmeans(X, max_clusters):
    inertia = []
    min_clusters = 2
    for n in range(min_clusters, max_clusters):
        kmeans = KMeans(n_clusters=n, init='k-means++',n_init = 10, max_iter=300, random_state=111)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    plt.figure(1 , figsize = (15 ,6))
    plt.plot(np.arange(min_clusters , max_clusters) , inertia , 'o')
    plt.plot(np.arange(min_clusters , max_clusters) , inertia , '-' , alpha = 0.5)
    plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
    plt.show()

#***  kmeans function perform the clustering and printout the results ***#
def kmeans_algorithm(X, clusters):
    kmeans = KMeans(n_clusters = clusters, init='k-means++',n_init = 10, max_iter=300, random_state=111)
    kmeans.fit(X)
    y_kmeans = kmeans.fit_predict(X)
    print('Silhoutte score = ', silhouette_score(X, y_kmeans))
    df['cluster'] = pd.DataFrame(y_kmeans)
    return kmeans.labels_, kmeans.cluster_centers_

#***  Display the  Clusters and Centroid on Graph ***#
def twoDkmeans(X, clusters, attributes, dataFrame):
    labels, centroids = kmeans_algorithm(X, clusters)
    plt.figure(1, figsize = (15,7))
    plt.clf()
    plt.scatter(x=attributes[0], y=attributes[1], data = dataFrame, c=labels, s=100)
    plt.scatter(x = centroids[: , 0], y=centroids[: , 1], s = 300, c = 'red', alpha = 0.5)
    plt.ylabel(attributes[1]) , plt.xlabel(attributes[0])
    plt.show() 


df1 = dropNullAndDuplicates(df)
#describeDataframe(df1)
#dataCorrelationHeatmap(df1, "Full DataSet")
#dataVisualization(df1, "Hotel/Restaurant/Cafe vs Retail")
#dataProportions(df1, "Full DataSet")
#dataCorrelations(df1, "Full DataSet")

#horecaDF = filterApps(df1,1)
#retailDF = filterApps(df1,2)
#dataCorrelationHeatmap(horecaDF, "Hotel/Restaurant/Cafe")
#dataCorrelationHeatmap(retailDF, "Retail")

Att1 = ['Grocery','Milk']
Att2 = ['Grocery', 'Detergents_Paper']
X1 = df1[Att1].values
X2 = df1[Att2].values

#elbow_kmeans(X1, 15)
#elbow_kmeans_silhouette(X1,15)
#twoDkmeans(X1,3,Att1,df1)

#elbow_kmeans(X2, 15)
#elbow_kmeans_silhouette(X2,15)
#twoDkmeans(X2,3,Att2,df1)

