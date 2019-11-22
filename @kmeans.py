# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values
y = dataset.iloc[:, 3].values
# using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
      kmeans = KMeans(n_clusters = i, init =  'k-means++', max_iter = 300, n_init = 20, random_state = 0)
      kmeans.fit(X)
      wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("The Elbow Method")
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.show()

#applying k-means to the mall dataset
from sklearn.cluster import KMeans
kmeans_opt = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
y_means = kmeans_opt.fit_predict(X)

#visualing the clusters
plt.scatter(X[y_means==0, 0], X[y_means==0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(X[y_means==1, 0], X[y_means==1, 1], s = 100, c = 'yellow', label = 'Standard')
plt.scatter(X[y_means==2, 0], X[y_means==2, 1], s = 100, c = 'blue', label = 'Main Target')
plt.scatter(X[y_means==3, 0], X[y_means==3, 1], s = 100, c = 'black', label = 'Careless')
plt.scatter(X[y_means==4, 0], X[y_means==4, 1], s = 100, c = 'pink', label = 'less important')
plt.scatter(kmeans_opt.cluster_centers_[:,0], kmeans_opt.cluster_centers_[:,1], s = 300, c = 'green', label = "Centroids")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""