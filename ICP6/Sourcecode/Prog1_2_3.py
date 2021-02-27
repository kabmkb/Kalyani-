
# # Question 1

import pandas as pd
import seaborn as sns

sns.set(style="white", color_codes=True)
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('CC.csv')

# check for null values
nulls = df.isnull().sum()
nulls[nulls > 0]
print(nulls[nulls > 0])

# we now know that credit limit and minimum_payments columns have missing values
# replacing with mean values of respective columns
df['CREDIT_LIMIT'] = df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].mean())
df['MINIMUM_PAYMENTS'] = df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].mean())



# K means model building
x = df.iloc[:, 1:].values

# we do not consider cust_id categorical column to our model training.
from sklearn.cluster import KMeans

Error = []
for nclusters in range(1, 21):
    km = KMeans(n_clusters=nclusters)
    km.fit(x)
    Error.append(km.inertia_)

import matplotlib.pyplot as plt

plt.plot(range(1, 21), Error)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Error')
plt.show()

# best number of clusters is 8
# predict the cluster for each data point
nclusters = 8  # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x)
y_cluster_kmeans = km.predict(x)

from sklearn import metrics

score = metrics.silhouette_score(x, y_cluster_kmeans)
print(score)

## Program 4
# visualize Kmeans
# divide customer data - 8950 customers into 8 clusters
# difficult to make sense out of viz. with 8 clusters, but here is my attempt anyways
plt.scatter(x[:, 0], x[:, 5], c=y_cluster_kmeans, cmap='rainbow')
plt.show()


# # Question 2


# feature scaling
from sklearn import preprocessing
x = df.iloc[:,1:]
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)

# silhouette score
x_scaled = X_scaled.iloc[:,0:].values

nclusters = 8 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x_scaled)
y_cluster_kmeans = km.predict(x_scaled)

from sklearn import metrics
score = metrics.silhouette_score(x_scaled, y_cluster_kmeans)
print(score)


# # silhouette score reported is smaller compared to previous.
# # had it been larger, the reason would have been that since each predictor is on the same scale (between 0 to 1), such distance based algorithms like K-means work well. It avoids bias towards a predictor which has a huge range of values like (1 to 100,000).

# # Question 3

# feature scaling before PCA
from sklearn import preprocessing
x = df.iloc[:,1:]
scaler = preprocessing.StandardScaler()
scaler.fit(x)
X_scaled_array = scaler.transform(x)
X_scaled = pd.DataFrame(X_scaled_array, columns = x.columns)

# PCA
from sklearn.decomposition import PCA
pca = PCA(2) # 2 is number of PCs.
x_pca = pca.fit_transform(X_scaled_array)
df3 = pd.DataFrame(data=x_pca)

# silhouette score
x_scaled_pca = df3.iloc[:,:].values

nclusters = 8 # this is the k in kmeans
km = KMeans(n_clusters=nclusters)
km.fit(x_scaled_pca)
y_cluster_kmeans = km.predict(x_scaled_pca)

from sklearn import metrics
score = metrics.silhouette_score(x_scaled_pca, y_cluster_kmeans)
print(score)


# # Score improved on applying PCA with scaling when compared with other two cases.
# # Dimensionality reduction does help K-means to increase inter cluster distance and reduce the intra cluster distance.


