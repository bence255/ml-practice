# -*- coding: utf-8 -*-
"""
Created on Mon May 4 09:23:04 2020
Corrected on Wen Nov 20 12:47:12 2024

Task: Advanced clustering (hierarchical and DBSCAN) of Aggregation dataset from the URL
https://arato.inf.unideb.hu/ispany.marton/DataMining/Practice/Clustering/

Python tools
Libraries: numpy, matplotlib, urllib, sklearn, scipy
Modules: pyplot, request, cluster, metrics
Classes: AgglomerativeClustering, DBSCAN
Functions: plot_dendrogram, urlopen, davies_bouldin_score, contingency_matrix

@author: Márton Ispány, Faculty of Informatics, University of Debrecen
License: BSD 3 clause
"""

import numpy as np  # Numerical Python library
import matplotlib.pyplot as plt  # Matlab-like Python module
from urllib.request import urlopen  # importing url handling
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
)  # class of clustering algorithms
from sklearn.metrics import davies_bouldin_score  # Davies-Bouldin goodness-of-fit
from sklearn.metrics.cluster import (
    contingency_matrix,
)  # function for contingency matrix
from scipy.cluster.hierarchy import dendrogram  # clustering visualization


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


url = "https://arato.inf.unideb.hu/ispany.marton/DataMining/Practice/Clustering/Aggregation.tsv"
raw_data = urlopen(url)  # opening url
data = np.loadtxt(raw_data, delimiter="\t")  # loading dataset
X = data[:, 0:2]  #  splitting the input attributes
y = data[:, 2]  #  label attribute

# Visualizing datapoints using label colors
fig = plt.figure(1)
plt.title("Scatterplot of datapoints with labels")
plt.xlabel("X1")
plt.ylabel("X2")
plt.scatter(X[:, 0], X[:, 1], s=50, c=y)
plt.show()

# Visualizing datapoints without label colors
fig = plt.figure(11)
plt.title("Scatterplot of datapoints without labels")
plt.xlabel("X1")
plt.ylabel("X2")
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()


# Agglomerative clustering with single linkage method
link = "single"  #  linkage method
# Building the full tree
single_cluster = AgglomerativeClustering(
    distance_threshold=0, n_clusters=None, linkage=link
)  # instance of AgglomerativeClustering
single_cluster.fit(X)

# Plot the top p levels of the dendrogram
fig = plt.figure(2)
plt.title("Hierarchical Clustering Dendrogram (single linkage)")
plot_dendrogram(single_cluster, truncate_mode="level", p=4)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# Default cluster number
K = 7

# Enter parameters from consol
user_input = input("Number of clusters [default:7]: ")
if len(user_input) != 0:
    K = np.int8(user_input)


# Generating clusters
single_cluster = AgglomerativeClustering(
    n_clusters=K, linkage=link
)  # instance of AgglomerativeClustering
single_cluster.fit(X)
pred_single = single_cluster.labels_
db_single = davies_bouldin_score(X, pred_single)  # goodness-of-fit
cm_single = contingency_matrix(
    y, pred_single
)  # comparing true label and cluster membership


# Visualizing datapoints using cluster label
fig = plt.figure(3)
plt.title("Scatterplot of datapoints with single linkage clustering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.scatter(X[:, 0], X[:, 1], s=50, c=pred_single)
plt.show()

# Agglomerative clustering with complete linkage method
link = "complete"  #  linkage method
# Building the full tree
complete_cluster = AgglomerativeClustering(
    distance_threshold=0, n_clusters=None, linkage=link
)  # instance of AgglomerativeClustering
complete_cluster.fit(X)

# Plot the top p levels of the dendrogram
fig = plt.figure(4)
plt.title("Hierarchical Clustering Dendrogram (complete linkage)")
plot_dendrogram(complete_cluster, truncate_mode="level", p=4)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# Default parameters
K = 7

# Enter parameters from consol
user_input = input("Number of clusters [default:7]: ")
if len(user_input) != 0:
    K = np.int8(user_input)

# Generating clusters
complete_cluster = AgglomerativeClustering(
    n_clusters=K, linkage=link
)  # instance of AgglomerativeClustering
complete_cluster.fit(X)
pred_complete = complete_cluster.labels_
db_complete = davies_bouldin_score(X, pred_complete)  # goodness-of-fit
cm_complete = contingency_matrix(
    y, pred_complete
)  # comparing true label and cluster membership

# Visualizing datapoints using cluster label
fig = plt.figure(5)
plt.title("Scatterplot of datapoints with complete linkage clustering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.scatter(X[:, 0], X[:, 1], s=50, c=pred_complete)
plt.show()

# Agglomerative clustering with Ward method
link = "ward"  #  linkage method
# Building the full tree
ward_cluster = AgglomerativeClustering(
    distance_threshold=0, n_clusters=None, linkage=link
)  # instance of AgglomerativeClustering
ward_cluster.fit(X)

# Plot the top p levels of the dendrogram
fig = plt.figure(6)
plt.title("Hierarchical Clustering Dendrogram (Ward)")
plot_dendrogram(ward_cluster, truncate_mode="level", p=4)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# Default parameters
K = 7

# Enter parameters from consol
user_input = input("Number of clusters [default:7]: ")
if len(user_input) != 0:
    K = np.int8(user_input)
del user_input

# Generating clusters
ward_cluster = AgglomerativeClustering(
    n_clusters=K, linkage=link
)  # instance of AgglomerativeClustering
ward_cluster.fit(X)
pred_ward = ward_cluster.labels_
db_ward = davies_bouldin_score(X, pred_ward)  # goodness-of-fit
cm_ward = contingency_matrix(
    y, pred_ward
)  # comparing true label and cluster membership

# Visualizing datapoints using cluster label
fig = plt.figure(7)
plt.title("Scatterplot of datapoints with Ward linkage clustering")
plt.xlabel("X1")
plt.ylabel("X2")
plt.scatter(X[:, 0], X[:, 1], s=50, c=pred_ward)
plt.show()

# DBSCAN clustering
radius = 1.06  # radius of neighborhood
inner_points = 5  #  inner point definition
dbscan_cluster = DBSCAN(eps=radius, min_samples=inner_points)  # instance of DBSCAN
dbscan_cluster.fit(X)
pred_dbscan = dbscan_cluster.labels_
db_dbscan = davies_bouldin_score(X, pred_dbscan)  # goodness-of-fit
cm_dbscan = contingency_matrix(
    y, pred_dbscan
)  # comparing true label and cluster membership

# Visualizing datapoints using cluster label
fig = plt.figure(8)
plt.title("Scatterplot of datapoints with DBSCAN")
plt.xlabel("X1")
plt.ylabel("X2")
plt.scatter(X[:, 0], X[:, 1], s=50, c=pred_dbscan)
plt.show()

# The parameters (eps and min_samples) are determined by guessing
# The best result is DBSCAN which can recover the 7 original cluster
# more or less precisely
# By contingency matrix (cm) the clustering results can also be compared
# where rows are the true labels, columns are the cluster labels

# End of code
