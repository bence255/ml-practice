# -*- coding: utf-8 -*-
"""
Created on Sun March 29 23:02:03 2020
Corrected on Fri Nov 15 17:09:55 2024

Task: Decision tree analysis of Spambase data reading from URL
Original data source: https://archive.ics.uci.edu/ml/datasets/spambase

Python tools
Libraries: numpy, matplotlib, urllib, sklearn
Modules: pyplot, request, model_selection, tree
Classes: DecisionTreeClassifier
Functions: loadtxt, urlopen, train_test_split, plot_tree

@author: Márton Ispány, Faculty of Informatics, University of Debrecen
License: BSD 3 clause
"""

import numpy as np  # Numerical Python library
import matplotlib.pyplot as plt  # Matlab-like Python module
from urllib.request import urlopen  # url handling
from sklearn.model_selection import train_test_split  # partitioning function
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
)  # importing decision tree tools

# Reading the dataset
url = "https://arato.inf.unideb.hu/ispany.marton/DataMining/Practice/Datasets/spamdata.csv"
raw_data = urlopen(url)
data = np.loadtxt(
    raw_data, skiprows=1, delimiter=";"
)  # reading numerical data from csv file
del raw_data

# Reading attribute names
url_names = "https://arato.inf.unideb.hu/ispany.marton/DataMining/Practice/Datasets/spambase.names.txt"
raw_names = urlopen(url_names)
attribute_names = []  #  list for names
for line in raw_names:
    name = line.decode("utf-8")  # transforming bytes to string
    name = name[0 : name.index(":")]  # extracting attribute name from string
    attribute_names.append(name)  # append the name to a list
del raw_names

# Defining input and target variables
X = data[:, 0:57]
y = data[:, 57]
del data
input_names = attribute_names[0:57]
target_names = ["ham", "spam"]


# Partitioning into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=2024
)

# Initialize the decision tree object
impurity = "gini"  # homogeneity criteria for leaves
depth = 4  #  depth of the tree
# Instance of decision tree class
tree_clf_gini = DecisionTreeClassifier(criterion=impurity, max_depth=depth)

# Fitting decision tree on training dataset and computing the score on training and test
tree_clf_gini.fit(X_train, y_train)  # fitting decision tree to training data
score_train = tree_clf_gini.score(
    X_train, y_train
)  # Goodness of tree on training dataset
score_test = tree_clf_gini.score(X_test, y_test)  # Goodness of tree on test dataset

# Predicting spam for test data
y_pred_gini = tree_clf_gini.predict(X_test)

# Visualizing decision tree
fig = plt.figure(1, figsize=(16, 10), dpi=100)
plot_tree(
    tree_clf_gini,
    feature_names=input_names,
    class_names=target_names,
    filled=True,
    fontsize=6,
)
# Writing to local repository as C:\\Users\user_name
fig.savefig("spambase_tree_gini.png")

# Initialize the decision tree object
impurity = "entropy"  # homogeneity criteria for leaves
depth = 4  #  depth of the tree
# Instance of decision tree class
tree_clf_entropy = DecisionTreeClassifier(criterion=impurity, max_depth=depth)

# Fitting decision tree (tree induction + pruning)
tree_clf_entropy.fit(X_train, y_train)  # fitting decision tree to training data
score_entropy = tree_clf_entropy.score(
    X_train, y_train
)  # Goodness of tree on training dataset
score_test = tree_clf_entropy.score(X_test, y_test)  # Goodness of tree on test dataset

# Predicting spam for test data
y_pred_entropy = tree_clf_entropy.predict(X_test)

# Visualizing decision tree
fig = plt.figure(2, figsize=(16, 10), dpi=100)
plot_tree(
    tree_clf_entropy,
    feature_names=input_names,
    class_names=target_names,
    filled=True,
    fontsize=6,
)
# Writing to local repository as C:\\Users\user_name
fig.savefig("spambase_tree_entropy.png")

# End of code
