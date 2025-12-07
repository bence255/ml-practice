# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 07:44:01 2020
Corrected on Sun Nov 18 18:32:08 2024

Task: Decision tree analysis of Iris data

Python tools
Libraries: matplotlib, sklearn
Modules: pyplot, tree
Classes: DecisionTreeClassifier
Functions: plot_tree

@author: Márton Ispány, Faculty of Informatics, University of Debrecen
License: BSD 3 clause
"""

import matplotlib.pyplot as plt  # MATLAB-like plotting framework
from sklearn.datasets import load_iris  # iris loader
from sklearn.tree import DecisionTreeClassifier, plot_tree  # decision tree tools

# Loading the dataset
iris = load_iris()

# Initialize our decision tree object
impurity = "entropy"  # homogeneity criteria for leaves
depth = 3
# Instance of decision tree class
tree_clf_entropy = DecisionTreeClassifier(criterion=impurity, max_depth=depth)

# Fitting decision tree (tree induction + pruning)
tree_clf_entropy.fit(iris.data, iris.target)
score_entropy = tree_clf_entropy.score(iris.data, iris.target)  # Goodness of tree

# Visualizing decision tree
fig = plt.figure(1, figsize=(12, 6), dpi=100)
plot_tree(
    tree_clf_entropy,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    fontsize=8,
)
fig.savefig(
    "iris_tree_entropy.png"
)  # Writing to local repository as C:\\Users\user_name

# Initialize our decision tree object
impurity = "gini"  # homogeneity criteria for leaves
depth = 3  #  depth of the tree
# Instance of decision tree class
tree_clf_gini = DecisionTreeClassifier(criterion=impurity, max_depth=depth)

# Fitting decision tree (tree induction + pruning)
tree_clf_gini.fit(iris.data, iris.target)
score_gini = tree_clf_gini.score(iris.data, iris.target)  # Goodness of tree

# Visualizing decision tree
fig = plt.figure(2, figsize=(12, 6), dpi=100)
plot_tree(
    tree_clf_gini,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    fontsize=8,
)
fig.savefig("iris_tree_gini.png")  # Writing to local repository as C:\\Users\user_name

# End of code
