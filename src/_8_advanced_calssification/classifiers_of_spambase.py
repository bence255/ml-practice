# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 09:36:05 2020
Corrected on Tue Nov 19 10:22:55 2024

Task: Fitting classifiers for Spambase dataset
Classifiers: logistic regression, naive Bayes, nearest neighbor, neural network
Original data source: https://archive.ics.uci.edu/ml/datasets/spambase

Python tools
Libraries: numpy, matplotlib, urllib, sklearn
Modules: pyplot, request, linear_model, naive_bayes, neighbors, neural_network, model_selection
Classes: LogisticRegression, GaussianNB, KNeighborsClassifier, MLPClassifier
Functions: urlopen, loadtxt, train_test_split, scatter

@author: Márton Ispány, Faculty of Informatics, University of Debrecen
License: BSD 3 clause
"""

import numpy as np  # Numerical Python library
import matplotlib.pyplot as plt  # Matlab-like Python module
import matplotlib.colors as col  # coloring tools from MatPlotLib
from urllib.request import urlopen  # importing url handling
from sklearn.model_selection import train_test_split  # importing splitting
from sklearn.linear_model import LogisticRegression  #  Logistic Regression Classifier
from sklearn.naive_bayes import GaussianNB  #  Naive Bayes Classifier
from sklearn.neighbors import KNeighborsClassifier  # Nearest Neighbor Classifier
from sklearn.neural_network import MLPClassifier  # Neural Network Classifier
from matplotlib.lines import Line2D  # line handling tools from MatPlotLib

# Reading the dataset
url = "https://arato.inf.unideb.hu/ispany.marton/DataMining/Practice/Datasets/spamdata.csv"
raw_data = urlopen(url)
data = np.loadtxt(
    raw_data, skiprows=1, delimiter=";"
)  # reading numerical data from csv file
del raw_data

# Reading attribute names
url_names = "https://arato.inf.unideb.hu/ispany.marton/DataMining/Practice/Datasets/spambase.names.txt	"
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

# Printing the basic parameters
n = X.shape[0]  # number of records
p = X.shape[1]  # number of attributes
print(f"Number of records:{n}")
print(f"Number of attributes:{p}")


# Partitioning into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=True, random_state=2024
)

# Fitting logistic regression
logreg_clf = LogisticRegression(solver="liblinear", max_iter=1000)
logreg_clf.fit(X_train, y_train)
score_train_logreg = logreg_clf.score(X_train, y_train)  #  goodness of fit
score_test_logreg = logreg_clf.score(X_test, y_test)  #  goodness of fit
pred_logreg = logreg_clf.predict(X_test)  # spam prediction
proba_logreg = logreg_clf.predict_proba(X_test)  #  prediction probabilities

# Fitting naive Bayes classifier
naive_bayes_clf = GaussianNB()
naive_bayes_clf.fit(X_train, y_train)
score_train_naive_bayes = naive_bayes_clf.score(X_train, y_train)  #  goodness of fit
score_test_naive_bayes = naive_bayes_clf.score(X_test, y_test)  #  goodness of fit
pred_naive_bayes = naive_bayes_clf.predict(X_test)  # spam prediction
proba_naive_bayes = naive_bayes_clf.predict_proba(X_test)  #  prediction probabilities

# Fitting nearest neighbor classifier
K = 5  # number of neighbors
knn_clf = KNeighborsClassifier(n_neighbors=K)
knn_clf.fit(X_train, y_train)
score_train_knn = knn_clf.score(X_train, y_train)  #  goodness of fit
score_test_knn = knn_clf.score(X_test, y_test)  #  goodness of fit
pred_knn = knn_clf.predict(X_test)  # spam prediction
proba_knn = knn_clf.predict_proba(X_test)  #  prediction probabilities

# Fitting neural network classifier
neural_clf = MLPClassifier(
    hidden_layer_sizes=(5), activation="logistic", max_iter=500
)  #  number of hidden neurons: 5
neural_clf.fit(X_train, y_train)
score_train_neural = neural_clf.score(X_train, y_train)  #  goodness of fit
score_test_neural = neural_clf.score(X_test, y_test)  #  goodness of fit
pred_neural = neural_clf.predict(X_test)  # spam prediction
proba_neural = neural_clf.predict_proba(X_test)  #  prediction probabilities

#  The best model based on test score is MLP (Multilayer perceptron)
#  with 93.7%

# Comparing the results of the 4 classifiers
n_test = X_test.shape[0]
pred = np.zeros((n_test, 4))  # Prediction results
pred[:, 0] = pred_logreg
pred[:, 1] = pred_naive_bayes
pred[:, 2] = pred_knn
pred[:, 3] = pred_neural
proba_spam = np.zeros((n_test, 4))  # Spam probabilities
proba_spam[:, 0] = proba_logreg[:, 1]
proba_spam[:, 1] = proba_naive_bayes[:, 1]
proba_spam[:, 2] = proba_knn[:, 1]
proba_spam[:, 3] = proba_neural[:, 1]


# Visualization of spam prediction and probabilities using the best model (MLP)
# Color denotes the class, size denotes the probability
# Default axis
x_axis = 5  # x axis attribute (0..56)
y_axis = 22  # y axis attribute (0..56)
# Enter axis from consol
user_input = input("X axis [0..56, default:5]: ")
if len(user_input) != 0 and np.int8(user_input) >= 0 and np.int8(user_input) <= 56:
    x_axis = np.int8(user_input)
user_input = input("Y axis [0..56, default:22]: ")
if len(user_input) != 0 and np.int8(user_input) >= 0 and np.int8(user_input) <= 56:
    y_axis = np.int8(user_input)
del user_input

colors = ["blue", "red"]  #  colors of ham and spam
custom_lines = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[0], markersize=10),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[1], markersize=10),
]
fig = plt.figure(1)
plt.title("Scatterplot for test spam dataset")
plt.xlabel(input_names[x_axis])
plt.ylabel(input_names[y_axis])
plt.scatter(
    X_test[:, x_axis],
    X_test[:, y_axis],
    s=100 * proba_neural[:, 1],
    c=pred_neural,
    cmap=col.ListedColormap(colors),
)
plt.legend(custom_lines, target_names, loc="upper right")
plt.show()

# End of code
