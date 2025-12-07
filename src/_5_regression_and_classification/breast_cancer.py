# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 15:27:34 2020
Corrected on Sun Nov 17 09:03:40 2024

Task: Fitting logistic regression model for breast_cancer dataset
Results: regression model, prediction, and graphical comparisons

Python tools
Libraries: numpy, matplotlib, sklearn
Modules: random, special, pyplot, colors, linear_model, model_selection
Classes: LogisticRegression
Functions: load_breast_cancer, train_test_split, scatter, hist, int16, int8

@author: Márton, Faculty of Informatics, University of Debrecen
License: BSD 3 clause
"""

import numpy as np  # Numerical Python library
import scipy as sp  # Scientific Python library
import matplotlib.pyplot as plt  # MATLAB-like plotting framework
import matplotlib.colors as mcolors  # coloring tools from MatPlotLib
from matplotlib.lines import Line2D  # line handling tools from MatPlotLib
from sklearn.datasets import load_breast_cancer  # dataset loader
from sklearn.model_selection import train_test_split  # splitting function
from sklearn.linear_model import LogisticRegression  # class for logistic regression

# Loading the dataset
cancer = load_breast_cancer()
n = cancer.data.shape[0]  # number of records
p = cancer.data.shape[1]  # number of attributes

# Printing the basic parameters
print(f"Number of records:{n}")
print(f"Number of attributes:{p}")

# Printing a data value
# Deafult
record = 10
feature = 2
# Enter axis from consol
user_input = input("X axis [0..568, default:10]: ")
if len(user_input) != 0 and np.int16(user_input) >= 0 and np.int16(user_input) < n:
    record = np.int16(user_input)
user_input = input("Y axis [0..29, default:2]: ")
if len(user_input) != 0 and np.int8(user_input) >= 0 and np.int8(user_input) < p:
    feature = np.int8(user_input)
print(cancer.feature_names[feature], "[", record, "]:", cancer.data[record, feature])
del user_input

# Fitting logistic regression for whole dataset
logreg = LogisticRegression(
    solver="liblinear"
)  # an instance of LogisticRegression class
logreg.fit(cancer.data, cancer.target)  #  fitting the model to data
intercept = logreg.intercept_[0]  #  intercept (constant) parameter
weights = logreg.coef_[0, :]  #  regression coefficients (weights)
accuracy = logreg.score(cancer.data, cancer.target)  # accuracy of the model

# Prediction by scikit-learn
target_pred = logreg.predict(cancer.data)  # prediction of the output
proba_pred_sklearn = logreg.predict_proba(cancer.data)[
    :, 1
]  # aposterior probability of the positive (1) class
# Prediction by numpy
z = np.dot(cancer.data, weights) + intercept  #  latent linear prediction
proba_pred_numpy = sp.special.expit(z)  #  logistic transformation

# Visualizing the prediction
colors = ["blue", "red"]  # colors for target values
# Defining artifical hidden lines for classes refereeing them in the legend
custom_lines = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[0], markersize=10),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[1], markersize=10),
]
fig = plt.figure(1)
plt.title("Comparing various prediction methods")
plt.xlabel("Sklearn prediction")
plt.ylabel("Numpy prediction")
plt.scatter(
    proba_pred_sklearn,
    proba_pred_numpy,
    s=50,
    c=cancer.target,
    cmap=mcolors.ListedColormap(colors),
)
plt.legend(custom_lines, cancer.target_names, loc="lower right")
plt.show()

# Partitioning for train/test dataset
test_rate = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=test_rate, random_state=2024
)
n_train = X_train.shape[0]  # size of the training set
n_test = X_test.shape[0]  # size of the test set
# Printing the dataset size
print(f"Number of training records:{n_train}")
print(f"Number of test records:{n_test}")

# Fitting logistic regression
logreg1 = LogisticRegression(
    solver="liblinear"
)  # new instance of LogisticRegression class
logreg1.fit(X_train, y_train)
intercept1 = logreg1.intercept_[0]
weights1 = logreg1.coef_[0, :]
score_train = logreg1.score(X_train, y_train)
score_test = logreg1.score(X_test, y_test)

# Prediction of a random test record
ind = np.random.randint(0, n_test)
test_record = X_test[ind, :].reshape(1, -1)
pred_class = logreg1.predict(test_record)[0]
pred_distr = logreg1.predict_proba(test_record)
print(
    "Prediction of test record with index", ind, ":", pred_class, "/true: ", y_test[ind]
)
print("Prediction of positive class probability:", pred_distr[0, 1])

# Replication analysis of logistic regression model
rep = 1000
score = []
logreg2 = LogisticRegression(solver="liblinear")
for i in range(rep):
    X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target, test_size=test_rate
    )
    logreg2.fit(X_train, y_train)
    score.append(logreg2.score(X_test, y_test))

score_mean = np.mean(score)
score_std = np.std(score)

# Printing the results
print(f"Mean of accuracy:{score_mean}")
print(f"Standard deviation of accuracy:{score_std}")

# Histogram for the accuracy
plt.figure(2)
count, bins, ignored = plt.hist(np.array(score), 10, density=True)
plt.plot(
    bins,
    1
    / (score_std * np.sqrt(2 * np.pi))
    * np.exp(-((bins - score_mean) ** 2) / (2 * score_std**2)),
    linewidth=2,
    color="red",
)
plt.show()

# End of code
