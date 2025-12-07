# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 09:55:30 2020
Corrected on Sat Nov 16 14:02:28 2024

Task: Assessing of classifiers fitted for Spambase dataset
Binary (Binomial) classification problem
Classifiers: logistic regression, naive Bayes
Results: confusion matrix, ROC curve, AUC value
Original data source: https://archive.ics.uci.edu/ml/datasets/spambase

Python tools
Libraries: numpy, matplotlib, urllib, sklearn
Modules: pyplot, request, linear_model, naive_bayes, model_selection, metrics
Classes: LogisticRegression, GaussianNB, ConfusionMatrixDisplay, RocCurveDisplay
Functions: urlopen, loadtxt, train_test_split, confusion_matrix, roc_curve, auc

@author: Márton Ispány, Faculty of Informatics, University of Debrecen
License: BSD 3 clause
"""

import numpy as np  # Numerical Python library
import matplotlib.pyplot as plt  # Matlab-like Python module
from urllib.request import urlopen  # importing url handling
from sklearn.linear_model import LogisticRegression  #  Logistic Regression Classifier
from sklearn.naive_bayes import GaussianNB  #  Naive Bayes Classifier
from sklearn.model_selection import train_test_split  # splitting function
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    RocCurveDisplay,
)  #  performance metrics


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

# Fitting logistic regression
logreg_clf = LogisticRegression(solver="liblinear", max_iter=1000)
logreg_clf.fit(X_train, y_train)
pred_logreg_train = logreg_clf.predict(X_train)  # spam prediction for train
pred_logreg_test = logreg_clf.predict(X_test)  # spam prediction for test
accuracy_logreg_train = logreg_clf.score(X_train, y_train)  # goodness-of-fit for train
accuracy_logreg_test = logreg_clf.score(X_test, y_test)  # goodness-of-fit for test
cm_logreg_train = confusion_matrix(y_train, pred_logreg_train)  # train confusion matrix
cm_logreg_test = confusion_matrix(y_test, pred_logreg_test)  # test confusion matrix
proba_logreg = logreg_clf.predict_proba(X_test)  #  prediction probabilities for test


# Plotting non-normalized confusion matrix

disp = ConfusionMatrixDisplay(cm_logreg_train, display_labels=target_names)
disp.plot(cmap=plt.cm.Greens)
plt.title("Confusion matrix for training dataset (logistic regression)")
plt.show()

disp = ConfusionMatrixDisplay(cm_logreg_test, display_labels=target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion matrix for test dataset (logistic regression)")
plt.show()

# Fitting naive Bayes classifier
naive_bayes_clf = GaussianNB()
naive_bayes_clf.fit(X_train, y_train)
pred_naive_bayes_train = naive_bayes_clf.predict(X_train)  # spam prediction for train
pred_naive_bayes_test = naive_bayes_clf.predict(X_test)  # spam prediction for test
cm_naive_bayes_train = confusion_matrix(
    y_train, pred_naive_bayes_train
)  # train confusion matrix
cm_naive_bayes_test = confusion_matrix(
    y_test, pred_naive_bayes_test
)  # test confusion matrix
proba_naive_bayes = naive_bayes_clf.predict_proba(X_test)  #  prediction probabilities

# Plotting non-normalized confusion matrix
disp = ConfusionMatrixDisplay(cm_naive_bayes_train, display_labels=target_names)
disp.plot(cmap=plt.cm.Greens)
plt.title("Confusion matrix for training dataset (naive Bayes)")
plt.show()

disp = ConfusionMatrixDisplay(cm_naive_bayes_test, display_labels=target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion matrix for test dataset (naive Bayes)")
plt.show()

# Computing false and true positive rate and plotting ROC curve for logistic regression clf
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, proba_logreg[:, 1])
auc_logreg = auc(fpr_logreg, tpr_logreg)
disp = RocCurveDisplay(
    fpr=fpr_logreg,
    tpr=tpr_logreg,
    roc_auc=auc_logreg,
    estimator_name="Logistic regression",
)
disp.plot()
plt.title("ROC curve for test dataset (logistic regression)")
plt.show()

# Computing false and true positive rate and plotting ROC curve for naive Bayes clf
fpr_naive_bayes, tpr_naive_bayes, _ = roc_curve(y_test, proba_naive_bayes[:, 1])
auc_naive_bayes = auc(fpr_naive_bayes, tpr_naive_bayes)
disp = RocCurveDisplay(
    fpr=fpr_naive_bayes,
    tpr=tpr_naive_bayes,
    roc_auc=auc_naive_bayes,
    estimator_name="naive Bayes",
)
disp.plot()
plt.title("ROC curve for test dataset (naive Bayes)")
plt.show()


# Plotting previous ROC curves in one figure
plt.figure(7)
lw = 2
plt.plot(
    fpr_logreg,
    tpr_logreg,
    color="red",
    lw=lw,
    label="Logistic regression (AUC = %0.2f)" % auc_logreg,
)
plt.plot(
    fpr_naive_bayes,
    tpr_naive_bayes,
    color="blue",
    lw=lw,
    label="Naive Bayes (AUC = %0.2f)" % auc_naive_bayes,
)
plt.plot([0, 1], [0, 1], color="black", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic curve")
plt.legend(loc="lower right")
plt.show()

# End of code
