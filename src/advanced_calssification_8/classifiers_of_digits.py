# -*- coding: utf-8 -*-
"""
Created on Sun May 10 23:37:40 2020
Corrected on Wed Nov 20 13:34:25 2024

Task: Fitting classifiers for Digits dataset
Classifiers: logistic regression, naive Bayes, nearest neighbor, neural network

Python tools    
Libraries: numpy, sklearn
Modules: datasets, model_selection, linear_model, naive_bayes, neighbors, neural_network
Classes: LogisticRegression, GaussianNB, KNeighborsClassifier, MLPClassifier
Functions: load_digits, train_test_split

@author: Márton Ispány, Faculty of Informatics, University of Debrecen
License: BSD 3 clause
"""

import numpy as np;  # Numerical Python library
from sklearn.datasets import load_digits; # importing scikit-learn datasets
from sklearn.model_selection import train_test_split; # importing splitting
from sklearn.linear_model import LogisticRegression; #  Logistic Regression Classifier
from sklearn.naive_bayes import GaussianNB; #  importing naive Bayes classifier
from sklearn.neighbors import KNeighborsClassifier;    # importing nearest neighbor classifier
from sklearn.neural_network import MLPClassifier; # importing neural network classifier

# loading dataset
digits = load_digits();
n = digits.data.shape[0];  # number of records
p = digits.data.shape[1];  # number of attributes
k = digits.target_names.shape[0];  # number of classes

# Printing the basic dimensions
print(f'Number of records:{n}');
print(f'Number of attributes:{p}');
print(f'Number of classes:{k}');

# Partitioning into training and test sets
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, 
            test_size=0.3, shuffle = True, random_state=2024);

# Fitting logistic regression
logreg_clf = LogisticRegression(solver='liblinear');
logreg_clf.fit(X_train,y_train);
score_train_logreg = logreg_clf.score(X_train,y_train);  #  goodness of fit
score_test_logreg = logreg_clf.score(X_test,y_test);  #  goodness of fit
pred_logreg = logreg_clf.predict(X_test);   # spam prediction
proba_logreg = logreg_clf.predict_proba(X_test);  #  prediction probabilities

# Fitting naive Bayes classifier
naive_bayes_clf = GaussianNB();
naive_bayes_clf.fit(X_train,y_train);
score_train_naive_bayes = naive_bayes_clf.score(X_train,y_train);
score_test_naive_bayes = naive_bayes_clf.score(X_test,y_test);  #  goodness of fit
pred_naive_bayes = naive_bayes_clf.predict(X_test);  # spam prediction
proba_naive_bayes = naive_bayes_clf.predict_proba(X_test);  #  prediction probabilities

# Fitting nearest neighbor classifier
K = 5;  # number of neighbors
knn_clf = KNeighborsClassifier(n_neighbors=K);
knn_clf.fit(X_train,y_train);
score_train_knn = knn_clf.score(X_train,y_train);  #  goodness of fit
score_test_knn = knn_clf.score(X_test,y_test);  #  goodness of fit
pred_knn = knn_clf.predict(X_test);   # spam prediction
proba_knn = knn_clf.predict_proba(X_test);  #  prediction probabilities

# Fitting neural network classifier
neural_clf = MLPClassifier(hidden_layer_sizes=(16), activation='logistic', 
                           solver='lbfgs', max_iter=5000);  #  number of hidden neurons: 16
neural_clf.fit(X_train,y_train);
score_train_neural = neural_clf.score(X_train,y_train);
score_test_neural = neural_clf.score(X_test,y_test);  #  goodness of fit
pred_neural = neural_clf.predict(X_test);   # spam prediction
proba_neural = neural_clf.predict_proba(X_test);  #  prediction probabilities

#  The best model based on train score is Nearest neighbor with 99.8%
#  The best model based on test score is Logistic Regression with 97.4%

# Comparing the results of the 4 classifiers
n_test = X_test.shape[0];
pred = np.zeros((n_test,4));  # Prediction results
pred[:,0] = pred_logreg;
pred[:,1] = pred_naive_bayes;
pred[:,2] = pred_knn;
pred[:,3] = pred_neural;
proba = np.zeros((n_test,4,k));  # Spam probabilities
proba[:,0,:] = proba_logreg;
proba[:,1,:] = proba_naive_bayes;
proba[:,2] = proba_knn;
proba[:,3,:] = proba_neural;

# End of code
