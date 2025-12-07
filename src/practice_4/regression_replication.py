# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 15:06:05 2020
Corrected on Sat Nov 16 17:08:47 2024

Task: Replication analysis of basic linear regression
Results: descriptive stats and statistical graphs showing the randomness of model parameters

Python tools
Libraries: numpy, matplotlib, sklearn
Modules: pyplot, random, linear_model, utils, model_selection
Classes: LinearRegression
Functions: float64, int64, normal, figure, sample_without_replacement, hist, train_test_split, cross_validate

@author: Márton Ispány, Faculty of Informatics, University of Debrecen
License: BSD 3 clause
"""

import numpy as np  # Numerical Python library
import matplotlib.pyplot as plt  # Matlab-like Python module
from sklearn.linear_model import LinearRegression  # class for linear regression
from sklearn.utils.random import sample_without_replacement  #  sampling function
from sklearn.model_selection import train_test_split  # splitting function
from sklearn.model_selection import cross_validate  # crossvalidation

# Default parameters
n = 100000  # sample size
w0 = 3  # intercept
w1 = 2  # slope
sig = 1  # error

# Enter parameters from consol
user_input = input("Slope of regression line [default:2]: ")
if len(user_input) != 0:
    w1 = np.float64(user_input)
user_input = input("Intercept of regression line [default:3]: ")
if len(user_input) != 0:
    w0 = np.float64(user_input)
user_input = input("Sample size [default:100000]: ")
if len(user_input) != 0:
    n = np.int64(user_input)
user_input = input("Error standard deviation [default:1]: ")
if len(user_input) != 0:
    sigma = np.float64(user_input)

#  Generating random sample
x = np.random.normal(0, 1, n)  #  input 1D array of standard normal distribution
eps = np.random.normal(0, sig, n)  #  1D array of random error
y = w0 + w1 * x + eps  #  1D array of regression equation output

# Default replication parameters
rep = 100  #  number of replications
sample_size = 1000  # sample size
reg = LinearRegression()  # instance of the LinearRegression class
w0hat = []  #  list for intercept
w1hat = []  # list for slope
R2 = []  # list for R-squares

for i in range(rep):
    # random sampling from dataset
    index = sample_without_replacement(n_population=n, n_samples=sample_size)
    x_sample = x[index]
    y_sample = y[index]
    X_sample = x_sample.reshape(
        1, -1
    ).T  # transforming 1d array into 2D array (necessary for fit method)
    reg.fit(X_sample, y_sample)
    w0hat.append(reg.intercept_)
    w1hat.append(reg.coef_)
    R2.append(reg.score(X_sample, y_sample))

w0hat_mean = np.mean(w0hat)
w0hat_std = np.std(w0hat)
w1hat_mean = np.mean(w1hat)
w1hat_std = np.std(w1hat)
R2_mean = np.mean(R2)
R2_std = np.std(R2)

# Printing the results
print(
    f"Mean slope:{w1hat_mean:6.4f} (True slope:{w1}) with standard deviation {w1hat_std:6.4f}"
)
print(
    f"Mean intercept:{w0hat_mean:6.4f} (True intercept:{w0}) with standard deviation {w0hat_std:6.4f}"
)
print(
    f"Mean of R-square for goodness of fit:{R2_mean:6.4f} (standard deviation: {R2_std:6.4f})"
)

# Histograms for parameters and scores
plt.figure(1)
val, bins, patches = plt.hist(np.asarray(w1hat), bins=25, color="g", alpha=0.75)
plt.xlabel("Slope")
plt.ylabel("Frequency")
plt.title("Histogram of slope")
plt.text(
    w1 - 2.5 * w1hat_std, 10, f"$\\mu={w1hat_mean:4.3f},\\ \\sigma={w1hat_std:4.3f}$"
)
plt.xlim(w1 - 3 * w1hat_std, w1 + 3 * w1hat_std)
plt.grid(True)
plt.show()

plt.figure(2)
val, bins, patches = plt.hist(np.asarray(w0hat), bins=25, color="g", alpha=0.75)
plt.xlabel("Intercept")
plt.ylabel("Frequency")
plt.title("Histogram of intercept")
plt.text(
    w0 - 2.5 * w1hat_std, 10, f"$\\mu={w0hat_mean:4.3f},\\ \\sigma={w0hat_std:4.3f}$"
)
plt.xlim(w0 - 3 * w1hat_std, w0 + 3 * w1hat_std)
plt.grid(True)
plt.show()

plt.figure(3)
val, bins, patches = plt.hist(np.asarray(R2), bins=25, color="g", alpha=0.75)
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Histogram of R-square score")
plt.text(0.77, 10, f"$\\mu={R2_mean:4.3f},\\ \\sigma={R2_std:4.3f}$")
plt.xlim(R2_mean - 3 * R2_std, R2_mean + 3 * R2_std)
plt.grid(True)
plt.show()

# Above results clearly demonstrate the dependence of the parameter estimation of the training set
# As the training set comes from the data warehouse randomly the results of a machine learning process
# will be random

# Replication analysis for test dataset
# One training set, one parameter estimation
# Several test sets, distribution of R2 (score) value
R2 = []  # list for R-squares of test sets
# Fitting the model for the first n_train sample
n_train = 10000
x_sample = x[0:n_train]
y_sample = y[0:n_train]
X_sample = x_sample.reshape(
    1, -1
).T  # transforming 1d array into 2D array (necessary for fit method)
reg.fit(X_sample, y_sample)
w0hat = reg.intercept_
w1hat = reg.coef_
R2_train = reg.score(X_sample, y_sample)

for i in range(rep):
    # random sampling from dataset
    index = sample_without_replacement(n_population=n, n_samples=sample_size)
    x_test = x[index]
    y_test = y[index]
    X_test = x_test.reshape(
        1, -1
    ).T  # transforming 1d array into 2D array (necessary for fit method)
    R2.append(reg.score(X_test, y_test))

R2_mean = np.mean(R2)
R2_std = np.std(R2)

# Printing the results
print(
    f"Mean of test R-squares:{R2_mean:6.4f} (standard deviation: {R2_std:6.4f}), training R-square: {R2_train:6.4f}"
)

# Histogram of test R-square scores with train one as red line
plt.figure(4)
val, bins, patches = plt.hist(np.asarray(R2), bins=25, color="g", alpha=0.75)
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Histogram of test R-square score")
plt.text(0.77, 12, f"$\\mu={R2_mean:4.3f},\\ \\sigma={R2_std:4.3f}$")
plt.xlim(R2_mean - 3 * R2_std, R2_mean + 3 * R2_std)
plt.grid(True)
plt.vlines(R2_train, 0, 14, colors="r")
plt.show()

# Splitting the dataset for training and test ones
X_train, X_test, y_train, y_test = train_test_split(
    x.reshape(1, -1).T, y, test_size=0.3, shuffle=True, random_state=2024
)
reg.fit(X_train, y_train)
w0hat = reg.intercept_
w1hat = reg.coef_[0]
R2_train = reg.score(X_train, y_train)
R2_test = reg.score(X_test, y_test)

# Printing the results
print(f"Estimated slope:{w1hat:6.4f} (True slope:{w1})")
print(f"Estimated intercept:{w0hat:6.4f} (True intercept:{w0})")
print(f"Training R-square:{R2_train:6.4f}, Test R-square: {R2_test:6.4f})")

# Crossvalidation of regression model
cv_results = cross_validate(reg, x.reshape(1, -1).T, y, cv=10)
R2_mean = cv_results["test_score"].mean()
R2_std = cv_results["test_score"].std()

# Printing the results
print(
    f"Mean of R-square in crossvalidation:{R2_mean:6.4f} (standard deviation: {R2_std:6.4f})"
)

del val, bins, patches

# End of code
