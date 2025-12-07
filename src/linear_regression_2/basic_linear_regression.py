# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 15:01:42 2020
Corrected on Fri Nov 15 16:16:16 2024

Task: Basic univariate linear regression for simulated data
Input: slope and intercept parameters, sample size, error standard deviation
Output: estimated parameters, R2 performance metric, and diagnostic plots

Python tools
Libraries: numpy, matplotlib, sklearn
Modules: random, pyplot, linear_model
Classes: LinearRegression
Functions: normal, float64, figure, scatter, plot, polyfit

@author: Márton Ispány, Faculty of Informatics, University of Debrecen
License: BSD 3 clause
"""

import numpy as np  # Numerical Python library
import matplotlib.pyplot as plt  # Matlab-like Python module
from sklearn.linear_model import LinearRegression  # class for linear regression

# Default parameters
n = 1000  # sample size
w0 = 3  # intercept
w1 = 2  # slope of regression (weight of X)
sigma = 1  # error standard deviation

# Enter parameters from consol
user_input = input("Slope of regression line [default:2]: ")
if len(user_input) != 0:
    w1 = np.float64(user_input)
user_input = input("Intercept of regression line [default:3]: ")
if len(user_input) != 0:
    w0 = np.float64(user_input)
user_input = input("Sample size [default:1000]: ")
if len(user_input) != 0:
    n = np.int64(user_input)
user_input = input("Error standard deviation [default:1]: ")
if len(user_input) != 0:
    sigma = np.float64(user_input)
del user_input

#  Generating random sample: X - input array, y - output array  (Scikit-learn notation)
X = np.random.normal(0, 1, (n, 1))  # input 2D array of standard normal distribution
eps = np.random.normal(0, sigma, n)  # 1D array of random error
y = w0 + w1 * X[:, 0] + eps  # 1D array of regression equation output

# Scatterplot for the first 100 records with regression line
n_point = min(100, n)
plt.figure(1)
plt.title("Scatterplot of data with regression line")
plt.xlabel("x input")
plt.ylabel("y output")
xmin = min(X) - 0.3  #  left bound for X
xmax = max(X) + 0.3  #  right bound for X
ymin = w0 + w1 * xmin  # lower bound for y (for positive slope)
ymax = w0 + w1 * xmax  # upper bound for y (for positive slope)
plt.scatter(
    X[0:n_point, 0], y[0:n_point], color="blue", label="datapoints"
)  #  scatterplot of data
plt.plot(
    [xmin, xmax], [ymin, ymax], color="red", label="regression line"
)  #  plot of regression line
plt.legend(loc="lower right")
plt.show()

# Fitting linear regression
reg = LinearRegression()  # an instance of LinearRegression class
reg.fit(X, y)  #  fitting the model to data
w0hat = reg.intercept_  #  estimated intercept
w1hat = reg.coef_[0]  #  estimated slope
R2 = reg.score(X, y)  #  R-square for model fitting
y_pred = reg.predict(X)  #  prediction of the target

# Computing the regression coefficients by using basic numpy
# Compare estimates below with b0hat and b1hat
reg_coef = np.ma.polyfit(X[:, 0], y, 1)

# Printing the results
print(f"Estimated slope:{w1hat:6.4f} (True slope:{w1})")
print(f"Estimated intercept:{w0hat:6.4f} (True intercept:{w0})")
print(f"R-square for goodness of fit:{R2:6.4f}")

# Scatterplot for data with true and estimated regression line
plt.figure(2)
plt.title("Scatterplot of data with regression lines")
plt.xlabel("x input")
plt.ylabel("y output")
xmin = min(X) - 0.3
xmax = max(X) + 0.3
ymin = w0 + w1 * xmin
ymax = w0 + w1 * xmax
plt.scatter(X[0:n_point, 0], y[0:n_point], color="blue", label="datapoints")
plt.plot([xmin, xmax], [ymin, ymax], color="black", label="theoretical")
ymin = w0hat + w1hat * xmin
ymax = w0hat + w1hat * xmax
plt.plot([xmin, xmax], [ymin, ymax], color="red", label="estimated")
plt.legend(loc="lower right")
plt.show()

# Scatterplot for target prediction: closer to diagonal better fitting
n_point = min(1000, n)
plt.figure(3)
plt.title("Scatterplot for prediction")
plt.xlabel("True target")
plt.ylabel("Predicted target")
ymin = min(y) - 1
ymax = max(y) + 1
plt.scatter(y[0:n_point], y_pred[0:n_point], color="blue", label="datapoints")
plt.plot([ymin, ymax], [ymin, ymax], color="red", label="diagonal")
plt.legend(loc="lower right")
plt.show()

# End of code
