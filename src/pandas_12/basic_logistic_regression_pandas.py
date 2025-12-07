# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 23:10:39 2020
Corrected on Fri Nov 15 16:28:05 2024

Task: Basic univariate logistic regression for simulated data
Input: slope and intercept parameters, sample size
Output: estimated parameters, accuracy performance metric, and diagnostic plots

Python tools
Libraries: numpy, scipy, matplotlib, sklearn
Modules: random, special, pyplot, colors, linear_model
Classes: LogisticRegression
Functions: float64, normal, binomial, expit, scatter, plot, line

@author: Márton, Faculty of Informatics, University of Debrecen
License: BSD 3 clause
"""

import numpy as np  # Numerical Python library
import scipy as sp  # Scientific Python library
import matplotlib.pyplot as plt  # Matlab-like Python module
import pandas as pd  # Python Data Analysis Library
import matplotlib.colors as mcolors  # coloring tools from MatPlotLib
from sklearn.linear_model import LogisticRegression  # class for logistic regression


# Default parameters
n = 1000  # sample size
w0 = 2  # intercept
w1 = 3  # slope

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
del user_input

#  Generating random sample: x - input array, y - output array of {0,1}
x = np.random.normal(0, 1, n)  # input 1D array of standard normal distribution
z = w0 + w1 * x  # regression equation for latent variable
p = sp.special.expit(
    z
)  # logistic transformation of latent variable using special function module
y = np.random.binomial(
    1, p
)  # generating 1d array output of random target from latent probability p
# using Bernoulli (Binomial(1,p)) random generator of random module

d = {
    "input": x,
    "latent": z,
    "probability": p,
    "target": y,
}  # dictionary for generated data
df = pd.DataFrame(data=d)  #  data in dataframe
df_input = pd.DataFrame(data=df["input"])  # input data in dataframe

# Visualization of latent regression
df.plot.scatter(x="input", y="latent", c="target", colormap="coolwarm")
plt.title("Scatterplot of latent regression")
plt.show()

res = 0.0001  #  resolution of the graph
d_logistic = {
    "x": np.arange(-5, 5, res),
    "f(x)": sp.special.expit(np.arange(-5, 5, res)),
}
df_logistic = pd.DataFrame(data=d_logistic)  # logistic data into dataframe
df_logistic.plot.scatter(x="x", y="f(x)")
plt.title("Logistic function")
plt.show()


# Scatterplot for data with probabilities with common axis
colors = ["blue", "red"]  # colors for target
ax = df.plot.scatter(
    x="input", y="probability", color="black"
)  # 1st plot for latent probabilities
ax2 = df.plot.scatter(
    x="input", y="target", c=y, cmap=mcolors.ListedColormap(colors), ax=ax
)  # 2nd plot for datapoints
plt.title("Scatterplot for data with latent probabilities")
plt.xlabel("x input")
plt.ylabel("y output")
plt.show()


# Fitting logistic regression
logreg = LogisticRegression()  # an instance of LogisticRegression class
logreg.fit(df_input, df["target"])  #  fitting the model to data
w0hat = logreg.intercept_[0]  #  estimated intercept
w1hat = logreg.coef_[0, 0]  #  estimated slope
accuracy = logreg.score(df_input, df["target"])  #  accuracy for model fitting
y_pred_logreg = logreg.predict(df_input)  #  prediction of the target
proba_pred_logreg = logreg.predict_proba(
    df_input
)  # posterior distribution for the target
df["posterior"] = proba_pred_logreg[:, 1]  # adding the success probability to dataframe

# Printing the results
print(f"Estimated slope:{w1hat:6.4f} (True slope:{w1})")
print(f"Estimated intercept:{w0hat:6.4f} (True intercept:{w0})")
print(f"Accuracy:{accuracy:6.4f}")

# Scatterplot for posterior probabilities
ax = df.plot.scatter(
    x="probability", y="posterior", color="blue", label="probabilities"
)  # probabilities plot
line = pd.DataFrame(data=[[0, 0], [1, 1]])  # the 2 endpoints of diagonal line
line.plot.line(
    x=0, y=1, color="black", label="diagonal", ax=ax
)  # drawing the diagonal line
plt.title("Scatterplot for fitting latent probabilities")
plt.xlabel("p true")
plt.ylabel("p estimated")
plt.legend(loc="lower right")
plt.show()


# End of code
