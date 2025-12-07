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
Functions: float64, normal, binomial, expit, logit, figure, scatter, plot

@author: Márton, Faculty of Informatics, University of Debrecen
License: BSD 3 clause
"""

import numpy as np;  # Numerical Python library
import scipy as sp;  # Scientific Python library
import matplotlib.pyplot as plt;  # Matlab-like Python module
import matplotlib.colors as mcolors;  # coloring tools from MatPlotLib
from sklearn.linear_model import LogisticRegression; # class for logistic regression


# Default parameters
n = 1000;  # sample size
w0 = 2;   # intercept
w1 = 3;   # slope

# Enter parameters from consol
user_input = input('Slope of regression line [default:2]: ');
if len(user_input) != 0 :
    w1 = np.float64(user_input);
user_input = input('Intercept of regression line [default:3]: ');
if len(user_input) != 0 :
    w0 = np.float64(user_input);
user_input = input('Sample size [default:1000]: ');
if len(user_input) != 0 :
    n = np.int64(user_input);
del user_input;    

#  Generating random sample: x - input array, y - output array of {0,1}
x = np.random.normal(0, 1, n); # input 1D array of standard normal distribution
z = w0 + w1*x;   # regression equation for latent variable
p = sp.special.expit(z);  # logistic transformation of latent variable using special function module
y = np.random.binomial(1,p);  # generating 1d array output of random target from latent probability p
# using Bernoulli (Binomial(1,p)) random generator of random module

# Scatterplot for the first 100 records with hidden regression line 
n_point = min(100,n);
plt.figure(1);
plt.title('Scatterplot with regression line');
plt.xlabel('x input variable');
plt.ylabel('z latent variable');
xmin = min(x)-0.3;  #  left bound for X
xmax = max(x)+0.3;  #  right bound for X
zmin = w0 + w1*xmin;  # lower bound for z (for positive slope)
zmax = w0 + w1*xmax;  # upper bound for y (for positive slope)
plt.scatter(x[0:n_point], z[0:n_point], color='blue', label='datapoints');
plt.plot([xmin,xmax], [zmin,zmax], color='black', label='regression line');
plt.legend(loc='lower right');
plt.show(); 

# The logistic function
plt.figure(2);
plt.title('Logistic function');
plt.xlabel('x');
plt.ylabel('f(x)');
res = 0.0001;  #  resolution of the graph
base = np.arange(-5,5,res);
plt.scatter(base, sp.special.expit(base), s=5, color="blue", label='function curve');
plt.hlines(y=0, xmin=-5, xmax=5, colors='black', linestyles='--', lw=2);
plt.hlines(y=1, xmin=-5, xmax=5, colors='black', linestyles='--', lw=2);
plt.legend(loc='center right');
plt.show(); 

# Scatterplot for data with probabilities
plt.figure(3);
plt.title('Scatterplot for data with latent probabilities');
plt.xlabel('x input');
plt.ylabel('y output');
colors = ['blue','red'];
plt.scatter(x, p, color="black", label='latent probability');
plt.scatter(x, y, c=y, cmap=mcolors.ListedColormap(colors), label='datapoints');
plt.legend(loc='lower right');
plt.show(); 

# Fitting logistic regression
logreg = LogisticRegression();  # an instance of LogisticRegression class
X = x.reshape(1, -1).T;  # transforming 1d array input into 2D array (necessary for fit method)
logreg.fit(X,y);  #  fitting the model to data
w0hat = logreg.intercept_[0];  #  estimated intercept
w1hat = logreg.coef_[0,0];  #  estimated slope
accuracy = logreg.score(X,y);  #  accuracy for model fitting
y_pred_logreg = logreg.predict(X);  #  prediction of the target
proba_pred_logreg = logreg.predict_proba(X);  # posterior distribution for the target

# Printing the results
print(f'Estimated slope:{w1hat:6.4f} (True slope:{w1})');
print(f'Estimated intercept:{w0hat:6.4f} (True intercept:{w0})');
print(f'Accuracy:{accuracy:6.4f}');

# Scatterplot for latent probabilities
plt.figure(4);
n_point = min(1000,n);
plt.title('Scatterplot for fitting latent probabilities');
plt.xlabel('p true');
plt.ylabel('p estimated');
plt.scatter(p[0:n_point], proba_pred_logreg[0:n_point,1], color='blue', label='probabilities');
plt.plot([0,1], [0,1], color='black', label='diagonal');
plt.legend(loc='lower right');
plt.show(); 

# Computing the latent variable z as decision function: 
# the predicted target is 1 if y>0 and 0 if y<0
z_pred = logreg.decision_function(X);
# Predicition of latent variable z by logit transformation
# Compare the above values
z_pred1 = sp.special.logit(proba_pred_logreg[:,1]);  

# Scatterplot for latent variable z
plt.figure(5);
n_point = min(1000,n);
plt.title('Scatterplot for latent variable');
plt.xlabel('z true');
plt.ylabel('z estimated');
plt.scatter(z[0:n_point], z_pred[0:n_point], color='blue', label='z points');
zmin = min(z)-0.3;
zmax = max(z)+0.3;
zmin1 = min(z_pred)-0.3;
zmax1 = max(z_pred)+0.3;
plt.plot([zmin,zmax], [zmin1,zmax1], color='black', label='diagonal');
plt.legend(loc='lower right');
plt.show(); 

# End of code

