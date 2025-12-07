# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 11:05:41 2019
Corrected on Sat Nov 16 18:26:32 2024

Task: Fitting linear regression model for diabetes dataset (Scikit-learn toy dataset)
Results: regression model, prediction and graphical comparisons

Python tools    
Libraries: numpy, matplotlib, sklearn
Modules: pyplot, datasets, model_selection, linear_model
Classes: LinearRegression
Functions: load_diabetes, figure, scatter, plot, train_test_split, int16, int8

@author: Márton Ispány, Faculty of Informatics, University of Debrecen
License: BSD 3 clause
"""

import numpy as np;   # Numerical Python library
import matplotlib.pyplot as plt;   # Matlab-like Python module
from sklearn.datasets import load_diabetes;  # dataset loader
from sklearn.model_selection import train_test_split; # splitting function
from sklearn.linear_model import LinearRegression;  # class for linear regression

# Loading the dataset
diabetes = load_diabetes();
n = diabetes.data.shape[0];  # number of records
p = diabetes.data.shape[1];  # number of attributes

# Printing the basic dimensions
print(f'Number of records:{n}');
print(f'Number of attributes:{p}');

# Printing a data value
# Deafult
record = 10;
feature = 2;
# Enter axis from consol
user_input = input('X axis [0..441, default:10]: ');
if len(user_input) != 0 and np.int16(user_input)>=0 and np.int16(user_input)<n :
    record = np.int16(user_input);
user_input = input('Y axis [0..9, default:2]: ');
if len(user_input) != 0 and np.int8(user_input)>=0 and np.int8(user_input)<p :
    feature = np.int8(user_input); 
print(diabetes.feature_names[feature],'[',record,']:', diabetes.data[record,feature]); 
del user_input;

# Partitioning into training and test sets
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, 
                                                    test_size=0.2, random_state=2024)

# Fitting linear regression
reg = LinearRegression();  # an instance of LinearRegression class
reg.fit(X_train,y_train);   #  fitting the model to data
intercept = reg.intercept_;  #  intecept (constant) parameter
weights = reg.coef_;    #  regression coefficients (weights)
R2_train = reg.score(X_train,y_train);   #  R-square for goodness of fit
R2_test = reg.score(X_test,y_test);
y_test_pred = reg.predict(X_test);   # prediction for test dataset

# Comparison of true and predicted target values  
plt.figure(1);
plt.title('Diabetes prediction');
plt.xlabel('True disease progression');
plt.ylabel('Predicted disease progression');
ymin = min(y_test)-5;
ymax = max(y_test)+5;
plt.scatter(y_test, y_test_pred, color="blue", label='datapoints');
plt.plot([ymin,ymax], [ymin,ymax], color='red', label='diagonal');
plt.legend(loc='lower right');
plt.show(); 

# Prediction for whole dataset
pred_sklearn = reg.predict(diabetes.data);  # prediction by sklearn
pred_numpy = intercept*np.ones((n))+np.dot(diabetes.data,weights);  # prediction by numpy
error = diabetes.target-pred_numpy;  # error of prediction
centered_target = diabetes.target-diabetes.target.mean(); 
R2_sklearn = reg.score(diabetes.data, diabetes.target);  # computing R-square by sklearn
R2_numpy = 1-np.dot(error,error)/np.dot(centered_target,centered_target); # computing R-square by numpy
# Compare the last two value!


# End of code
