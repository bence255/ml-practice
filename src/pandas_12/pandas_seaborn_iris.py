# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 13:33:31 2020
Corrected on Sun Dec 01 10:47:12 2024

Task: Statistical analysis and visualization of Iris dataset
using pandas and seaborn

Python tools    
Libraries: numpy, matplotlib, sklearn, pandas, seaborn
Modules: pyplot, plotting
Classes:  
Functions: load_iris, groupby, mean, std, corr, describe, andrews_curves, 
parallel_coordinates, scatter_matrix, displot, histplot, boxplot, pairplot, heatmap

@author: Márton Ispány, Faculty of Informatics, University of Debrecen
License: BSD 3 clause
"""

import numpy as np;  # Numerical Python library
import matplotlib.pyplot as plt;  # MATLAB-like Python module
import pandas as pd;  # Python Data Analysis Library
import seaborn as sns;  # Statistical Graphics in Python
from sklearn.datasets import load_iris; # Iris dataset

# Loading dataset as frame
iris = load_iris(as_frame=True);
n = iris.data.shape[0]; # number of records
p = iris.data.shape[1]; # number of attributes
k = iris.target_names.shape[0]; # number of target values

# Basic pandas

iris_by_target = iris.frame.groupby(by='target');  # grouping by target
# Exporting a target group from the grouped dataframe
for i in range(k):
    if iris.target_names[i]=='setosa': setosa_ind=i;
iris_setosa = iris_by_target.get_group(setosa_ind);
iris_set = iris_setosa.drop(columns=['target']);
first = 5;
# Printing the first observations by target
for target_value, group in iris_by_target:
   print ('Target: ',iris.target_names[target_value]);
   print (group[0:first]);
 
# Basic descriptive stats
mean = iris.frame.drop(columns=['target']).mean(); 
std = iris.frame.drop(columns=['target']).std();
corr = iris.frame.drop(columns=['target']).corr();
mean_by_target = iris_by_target.mean();  # mean
std_by_target = iris_by_target.std();  # standard deviation
corr_by_target = iris_by_target.corr();  #  correlations
desc_stat_by_target = iris_by_target.describe();  # desc stat with quantiles


# Plotting using multidimensional tools of pandas
# Andrews curves
plt.figure(1);
pd.plotting.andrews_curves(iris.frame,class_column='target',
                           color=['blue','green','red']);
plt.show();
# Parallel axis
plt.figure(2);
pd.plotting.parallel_coordinates(iris.frame,class_column='target',color=['blue','green','red']);
plt.show();

# Scatter matrix
pd.plotting.scatter_matrix(iris.data);

# Basic seaborn                         

# Loading seaborn's default theme and color palette
sns.set(); 

# Plotting two attributes
colors = ['blue','red','green'];
sns.relplot(data=iris.frame, x='sepal length (cm)', y='petal length (cm)', 
            hue='target', palette=colors);
# Default axis
x_axis = 0;  # x axis attribute (0..3)
y_axis = 1;  # y axis attribute (0..3)
# Enter axis from consol
user_input = input('X axis [0..3, default:0]: ');
if len(user_input) != 0 and np.int8(user_input)>=0 and np.int8(user_input)<=3 :
    x_axis = np.int8(user_input);
user_input = input('Y axis [0..3, default:1]: ');
if len(user_input) != 0 and np.int8(user_input)>=0 and np.int8(user_input)<=3 :
    y_axis = np.int8(user_input); 
sns.relplot(data=iris.frame, x=iris.feature_names[x_axis], 
            y=iris.feature_names[y_axis], hue='target', palette=colors);
# Scatterplot of datapoints according to the two input axis
plt.figure(6);    
sns.scatterplot(data=iris.frame, x=iris.feature_names[x_axis], 
                        y=iris.feature_names[y_axis], hue='target', palette=colors);
plt.show();

# Distribution plot of an attribute with kernel density estimation
sns.displot(data=iris.frame, x=iris.feature_names[x_axis], kde=True)
plt.show();

# Histogram of an attribute
plt.figure(8);
sns.histplot(data=iris.frame, x=iris.feature_names[x_axis], 
             hue='target', palette=colors);
plt.show();

# 
plt.figure(9);
sns.histplot(data=iris.frame, x=iris.feature_names[x_axis], 
             y=iris.feature_names[y_axis], hue='target', palette=colors);
plt.show();

# Boxplot of an attribute grouped by target
plt.figure(10);
sns.boxplot(x=iris.frame['target'], 
            y=iris.frame['sepal length (cm)']);
plt.show();

# Matrix plot of attributes coloring by target
sns.pairplot(data=iris.frame,hue='target', palette=colors);
plt.show();

# Heatmap of the full dataset
plt.figure(12);
sns.heatmap(iris.data)
plt.show();
