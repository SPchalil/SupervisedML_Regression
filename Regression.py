# Supervised ML - Regression

#  Step 1: %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

print('Libraries have been imported.')


# Step 2: Importing the Dataset
# Read in the dataset
data = pd.read_csv('advertising.csv')


# Step 3: Inspecting the Dataset
# Check the number of features and observations
data.shape

# Display the first 5 rows - (scan few entries of the dataset to get a sense of the features and their values)
data.head()

# Step 4: Performing Some Preliminary Data Visualizations
# Before, running a regression model, it is good practice to perform some preliminary data visualizations.
# For that, plot scatter plots of sales (the outcome variable) versus each of the three predictor variables.

# Plot a scatter plot of sales vs. TV ad spending budget
data.plot.scatter(x=['TV'],y=['sales'])
# Plot a scatter plot of sales vs. newspaper ad spending budget
data.plot.scatter(x=['newspaper'],y=['sales'])

# Step 5: Running a Simple Linear Regression
# Now, run a simple linear regression on sales and newspaper advertising to figure out if there is a significant relationship between the two.

# Define the variables
feature_cols = ['newspaper']      # Extract the feature of interest
X = data[feature_cols]            # Select the predictor for regression                   
y = data['sales']                 # Select the outcome variable for regression
# Split the data to test and training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
# test_size parameter defines the fraction of data that will be used as test data
# Run a linear regression model
lr = LinearRegression()           # Select the estimator
lr.fit(X_train, y_train)               # Fit the model
# Print the coefficients
print ("intercept : ",lr.intercept_)
print ("coefficient : ",lr.coef_)

# In fact, for linear regression models we need to use a different module called statsmodels which outputs not only the coefficients 
# but also the p-values which will allow us to test for statistical significance. 
# Recall that statsmodels employs ordinary least squares (OLS) which is a technique to estimate the unknown parameters in a linear regression model.

# Import statsmodels 
import statsmodels.api as sm
import statsmodels.formula.api as smf
# Estimate the simple linear regression
est = smf.ols('sales ~ newspaper', data).fit()    # Regresses sales (y) on newspaper (X)
est.summary()   # Print results
Let us now compute the MSE to evaluate the model performance.
# Perform predictions using the test data
predictions_1 = lr.predict(X_test)
predictions_1
# Obtain the MSE
metrics.mean_squared_error(y_test, lr.predict(X_test))
# Note that the MSE is almost 28. This value is meaningless unless compared with an MSE value of another model. We will then, run a multiple linear regression on the same dataset and compare the results.

# Step 6: Running a Multiple Linear Regression
# Run a multiple linear regression on sales and all three advertising media. And check if there is a significant relationship between between sales and newspaper advertising now.
# Define new parameters
X = data.drop('sales', axis=1)     # Extract the feature of interest                  
y = data['sales']                 # Select the outcome variable for regression
# Split the data to test and training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)
# test_size parameter defines the fraction of data that will be used as test data
# Run a multiple linear regression model 
mlr = LinearRegression()           # Select the estimator
mlr.fit(X_train, y_train)                      # Perform the fit
# Print the coefficients
print ("intercept : ",mlr.intercept_)
print ("coefficient : ",mlr.coef_)
# Make predictions using the test data
predictions_2 = mlr.predict(X_test)
predictions_2

# Estimate the multiple linear regression
est = smf.ols('sales ~ TV + radio + newspaper', data).fit()
est.summary()
# Now, let us compute the MSE for this model.
metrics.mean_squared_error(y_test, mlr.predict(X_test))
# Note the value of MSE is now almost 2.4 which is much lower than the MSE obtained by the simple linear regression model. That indicates that multiple linear regression performs much better for the task at hand.
