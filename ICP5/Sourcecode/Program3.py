"""Importing pandas library"""
import pandas as pd

"""Using linear_model from sklearn library"""
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

"""Importing numpy library as np"""
import numpy as np

"""Importing matplotlib.pyplot library for plotting graphs"""
import matplotlib.pyplot as plt

"""Reading data through data2.csv file"""
restaurant_data = pd.read_csv("data2.csv")

"""selecting train_data and test_data"""
train_data = restaurant_data.drop(['revenue', 'City Group', 'Type'], axis=1)
test_data = restaurant_data["revenue"].astype(str)

"""Implementing linear model using library function"""
REGRESSION = linear_model.LinearRegression()
REGRESSION.fit(train_data, test_data)
REVENUE_prediction = REGRESSION.predict(train_data)

"""Printing R2 score"""
print("R2: %.2f" % r2_score(test_data, REVENUE_prediction))

"""Printing RMSE score"""
print("RMSE: %.2f" % mean_squared_error(test_data, REVENUE_prediction))

""" Working with numeric features"""
numeric_features = restaurant_data.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print(corr['revenue'].sort_values(ascending=False)[0:6],'\n')

quality_pivot = restaurant_data.pivot_table(index=['P2'], values=['revenue'], aggfunc=np.median)
quality_pivot.plot(kind='bar', color='red')
plt.show()

correlated_features = restaurant_data[['P2', 'P28', 'P6', 'P21', 'P11']]
correlation = restaurant_data['revenue']
Regression_1 = linear_model.LinearRegression()
Regression_1.fit(correlated_features, correlation)

prediction = Regression_1.predict(correlated_features)

print("R2: %.2f" %r2_score(correlation, prediction))
print("RMSE:", mean_squared_error(correlation, prediction))