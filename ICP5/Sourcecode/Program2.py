"""Importing pandas library"""
import pandas as pd

"""Using linear_model from sklearn library"""
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

"""Reading data through data2.csv file"""
Restaurant=pd.read_csv('data2.csv')

"""selecting train_data and test_data"""
train_data = Restaurant.drop(['revenue','City Group','Type'], axis=1)
test_data = Restaurant["revenue"].astype(str)

"""Implementing linear model using library function"""
REGRESSION= linear_model.LinearRegression()
REGRESSION.fit(train_data, test_data)
revenue_pred =REGRESSION.predict(train_data)

"""Printing R2 score"""
print("R2: %.4f" % r2_score(test_data, revenue_pred))

"""Printing RMSE score"""
print("RMSE: %0.4f" % mean_squared_error(test_data, revenue_pred))