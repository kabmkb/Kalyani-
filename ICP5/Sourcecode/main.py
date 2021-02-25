"""Importing pandas library"""
import pandas as pd

"""Importing matplotlib.pyplot library for plotting graphs"""
import matplotlib.pyplot as plt

"""Using stats function from scipy library for finding statistics"""
from scipy import stats

"""Importing numpy library as np"""
import numpy as np

"""Reading data from data.csv file"""
read_data = pd.read_csv('data.csv')

"""reading GarageArea and SalePrice columns from dataset"""
GARAGE_AREA = read_data['GarageArea']
SALE_PRICE = read_data['SalePrice']

"""Plotting GarageArea and SalePrice in scatter plot"""
plt.scatter(GARAGE_AREA, SALE_PRICE, color="blue")

"""Giving labels for 'X', 'Y' axis and title for the graph"""
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.title('LINEAR REGRESSION MODEL')
plt.show()

"""Computing Z score of each value and removing outlier data"""
data_all = pd.concat([read_data['GarageArea'], read_data['SalePrice']], axis=1)
z = np.abs(stats.zscore(data_all))
threshold = 3
data = data_all[(z < threshold).all(axis=1)]
data_abnormal = data_all[(z >= threshold).all(axis=1)]

"""Plotting GarageArea and SalePrice in scatter plot after removing outlier data"""
GARAGE_AREA=data['GarageArea']
SALE_PRICE=data['SalePrice']
plt.scatter(GARAGE_AREA, SALE_PRICE, color="red")
plt.xlabel('GARAGE_AREA')
plt.ylabel('SALE_PRICE')
plt.title('LINEAR REGRESSION MODEL')
plt.show()