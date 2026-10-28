import numpy as np
house_sizes = np.array([500,800,600,1500,1200,2000,2500,3000,1400,2200])
house_prices = np.array([20,30,24,60,50,85,95,105,120,57,100])
print(house_prices)
print(house_sizes)
print("mean of the given data is:")
m1 = np.mean(house_prices)
print(m1)
m2 = np.mean(house_sizes)
print(m2)
print("minimum and maximum values of house prices respectively are")
print(np.min(house_prices))
print(np.max(house_prices))
print("minimum and maximum values of house sizes respectively are")
print(np.min(house_sizes))
print(np.max(house_sizes))
print("median of house prices and house sizes respectively are")
np.median(house_prices)
np.median(house_sizes)
print("standard deviation of house prices and house sizes respectively are")
print(np.std(house_prices))
print(np.std(house_sizes))

x_train =  house_sizes[:8]
y_train =  house_prices[:8]
print(x_train)
print(y_train)
x_trainr  = x_train.reshape(-1,1)
print(x_trainr)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_trainr, y_train)


slope = model.coef_[0]
intercept = model.intercept_
print(slope)
print(intercept)






x = input("enter the house size in sq.ft: ")
x = float(x)
y = slope*x + intercept
print(y,"lacs")