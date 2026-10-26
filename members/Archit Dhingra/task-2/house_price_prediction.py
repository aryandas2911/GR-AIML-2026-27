import numpy as np
#step 1
print("Version: ",np.__version__)
house_sizes = np.array([
500, 800, 1000, 1200, 1500,
1800, 2000, 2200, 2500, 2800])
house_prices = np.array([
15, 24, 30, 36, 45,
54, 60, 66, 75, 84])
#step 2
print("House sizes: ",house_sizes)
print("House prices: ",house_prices)
print("Number of houses: ",house_sizes.size)
print("Shape of house_sizes: ",house_sizes.shape)
print("Shape of house_prices: ",house_prices.shape)
print("Size and price of third house: ",house_sizes[2],",",house_prices[2])
new_house_sizes=np.append(house_sizes,3500)
new_house_prices=np.append(house_prices,95)
print("Now House sizes: ",new_house_sizes)
print("Now House prices: ",new_house_prices)
#step 3
print("Average of house_sizes: ",np.mean(house_sizes))
print("Minimum of house_sizes: ",np.min(house_sizes))
print("Maximum of house_sizes: ",np.max(house_sizes))
print("Median of house_sizes: ",np.median(house_sizes))
print("SD of house_prices: ",np.std(house_sizes))
print("Average of house_prices: ",np.mean(house_prices))
print("Minimum of house_prices: ",np.min(house_prices))
print("Maximum of house_prices: ",np.max(house_prices))
print("Median of house_prices: ",np.median(house_prices))
print("SD of house_prices: ",np.std(house_prices))
print("Average price can easily change due to expensive houses but the median stays stable")
#step 4
print("Price difference between 3rd and 1st house: ",house_prices[2]-house_prices[0])
print("Size difference between 3rd and 1st house: ",house_sizes[2]-house_sizes[0])
w=(house_prices[2]-house_prices[0])/(house_sizes[2]-house_sizes[0])
print("Relation b/w size and price(w): ",w)
print("Predicted price for 3500 house(price=w*size+b): ",w*3500)
#step 5
X_train = house_sizes[:8]
y_train = house_prices[:8]
X_test = house_sizes[8:]
y_test = house_prices[8:]
print("X_train: ",X_train)
print("y_train: ",y_train)
print("X_test: ",X_test)
print("y_test: ",y_test)
print("Training examples: ",X_train.size)
print("Test examples: ",X_test.size)
print("For 2nd house,X_train: ",X_train[1],"and y_train: ",y_train[1])
print("For 10th house,X_test: ",X_test[1],"and y_test: ",y_test[1])
print("If we test only on training data, model might memorise the data points instead of learning the trend")
#step 6
X=np.mean(X_train)
y=np.mean(y_train)
print("Mean of X_train: ",X)
print("Mean of y_train: ",y)
num_slope=np.sum((X_train-X)*(y_train-y))
print("Numerator of w: ",num_slope)
den_slope=np.sum((X_train-X)**2)
print("Denominator of w: ",den_slope)
w=num_slope/den_slope
print("w: ",w)
b=y-(w*X)
print("b: ",b)
def prediction(size):
  return (w*size)+b
print("Predicted price for 3500 size: ",prediction(3500))
#step 7
print("Predicted price for ",X_test,": ",prediction(X_test))
error=y_test-prediction(X_test)
print("Errors: ",error)
mae=np.mean(np.abs(error))
mse=np.mean(error**2)
print("MAE: ",mae)
print("MSE: ",mse)
print("MAE is easier to interpret as it calculates price in lakh while MSE calculates price in lakh^2")
#step 8
while True:
  house=input("Enter house size in square feet: ")
  try:
    size=float(house)
    print("Predicted house price: ",prediction(size))
  except ValueError:
    print("Invalid Input. Enter a number")
  run=input("\nDo you wish to run again? (y/n): ")
  if run=='n':
    print("Exiting")
    break 
#step 9
#Challenge A
updated_house_prices=np.append(house_prices,1000)
print("Old mean: ",np.mean(house_prices))
print("New mean: ",np.mean(updated_house_prices))
print("Old median: ",np.median(house_prices))
print("New median: ",np.median(updated_house_prices))
print("Unusual values can affect a model as the prediction line tilts to one single value thereby reducing accuracy")
#Challenge B
np.random.seed(18)
array=np.arange(house_sizes.size)
shuffled_array=np.random.permutation(array)
print("Shuffled data: ",shuffled_array)
print("Shuffled sizes: ",house_sizes[shuffled_array])
print("Shuffled prices: ",house_prices[shuffled_array])
#Step 10
print("This model uses the Linear Regression algorithm to find a straight line relation between size and price of a house.We find the slope(w) and the intercept(b) using the training data and then use the formula y=wx+b to find the value on the testing data.")