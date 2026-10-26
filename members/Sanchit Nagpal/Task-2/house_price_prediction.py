# House price Prediction using Numpy

# step 1
import numpy as np
print("--------Step 1--------")
print("Numpy Version : ",np.__version__)
print("\n")

# step 2
#np.array() convert python list into numpy array
house_sizes = np.array([500,800,1000,1200,1500,             
                        1800,2000,2200,2500,2800])  # sq feet
house_prices = np.array([15,24,30,36,45,54,60,             
                         66,75,84])              # lakh 
print("--------Step 2--------")
print("House sizes : ",house_sizes)
print("House Prices : ",house_prices)
print("Number of Houses : ",len(house_sizes))
print(f"House Size array -> size: {house_sizes.size}\t shape: {house_sizes.shape}")   # shape =(rows,column)  size = no.of elements in entire arry
print(f"House Price array -> size: {house_prices.size}\t shape: {house_prices.shape}")
print(f"Third House -> size : {house_sizes[2]}sq ft\t price : Rs. {house_prices[2]} lakhs")
house_sizes_2 = np.append(house_sizes,1650)
house_prices_2 = np.append(house_prices,49)
print("New House sizes : ",house_sizes_2)
print("New House Prices : ",house_prices_2)
print("\n")

print("--------Step 3--------")
def stats(a):
    print("\tAverage:",np.mean(a))
    print("\tMinimum:",np.min(a))
    print("\tMaximum:",np.max(a))
    print("\tMedian:",np.median(a))
    print(f"\tStandard Deviation: {np.std(a): .2f}\n")
print("For house sizes :")
stats(house_sizes)
print("For house prices :")
stats(house_prices)
# "Average price is 48.9 lakh, so a typical house costs about 49 lakh."

print("--------Step 4--------")
price_diff = house_prices[1] - house_prices[0]   # 24 - 15 = 9
size_diff = house_sizes[1] - house_sizes[0]      # 800 - 500 = 300
w = price_diff / size_diff                   # 9 / 300 = 0.03
print("Price difference:", price_diff)
print("Size difference :", size_diff)
print("Price per extra sq ft:", w)          
# Predicted price = w * size + b,  w = 0.03 and b = 0
size = int(input("Enter house size in sq feet: "))
print(f"Predicted House price: Rs.{w*size} Lakhs\n")

print("--------Step 5--------")
X_train = house_sizes[:8]    #first 8 hoises
Y_train = house_prices[:8]
X_test = house_sizes[8:]     # last 2 houses
Y_test = house_prices[8:]
print("X_train:", X_train)
print("Y_train:", Y_train)
print("X_test :", X_test)
print("Y_test :", Y_test)
print("Training examples:", len(X_train))
print("Test examples:", len(X_test))
#Pairing Check => index of house size = index of prices
for i in range(len(X_test)):
    print(f"Test House {i+1}: {X_test[i]}sq ft -> Rs.{Y_test[i]} lakhs \n")
# Why not test on training data? The model has already seen those
# answers, so the score looks too good.

print("--------Step 6--------")
x_mean= np.mean(X_train)
y_mean= np.mean(Y_train)

num = np.sum((X_train - x_mean)*(Y_train - y_mean))
den = np.sum((X_train - x_mean)**2)
w = num/den
b = y_mean - w * (x_mean)

def predict(size):
    return w*size + b
size = int(input("Enter house size in sq feet: "))
print(f"Predicted price for {size} sq ft = Rs. {predict(size) :.2f}lakhs")

print(f"Predicted price for 1300 sq ft: Rs. {predict(1300)}lakh\n")

print("--------Step 7--------")

y_predict = predict(X_test)
errors = Y_test - y_predict
mae = np.mean(np.abs(errors))
mse = np.mean(errors ** 2)
print("Predictions:", y_predict)
print("Actual     :", Y_test)
print("Errors     :", errors)
print("MAE:", mae, "lakh")
print("MSE:", mse)
# MAE is easier to understand because it is in the same unit (lakh) as price.
# MSE is in "lakh squared", which is hard to imagine.
print("\n")
print("--------Step 8--------")

try:
    user_size = float(input("Enter house size in square feet: "))
    result = predict(user_size)
    print(f"Predicted house price: {result:.2f} lakh")
except ValueError:
    print("Invalid input! Please enter a number")
print("\n")
print("--------Step 9--------")

print("\n--- Challenge A ---")    # My own unusual house: only 1600 sq ft but priced at 250 lakh.

sizes_outlier = np.append(house_sizes, 1600)
prices_outlier = np.append(house_prices, 250)
print("Mean   before:", np.mean(house_prices), "\t after:", np.mean(prices_outlier))
print("Median before:", np.median(house_prices), " \tafter:", np.median(prices_outlier))
 
xm = np.mean(sizes_outlier)
ym = np.mean(prices_outlier)
w_out = np.sum((sizes_outlier - xm) * (prices_outlier - ym)) / np.sum((sizes_outlier - xm) ** 2)
b_out = ym - w_out * xm
print("Slope before:", w, " \tafter outlier:", w_out)
print("1300 sq ft price before:", predict(1300), " \tafter outlier:", w_out * 1300 + b_out)

# After adding the 250 lakh house, the slope changed, so the 1300 sq ft prediction became less correct.
# so the model gives wrong predictions for normal houses.

print("\n--- Challenge C ---")

baseline_pred = np.mean(Y_train)   
baseline_mae = np.mean(np.abs(Y_test - baseline_pred))
print("Baseline MAE        :", baseline_mae)
print("Linear regression MAE:", mae)

# Lower MAE = better. Linear regression should win by a big margin.

