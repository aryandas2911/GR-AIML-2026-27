
import numpy as np

# STEP 1: NumPy version
print("NumPy Version:", np.__version__)


# STEP 2: Create dataset
house_sizes = np.array([
    500, 800, 1000, 1200, 1500,
    1800, 2000, 2200, 2500, 2800
])

house_prices = np.array([
    15, 24, 30, 36, 45,
    54, 60, 66, 75, 84
])

print("\nhouse sizes:", house_sizes)
print("house prices:", house_prices)

print("number of houses:", house_sizes.size)

print("size of house_sizes:", house_sizes.size)
print("size of house_prices:", house_prices.size)

print("Sizes shape:", house_sizes.shape)
print("Prices shape:", house_prices.shape)

print("Third house size:", house_sizes[2])
print("Third house price:", house_prices[2])

new_sizes = np.append(house_sizes, 3000)
new_prices = np.append(house_prices, 90)

print("Updated sizes:", new_sizes)
print("Updated prices:", new_prices)


# STEP 3: Exploratory Data Analysis
print("\nHOUSE SIZE STATISTICS")
print("Mean:", np.mean(house_sizes)) #gives avg value
print("Minimum:", np.min(house_sizes)) #gives min value
print("Maximum:", np.max(house_sizes)) #gives max value
print("Median:", np.median(house_sizes)) #gives median value
print("Standard Deviation:", np.std(house_sizes)) #tells about the spread

print("\nHOUSE PRICE STATISTICS")
print("Mean:", np.mean(house_prices))
print("Minimum:", np.min(house_prices))
print("Maximum:", np.max(house_prices))
print("Median:", np.median(house_prices))
print("Standard Deviation:", np.std(house_prices))


# STEP 4: Investigate relationship
size_diff = house_sizes[1] - house_sizes[0]
price_diff = house_prices[1] - house_prices[0]

print("\nSize Difference:", size_diff)
print("Price Difference:", price_diff)

rate = price_diff / size_diff
print("Price per square foot:", rate)

print("Estimated price for 1300 sqft:", rate * 1300)

# STEP 5: Train-Test Split
X_train = house_sizes[:8]
y_train = house_prices[:8]

X_test = house_sizes[8:]
y_test = house_prices[8:]

print("\nTRAINING DATA ")
print("X_train:", X_train)
print("y_train:", y_train)

print("\nTESTING DATA")
print("X_test:", X_test)
print("y_test:", y_test)

print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))

'''
Testing only on training data can give a misleading impression because the model has already learned 
from that data. It may show low errors on training data but perform poorly on new, unseen data.
Therefore, we use a separate test dataset to evaluate the model’s actual performance and check 
how well it predicts house prices for unseen houses
'''


# STEP 6: Linear Regression using NumPy
x_mean = np.mean(X_train)
y_mean = np.mean(y_train)

numerator = np.sum(
    (X_train - x_mean) * (y_train - y_mean)
)

denominator = np.sum(
    (X_train - x_mean) ** 2
)

w = numerator / denominator
b = y_mean - w * x_mean

print("\nMODEL PARAMETERS")
print("Slope -> w :", w)
print("Intercept -> b :", b)


# Prediction function
def predict_price(size):
    return w * size + b


prediction_1300 = predict_price(1300)
print("Predicted price for 1300 sqft:",
      round(prediction_1300, 2), "lakhs")


# STEP 7: Model Evaluation
y_pred = predict_price(X_test)

errors = y_test - y_pred

mae = np.mean(np.abs(errors))
mse = np.mean(errors ** 2)

print("\nMODEL EVALUATION")
print("Actual prices:", y_test)
print("Predicted prices:", y_pred)
print("Prediction errors:", errors)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
'''
MAE (Mean Absolute Error) is easier to interpret because it is measured in the original unit of house
prices, which is lakhs of rupees. It tells us the average absolute difference between actual and
predicted prices. MSE squares the errors, so its unit is squared lakhs, making it harder to understand
'''

# STEP 8: User Input
print("\nHOUSE PRICE PREDICTOR")

try:
    size = float(input("Enter house size in sqft: "))

    if size <= 0:
        print("House size must be positive.")
    else:
        predicted_price = predict_price(size)
        print("Predicted House Price:",
              round(predicted_price, 2), "lakhs")

except ValueError:
    print("Invalid input! Please enter a number.")


# STEP 9: BONUS CHALLENGES
# Challenge A: Unusual house price

unusual_prices = np.append(house_prices, 500)

print("\nUnusual house price")
print("Original Mean:", np.mean(house_prices))
print("New Mean:", np.mean(unusual_prices))

print("Original Median:", np.median(house_prices))
print("New Median:", np.median(unusual_prices))


# Challenge C: Compare with baseline model
baseline_prediction = np.full(
    y_test.shape, np.mean(y_train)
)

baseline_mae = np.mean(
    np.abs(y_test - baseline_prediction)
)

print("\nMODEL COMPARISON ")
print("Baseline MAE:", baseline_mae)
print("Linear Regression MAE:", mae)
