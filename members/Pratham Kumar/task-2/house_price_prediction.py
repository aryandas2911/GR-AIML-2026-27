"""
House Price Prediction Using NumPy
Beginner-friendly machine learning mini-project.

Run:
    python house_price_prediction.py

Requirement:
    pip install numpy
"""

import numpy as np

print("=" * 55)
print("HOUSE PRICE PREDICTION USING NUMPY")
print("=" * 55)

# STEP 1: Check NumPy
print("\nSTEP 1: NumPy setup")
print("NumPy version:", np.__version__)

# STEP 2: Create the dataset
# House sizes are in square feet. Prices are in lakh rupees.
house_sizes = np.array([500, 800, 1000, 1200, 1500,
                        1800, 2000, 2200, 2500, 2800])

house_prices = np.array([15, 24, 30, 36, 45,
                         54, 60, 66, 75, 84])

print("\nSTEP 2: Dataset")
print("House sizes:", house_sizes)
print("House prices (lakhs):", house_prices)
print("Number of houses:", len(house_sizes))
print("Sizes array - size:", house_sizes.size, "shape:", house_sizes.shape)
print("Prices array - size:", house_prices.size, "shape:", house_prices.shape)
print("Third house size:", house_sizes[2], "sq ft")
print("Third house price:", house_prices[2], "lakh")

# STEP 3: Explore the data
print("\nSTEP 3: Explore the data")


def show_statistics(label, values):
    print(f"\n{label}")
    print("Mean:", np.mean(values))
    print("Minimum:", np.min(values))
    print("Maximum:", np.max(values))
    print("Median:", np.median(values))
    print("Standard deviation:", np.std(values))


show_statistics("House sizes (sq ft)", house_sizes)
show_statistics("House prices (lakhs)", house_prices)

# STEP 4: Investigate the relationship
print("\nSTEP 4: Relationship between size and price")
size_difference = house_sizes[1] - house_sizes[0]
price_difference = house_prices[1] - house_prices[0]
print("Difference in size between first two houses:", size_difference, "sq ft")
print("Difference in price between first two houses:", price_difference, "lakh")
print("Price increases by", price_difference, "lakh for", size_difference, "sq ft.")
print("For this dataset, the relationship is: price = 0.03 * size")

# STEP 5: Split into training and testing data
X_train = house_sizes[:8]
y_train = house_prices[:8]
X_test = house_sizes[8:]
y_test = house_prices[8:]

print("\nSTEP 5: Training and testing split")
print("X_train (sizes):", X_train)
print("y_train (prices):", y_train)
print("X_test (sizes):", X_test)
print("y_test (actual prices):", y_test)
print("Training examples:", len(X_train))
print("Testing examples:", len(X_test))

# STEP 6: Train a simple linear regression model using NumPy
# Formula: predicted price = w * size + b
x_mean = np.mean(X_train)
y_mean = np.mean(y_train)

numerator = np.sum((X_train - x_mean) * (y_train - y_mean))
denominator = np.sum((X_train - x_mean) ** 2)

w = numerator / denominator
b = y_mean - w * x_mean


def predict_price(size):
    """Return the predicted house price in lakh rupees."""
    return w * size + b


print("\nSTEP 6: Train the model")
print("Mean training size:", x_mean)
print("Mean training price:", y_mean)
print("Slope numerator:", numerator)
print("Slope denominator:", denominator)
print("Learned slope (w):", w)
print("Learned intercept (b):", b)
print("Predicted price for a 1300 sq ft house:", predict_price(1300), "lakh")

# STEP 7: Evaluate the model on test data
predictions = predict_price(X_test)
errors = y_test - predictions
mae = np.mean(np.abs(errors))
mse = np.mean(errors ** 2)

print("\nSTEP 7: Model evaluation")
for size, actual, predicted in zip(X_test, y_test, predictions):
    print(f"Size: {size} sq ft | Actual: {actual} lakh | Predicted: {predicted:.2f} lakh")
print("Errors (actual - predicted):", errors)
print("Mean Absolute Error (MAE):", mae, "lakh")
print("Mean Squared Error (MSE):", mse, "lakh-squared")
print("MAE is easier to understand because it uses the original price unit (lakhs).")

# STEP 8: Interactive prediction
print("\nSTEP 8: Try your own prediction")
user_entry = input("Enter a house size in square feet (or press Enter to skip): ").strip()
if user_entry:
    try:
        user_size = float(user_entry)
        if user_size <= 0:
            print("Please enter a positive house size.")
        else:
            print(f"Predicted house price: {predict_price(user_size):.2f} lakh rupees")
            print("Reminder: this is only a learning model, not a real property valuation.")
    except ValueError:
        print("Invalid input. Please enter a number, such as 1600.")
else:
    print("Interactive prediction skipped.")

# CHALLENGE A: See how an unusual value affects mean and median
print("\nCHALLENGE A: Unusual house price")
prices_with_outlier = np.append(house_prices, 300)
print("Original price mean:", np.mean(house_prices))
print("Mean after adding a 300-lakh house:", np.mean(prices_with_outlier))
print("Original price median:", np.median(house_prices))
print("Median after adding a 300-lakh house:", np.median(prices_with_outlier))
print("The mean changes more because it uses every value; the median depends on the middle value.")

# CHALLENGE C: Compare with a simple baseline model
baseline_predictions = np.full(len(y_test), np.mean(y_train))
baseline_mae = np.mean(np.abs(y_test - baseline_predictions))

print("\nCHALLENGE C: Compare predictions with a baseline")
print("Baseline prediction for each test house:", baseline_predictions)
print("Baseline test MAE:", baseline_mae, "lakh")
print("Linear regression test MAE:", mae, "lakh")
if mae < baseline_mae:
    print("Linear regression has a lower MAE than the baseline on these test examples.")
elif mae > baseline_mae:
    print("The baseline has a lower MAE than linear regression on these test examples.")
else:
    print("Both methods have the same MAE on these test examples.")

# MINI-REPORT
print("\nMINI-REPORT")
print("Objective: Predict house prices from house size.")
print("Feature: house size in square feet. Target: price in lakh rupees.")
print(f"Method: predicted price = w * size + b, where w={w:.4f} and b={b:.4f}.")
print(f"Test MAE: {mae:.4f} lakh. Test MSE: {mse:.4f} lakh-squared.")
print("Limitations:")
print("1. The dataset is very small and intentionally follows an almost perfect straight line.")
print("2. Real house prices also depend on location, age, rooms, condition, amenities, and market conditions.")
print("This program is for learning only, not for estimating real property prices.")
