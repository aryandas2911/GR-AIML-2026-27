"""
House Price Prediction Using NumPy
Completed beginner-friendly assignment.

Install NumPy if needed:
    pip install numpy

Dataset units:
- House sizes: square feet
- House prices: lakhs of rupees

This is a toy learning example, not a real property valuation model.
"""

import numpy as np

def main():
    # Step 1: Set up
    print("=" * 65)
    print("HOUSE PRICE PREDICTION USING NUMPY")
    print("=" * 65)
    print(f"NumPy version: {np.__version__}")

    # Step 2: Create the dataset
    house_sizes = np.array(
        [500, 800, 1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800],
        dtype=float,
    )
    house_prices = np.array(
        [15, 24, 30, 36, 45, 54, 60, 66, 75, 84],
        dtype=float,
    )

    print("\nSTEP 2 — DATASET")
    print("House sizes (sq ft):", house_sizes)
    print("House prices (lakhs):", house_prices)
    print("Number of houses:", len(house_sizes))
    print("Sizes array size/shape:", house_sizes.size, house_sizes.shape)
    print("Prices array size/shape:", house_prices.size, house_prices.shape)
    print(
        "Third house:",
        f"{house_sizes[2]:.0f} sq ft, price = {house_prices[2]:.2f} lakh",
    )

    # Demonstrate adding a matching house without changing the prescribed
    # original 10-house train/test split used below.
    extended_sizes = np.append(house_sizes, 3000)
    extended_prices = np.append(house_prices, 90)
    print(
        "Added-house example:",
        f"{extended_sizes[-1]:.0f} sq ft, price = {extended_prices[-1]:.2f} lakh",
    )

    # Step 3: Explore the data
    print("\nSTEP 3 — EXPLORATORY DATA ANALYSIS")

    def print_statistics(label, values, unit):
        print(f"\n{label} ({unit})")
        print(f"  Mean:   {np.mean(values):.2f}")
        print(f"  Min:    {np.min(values):.2f}")
        print(f"  Max:    {np.max(values):.2f}")
        print(f"  Median: {np.median(values):.2f}")
        print(f"  Std:    {np.std(values):.2f}")
        print("  Interpretation: mean is the average; min/max show the range;")
        print("  median is the middle value; standard deviation describes spread.")

    print_statistics("House sizes", house_sizes, "square feet")
    print_statistics("House prices", house_prices, "lakhs of rupees")
    print(
        "\nMean vs median: an unusually high value can pull the mean upward, "
        "while the median is usually less affected."
    )

    # Step 4: Investigate the relationship
    print("\nSTEP 4 — RELATIONSHIP BETWEEN SIZE AND PRICE")
    i, j = 1, 4  # Compare the second and fifth houses.
    size_difference = house_sizes[j] - house_sizes[i]
    price_difference = house_prices[j] - house_prices[i]
    print(f"Compared sizes: {house_sizes[i]:.0f} and {house_sizes[j]:.0f} sq ft")
    print(f"Size difference: {size_difference:.0f} sq ft")
    print(f"Price difference: {price_difference:.2f} lakh")
    print(f"Price change per square foot: {price_difference / size_difference:.4f} lakh")
    print("Observed relationship: approximately 3 lakh per additional 100 sq ft.")
    print("Example expression for this dataset: predicted_price = 0.03 * size")

    # Step 5: Train/test split as specified in the assignment
    X_train = house_sizes[:8]
    y_train = house_prices[:8]
    X_test = house_sizes[8:]
    y_test = house_prices[8:]

    print("\nSTEP 5 — TRAIN/TEST SPLIT")
    print("X_train:", X_train)
    print("y_train:", y_train)
    print("X_test: ", X_test)
    print("y_test: ", y_test)
    print("Training examples:", len(X_train))
    print("Testing examples:", len(X_test))
    print(
        "Each size stays paired with the price at the same index. Testing on "
        "training data alone can look overly optimistic because the model has "
        "already seen those examples."
    )

    # Step 6: Fit simple linear regression using the closed-form formula
    print("\nSTEP 6 — TRAIN LINEAR REGRESSION")
    x_mean = np.mean(X_train)
    y_mean = np.mean(y_train)
    numerator = np.sum((X_train - x_mean) * (y_train - y_mean))
    denominator = np.sum((X_train - x_mean) ** 2)

    if denominator == 0:
        raise ValueError("Cannot fit the model: all training sizes are identical.")

    w = numerator / denominator
    b = y_mean - w * x_mean

    print(f"Mean of X_train: {x_mean:.4f}")
    print(f"Mean of y_train: {y_mean:.4f}")
    print(f"Slope numerator: {numerator:.4f}")
    print(f"Slope denominator: {denominator:.4f}")
    print(f"Learned slope (w): {w:.6f}")
    print(f"Learned intercept (b): {b:.6f}")
    print(f"Model equation: predicted_price = {w:.6f} * size + {b:.6f}")

    def predict_price(size):
        """Return the predicted house price in lakhs for a size in sq ft."""
        return w * size + b

    print(f"Predicted price for a 1,300 sq ft house: {predict_price(1300):.2f} lakh")

    # Step 7: Evaluate on held-out test data
    print("\nSTEP 7 — MODEL EVALUATION")
    y_pred = predict_price(X_test)
    errors = y_test - y_pred
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)

    print("Actual test prices:    ", y_test)
    print("Predicted test prices: ", np.round(y_pred, 4))
    print("Errors (actual - predicted):", np.round(errors, 4))
    print(f"Mean Absolute Error (MAE): {mae:.6f} lakh")
    print(f"Mean Squared Error (MSE):  {mse:.6f} lakh-squared")
    print(
        "MAE is easier to interpret because it is measured in lakhs, the same "
        "unit as the target price. MSE uses squared units and penalizes larger errors."
    )

    # Challenge B: Shuffle rows using one shared permutation so pairs remain aligned.
    print("\nCHALLENGE B — SHUFFLE THE DATASET")
    rng = np.random.default_rng(42)  # Fixed seed makes the result reproducible.
    permutation = rng.permutation(len(house_sizes))
    shuffled_sizes = house_sizes[permutation]
    shuffled_prices = house_prices[permutation]
    print("Shared permutation:", permutation)
    print("Shuffled sizes: ", shuffled_sizes)
    print("Shuffled prices:", shuffled_prices)
    print(
        "The same permutation is applied to both arrays, so every size remains "
        "matched with its original price."
    )

    # Challenge C: Compare the linear model with a mean-price baseline.
    print("\nCHALLENGE C — COMPARE WITH A BASELINE")
    baseline_predictions = np.full(len(y_test), np.mean(y_train))
    baseline_mae = np.mean(np.abs(y_test - baseline_predictions))
    print("Baseline predictions (training mean):", baseline_predictions)
    print(f"Baseline test MAE: {baseline_mae:.6f} lakh")
    print(f"Linear regression test MAE: {mae:.6f} lakh")
    if mae < baseline_mae:
        print("Result: linear regression has lower test MAE than the baseline.")
    elif mae > baseline_mae:
        print("Result: the mean-price baseline has lower test MAE on this split.")
    else:
        print("Result: both methods have the same test MAE on this split.")

    # Step 8: Interactive prediction
    print("\nSTEP 8 — INTERACTIVE PREDICTION")
    user_input = input("Enter house size in square feet (or press Enter to skip): ").strip()
    if user_input:
        try:
            size = float(user_input)
            if not np.isfinite(size) or size <= 0:
                print("Please enter a finite positive number for house size.")
            else:
                print(f"Predicted house price: {predict_price(size):.2f} lakh")
                if size < np.min(X_train) or size > np.max(X_train):
                    print(
                        "Note: this size is outside the training-size range; "
                        "the prediction is an extrapolation and may be unreliable."
                    )
        except ValueError:
            print("Invalid input. Please enter a number, such as 1600.")

    # Step 10 / mini-report
    print("\nMINI-REPORT")
    print("Objective: Predict house price from house size.")
    print("Feature: house size in square feet; target: price in lakhs of rupees.")
    print(f"Method: closed-form simple linear regression; slope={w:.6f}, intercept={b:.6f}.")
    print(f"Test MAE={mae:.6f} lakh; test MSE={mse:.6f} lakh-squared.")
    print("Limitations:")
    print("  1. The dataset is tiny, clean, and almost perfectly linear.")
    print("  2. Real prices also depend on location, age, bedrooms, amenities, and market conditions.")
    print("  3. Only two examples are in the test set, so evaluation is not reliable for real-world use.")
    print("\nProject complete.")


if __name__ == "__main__":
    main()
