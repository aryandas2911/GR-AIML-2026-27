# NumPy House Price Prediction

## Objective
The objective of this project is to predict house prices based on house sizes using linear regression.

## Dataset
House size is the feature, measured in square feet. House price is the target, measured in lakhs of rupees.

## Method
I used NumPy to calculate statistical values such as mean, median, maximum, minimum and standard deviation. I calculated the slope and intercept using the training data and used them to predict house prices.

The prediction formula is:

Predicted price = w * house size + b

## Results
The program displays the predicted prices for test data and calculates Mean Absolute Error (MAE) and Mean Squared Error (MSE).

## Challenges Completed
- Challenge A: Added an unusually expensive house and compared the mean and median.
- Challenge B: Shuffled the dataset using a shared permutation, preserving the size-price pairs, and retrained the model.

## Reflection
I learned how to analyse data using NumPy, implement basic linear regression, make predictions and evaluate errors.

## Limitations
1. The dataset is small and may not represent real-world house prices.
2. The model uses only house size and ignores other factors such as location, age and condition.
