
# Task 1: House Price Prediction Using NumPy

## 1. Objective

The objective of this project is to understand the basic machine learning workflow by using NumPy to analyze house data, implement Linear Regression from scratch, predict house prices, and evaluate the model's performance.

## 2. Dataset

The project uses a sample dataset containing 10 houses.

- **Feature (Input):** House size in square feet.
- **Target (Output):** House price in lakhs of rupees.
- **Dataset:** House sizes range from 500 to 2800 square feet, with corresponding prices from ₹15 lakh to ₹84 lakh.

The dataset follows a simple linear relationship between house size and price.

## 3. Methodology

The following steps were implemented:

1. Imported NumPy and displayed its version.
2. Created NumPy arrays for house sizes and prices.
3. Performed Exploratory Data Analysis (EDA) using mean, minimum, maximum, median, and standard deviation.
4. Investigated the relationship between house size and price.
5. Split the dataset into training and testing sets, using the first eight houses for training and the last two for testing.
6. Implemented Linear Regression using NumPy's mathematical operations.
7. Predicted house prices and evaluated the model using Mean Absolute Error (MAE) and Mean Squared Error (MSE).
8. Added interactive user input with error handling.
9. Completed Challenge A and Challenge C.

## 4. Linear Regression Model

The model uses the equation:

`Predicted Price = w × House Size + b`

Where:

- `w` represents the slope.
- `b` represents the intercept.

The slope and intercept were calculated from the training data using the closed-form Linear Regression formulas.

## 5. Results

- **Slope (w):** 0.03
- **Intercept (b):** 0.0
- **Predicted price for a 1300 sq. ft. house:** ₹39 lakh
- **Test predictions:** ₹75 lakh and ₹84 lakh
- **MAE:** 0.0 lakh
- **MSE:** 0.0 (lakh rupees)²

The model produces zero test error because the sample data follows an exact linear relationship. This result should not be interpreted as evidence of perfect performance on real-world house prices.

## 6. Completed Challenges

### Challenge A: Find Unusual Houses

An unusually high price of ₹500 lakh was added to the dataset. The mean and median were recalculated before and after adding this value.

The mean increases significantly because it is sensitive to extreme values, while the median is generally more resistant to outliers.

### Challenge C: Compare Predictions

A baseline model was created using the average training price as the prediction for every test house. Its MAE was compared with the Linear Regression model's MAE.

The Linear Regression model achieves an MAE of 0.0 lakh on this test set because the data follows a perfect linear relationship.

## 7. Limitations

1. The dataset is small and artificially simple.
2. The model considers only house size and ignores factors such as location, property age, bedrooms, and amenities.
3. The model may produce unreliable predictions outside the dataset's range.
4. Perfect test results on this dataset do not guarantee good performance on real housing data.

## 8. Conclusion

This project demonstrates how NumPy can be used for data analysis, implementing Linear Regression from scratch, making predictions, and evaluating model performance. It also illustrates how outliers affect statistical measures and how a baseline model can be used for comparison.

This project is intended for learning purposes and is not suitable for actual property valuation.

## Technologies Used

- Python
- NumPy
- Jupyter Notebook
