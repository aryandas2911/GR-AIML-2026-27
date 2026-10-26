# House Price Prediction Using NumPy

1. Project Overview

This project demonstrates a basic Machine Learning workflow by predicting house prices based on house size using Python and NumPy. It uses a simple linear regression model to learn the relationship between the area of a house and its price.

The project is designed to understand the fundamental steps involved in a Machine Learning project, from exploring data to making predictions and evaluating the model.

2. Problem Statement

House prices can vary depending on several factors. In this beginner-level project, we focus on one factor: house size.

The goal is to use the given examples of house sizes and prices to learn a mathematical relationship and predict the price of a house whose size is provided as input.

3. Dataset

The dataset contains 11 house examples.

-Feature (Input): House size, measured in square feet.
-Target (Output): House price, measured in lakh rupees.

The dataset follows a simple linear relationship in which every additional 100 square feet corresponds to an increase of ₹3 lakh in price.

The data is divided into:
- Training set: 8 houses used to calculate the model parameters.
- Testing set:3 houses used to evaluate the model's predictions.

4. Tools and Technologies

-Python: For implementing the project logic.
- NumPy: For storing data in arrays and performing numerical calculations.
- Google Colab: For writing and running the Python notebook.

5. Methodology

The project follows these steps:

1. Import NumPy and create arrays for house sizes and prices.
2. Explore the dataset using statistical measures such as mean, minimum, maximum, median, and standard deviation.
3. Investigate the relationship between house size and price.
4. Split the dataset into training and testing sets.
5. Calculate the slope (`w`) and intercept (`b`) using the training data.
6. Build a prediction function using simple linear regression.
7. Evaluate predictions using Mean Absolute Error (MAE) and Mean Squared Error (MSE).
8. Make the prediction program interactive by accepting a house size from the user.
9. Investigate unusual values and write a mini-report.

6. Model and Prediction

The model uses the following linear equation:

**Predicted Price = w × House Size + b**

The calculated parameters are:

- Slope (w): 0.03
- Intercept (b): 0.0

Therefore, the prediction equation is:

Predicted Price = 0.03 × House Size

For example, for a house of 1,300 square feet:

Predicted Price = 0.03 × 1300 = ₹39 lakh.

7. Results

The model was evaluated on the three test houses.

| House Size (sq. ft.) | Actual Price (₹ lakh) | Predicted Price (₹ lakh) |
|---:|---:|---:|
| 2500 | 75 | 75 |
| 2800 | 84 | 84 |
| 3000 | 90 | 90 |

The evaluation results were:

- Mean Absolute Error (MAE): 0 lakh
- Mean Squared Error (MSE): 0

The predictions match the actual prices because the sample dataset follows an exact linear relationship. These results do not guarantee similar accuracy on real-world data.

8. Challenges Completed

#Challenge A: Find Unusual Houses

An unusually high price of ₹200 lakh was added to a separate price array to investigate its effect on the mean and median.

- Original mean: approximately ₹52.64 lakh
- New mean: approximately ₹64.92 lakh
- Original median: ₹54 lakh
- New median: ₹57 lakh

This experiment showed that the mean is more sensitive to an unusually high value than the median. The experiment was performed for statistical analysis and did not retrain the prediction model.

#Challenge D: Write a Mini-Report

A mini-report was prepared summarizing the dataset, model parameters, test predictions, evaluation metrics, and limitations.

9. Limitations

1. The model considers only house size and ignores other important factors such as location, age, number of bedrooms, and amenities.
2. The dataset is small and intentionally follows a simple linear relationship, so it does not represent the complexity of real-world property prices.

10. Learning Outcomes

Through this project, I learned how to:

- Store and analyse numerical data using NumPy arrays.
- Calculate descriptive statistics.
- Understand features, targets, training data, and testing data.
- Implement simple linear regression using mathematical formulas.
- Generate predictions and evaluate them using MAE and MSE.
- Create an interactive Python program with input validation.
- Investigate unusual values and document the model's limitations.

11. Conclusion

This project provided a practical introduction to the Machine Learning workflow using Python and NumPy. It demonstrated how a model can learn a relationship from sample data, generate predictions, and be evaluated using error metrics. The project also highlighted why real-world predictions require more representative data and additional features.

**Note:** This is a beginner learning project and is not intended for estimating actual property values.