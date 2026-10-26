# House Price Prediction

A small beginner-friendly machine learning project that predicts house price from house size using simple linear regression, built with only NumPy.

## What the notebook does

1. **Creates the dataset**: 10 houses, with size in square feet and price in thousands of dollars.
2. **Explores the data (EDA)**: prints size, count, shape, indexing, and summary statistics (mean, median, min, max, standard deviation).
3. **Splits the data**: first 8 houses for training, last 2 for testing.
4. **Trains a model**: calculates the slope (`w`) and intercept (`b`) of a straight line using the least-squares formulas directly in NumPy.
5. **Predicts prices**: a `predict_price(size)` function returns `w * size + b`.
6. **Evaluates the model**: computes errors, Mean Absolute Error (MAE) and Mean Squared Error (MSE) on the test set.
7. **Interactive predictor**: asks the user for a house size and prints the estimated price, with error handling for invalid input.
8. **Outlier experiment**: adds an unusual house (3,000 sq ft at $500k) to show that the mean is pulled up while the median barely moves.
9. **Shuffled split**: shuffles size/price pairs together with `np.random.permutation` before splitting.

## Dataset

| Size (sq ft) | 500 | 800 | 1000 | 1200 | 1500 | 1800 | 2000 | 2200 | 2500 | 2800 |
|---|---|---|---|---|---|---|---|---|---|---|
| Price ($k) | 15 | 24 | 30 | 36 | 45 | 54 | 60 | 66 | 75 | 84 |

## Results

- Slope `w` = **0.03** (about $30 per extra square foot)
- Intercept `b` = **0.0**
- Prediction for 1,300 sq ft: **$39.00k**
- Test MAE = **0.00k**, test MSE = **0.00**

The error is zero because the sample data follows an exact straight line (`price = 0.03 x size`). Real-world data is noisier, so expect non-zero errors on real datasets.

## Key takeaways

- Always test on data the model has not seen, otherwise you can't tell whether it generalizes or just memorized.
- The mean is sensitive to outliers; the median is more robust.
- Shuffle data before splitting so the test set isn't just the largest houses.

## Requirements

- Python 3.x
- NumPy
- Jupyter Notebook or JupyterLab

```bash
pip install numpy notebook
jupyter notebook House_price_prediction.ipynb
```

## How to run

Open the notebook and run the cells from top to bottom (**Kernel > Restart & Run All**). Running in order matters, since later cells reuse variables such as `w`, `b`, and `predict_price`.

## Notes and possible improvements

- The "Python expression for predicting price" cell uses `new_size` before it is defined; define it first (for example `new_size = 1300`) or skip that cell.
- A house of 3,000 sq ft / $90k is appended early in the notebook, so the arrays have 11 items after that point.
- Next steps: try scikit-learn's `LinearRegression`, add more features (bedrooms, location), and plot the data with Matplotlib.
