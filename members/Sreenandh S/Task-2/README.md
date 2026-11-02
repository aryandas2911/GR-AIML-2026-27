# House Price Predictor

A simple Python program that uses linear regression to estimate house prices from house size. Sizes are measured in square feet and prices in lakhs.

## Program structure

- **Data:** `load_data()` provides 10 sample house sizes and prices. `explore_data()` reports basic statistics, and `split_data()` uses 8 examples for training and 2 for testing.
- **Model:** `LinearRegression.fit()` learns the slope and intercept for `price = slope × size + intercept`. `predict()` uses them to estimate prices, and `save()` writes the model details to JSON.
- **Evaluation:** `MAE()` and `MSE()` calculate prediction errors. `evaluation()` logs predictions for the test examples and these metrics.
- **Interactive input:** `interface()` prompts for a house size and displays its estimated price. Enter `exit`, `quit`, or `q` to stop.
- **Program flow:** `main()` runs data loading, exploration, training, a sample prediction, evaluation, model saving, and the interactive prompt in order.

## Run

Install NumPy, then run from this folder:

```bash
python price_predictor.py
```

The trained model is saved to `house_price.json`.