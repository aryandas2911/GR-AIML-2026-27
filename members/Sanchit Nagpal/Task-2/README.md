# House Price Prediction using NumPy (Task 2)

**Goal:** predict the price of a house (in lakhs of rupees) from its size (in square feet).

**How it works:**
1. House sizes and prices are stored in NumPy arrays and explored with mean, median, min, max and standard deviation.
2. The data is split: the first 8 houses train the model and the last 2 test it.
3. Simple linear regression finds the best straight line, `price = w x size + b`. Here w = 0.03 and b = 0, so every extra square foot adds 0.03 lakh.
4. The model is checked with MAE and MSE (both 0.0 on the test houses), and a user can type a house size to get a predicted price.

**Challenges done:** an outlier test (one unusual house changes the line) and a baseline comparison (the model's MAE is 0.0, against 38.25 for always guessing the average).
