# House Price Prediction with NumPy

A beginner machine-learning project that predicts the sale price of a house from its features. A multiple linear regression model is fitted with NumPy (least squares) and evaluated on houses it has never seen. A second model, which describes location with a zip-code price rate, improves on the first.

> This is a learning project. It is not intended for real property valuation.

---

## Results

All models are compared on the same 909 held-out test houses.

| Model | Columns | Test MAE ($) | Better than baseline |
|---|---|---|---|
| Baseline (always predict the average training price) | 0 | 235,839 | – |
| Model 1: city columns | 29 | 126,934 | 46.2% |
| **Model 2: zip-code rate (final)** | **11** | **110,294** | **53.2%** |

The final model has a test MSE of about 40.05 billion (squared dollars) and a test RMSE of about $200,132. Its predictions are, on average, about $110,000 away from the actual price.

---

## Dataset

- **Source:** Kaggle house price data (Seattle area), 4,600 records and 18 columns, sold between May and July 2014.
- **Target:** `price` (US dollars).
- **Required file:** `Datasets/data.csv` must be placed next to the notebook.

### Cleaning

| Step | Records removed |
|---|---|
| Zero prices | 49 |
| Zero bedrooms or bathrooms | 2 |
| Price-per-square-foot outliers (below $50 or above $1,000) | 8 |
| **Records remaining** | **4,541** |

### Features used (final model, 11 columns)

`sqft_living`, `bathrooms`, `bedrooms`, `floors`, `waterfront`, `view`, `condition`, `sqft_lot`, `yr_built`, `renovated`, and `zip_rate`.

- `renovated` is a 0/1 flag created from `yr_renovated`.
- `zip_rate` is the average price per square foot of a house's zip code, calculated from the training houses only.

### Columns excluded

| Column(s) | Reason |
|---|---|
| `country` | Same value for every record |
| `street` | Nearly unique for every record |
| `sqft_above`, `sqft_basement` | Add up exactly to `sqft_living` (redundant) |
| `date` | All sales fall within about 10 weeks |
| `yr_renovated` | Mixes the code 0 with calendar years; replaced by `renovated` |
| `city`, `statezip` | Text columns; replaced by city dummies (Model 1) or the zip-code rate (Model 2) |

---

## Method

1. **Explore** the data with summary statistics, histograms and correlations.
2. **Clean** invalid records and choose features.
3. **Convert** the features and target to NumPy arrays (`X` and `y`).
4. **Split** the data 80% / 20% into training and test sets, after shuffling `X` and `y` together with one shared permutation (seed 42).
5. **Fit** the model by least squares with `np.linalg.lstsq`. A column of ones is added so the intercept is learned like any other weight.
6. **Evaluate** on the test set with MAE, MSE and RMSE, and compare with the baseline.
7. **Improve** the model by replacing 19 city columns with one zip-code rate column.
8. **Predict** interactively: the user enters a house's details and gets a price.

The prediction is a weighted sum:

```
price = w1·x1 + w2·x2 + ... + wn·xn + b
```

The weights are chosen so that the total squared error on the training houses is as small as possible.

### Metrics

- **MAE** (mean absolute error): the average size of a prediction error, in dollars.
- **MSE** (mean squared error): the average squared error, in squared dollars. It penalises large errors more.
- **RMSE:** the square root of MSE, which returns it to dollars.

---

## Interactive prediction

The last part of the notebook asks for the house's living area, bathrooms, bedrooms, floors, waterfront, view rating, condition rating, lot size, year built, renovation status and zip code. Pressing Enter on any question uses the typical (median) value from the training data.

- Text that is not a number is rejected and the question is asked again.
- A zip code that is not in the data is rejected and asked again.
- A warning is shown if the living area is outside the range the model was trained on.

---

## Challenges completed

| Challenge | Summary |
|---|---|
| **A: Unusual house** | Adding one $50,000,000 house raised the mean price by about $10,900 and left the median unchanged. It also made the final model's test MAE about 2% worse (110,294 to 112,284). |
| **B: Shuffle the dataset** | `X` and `y` are shuffled together with one shared permutation, so each house stays paired with its price. |
| **C: Baseline comparison** | Every model is compared with the average-price baseline. The final model's error is about 53% lower. |
| **D: Mini-report** | Summary, results, limitations and reflection are in Section 15 of the notebook. |

---

## Limitations

- The typical error (about $110,000) is large. The model does not use finish quality, the street or the exact location.
- The model is linear, so it assumes each feature adds a fixed amount to the price.
- One set of weights is used for the whole region, so effects that differ between areas are not captured.
- The zip-code rate of a training house includes that house's own price, so the training fit is slightly optimistic. Test results are not affected.
- The data covers about 10 weeks in 2014 in one region, so it does not describe current prices or other places.
- The outlier thresholds were chosen by inspecting the data and are a judgement.

---

## How to run

### Requirements

- Python 3.13 or later
- Jupyter Notebook, JupyterLab or Google Colab
- `numpy`, `pandas` and `matplotlib` (pandas uses matplotlib for the plots)

The notebook was developed with NumPy 2.5.4 and pandas 3.1.0.

### Steps

```bash
pip install numpy pandas matplotlib jupyter
```

1. Place the dataset at `Datasets/data.csv`.
2. Open the notebook and choose **Kernel → Restart & Run All**.
3. The interactive cells in Section 13 wait for typed input when run. Press Enter to accept typical values, or type your own.

### Project structure

```
.
├── solution.ipynb   # the full analysis and model (rename to match your file)
├── Datasets/
│   └── data.csv                   # Kaggle house price data
└── README.md
```

---

## Notebook outline

| Section | Content |
|---|---|
| 1-2 | Environment setup and loading the data |
| 3 | Exploratory data analysis |
| 4 | Data cleaning |
| 5-6 | Feature analysis and the final dataset |
| 7 | NumPy arrays, statistics and the size-price relationship |
| 8 | Train/test split |
| 9-10 | Model fitting and evaluation (MAE, MSE, RMSE) |
| 11 | Baseline comparison |
| 12 | Model improvement with the zip-code rate |
| 13 | Interactive prediction |
| 14 | Challenges |
| 15 | Summary, limitations and reflection |
