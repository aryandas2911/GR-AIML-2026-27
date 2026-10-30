"""
House Price Prediction program | Task - 2 Geek Room AI/ML 2026-27
"""
import logging
import sys
import json
import numpy as np
from pathlib import Path

fpath= Path("house_price.json")

# made a logger cuz why tf not 
def configure_logging():
    """Configure a logger for structured conclose output"""
    logger = logging.getLogger("HousePricePred")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)-7s | %(message)s",datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = configure_logging()

#custom error domain
class ModelNotFitError(Exception):
    pass
class DataError(Exception):
    pass

#MAE
def MAE(y_true , y_pred):
    """Calculate Mean absolute Error (mae)"""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.shape != y_pred_arr.shape:
        raise DataError("Array shapes dont match")
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))

def MSE(y_true , y_pred):
    """Calculates mean squared error (mse)"""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    if y_true_arr.shape != y_pred_arr.shape:
        raise DataError("Array shapes dont match")
    return float(np.mean( (y_true_arr - y_pred_arr)**2 ) )

class LinearRegression:
    """Linear Regression Estimator y = w*x+b"""
    def __init__(self) -> None:
        self.w : float | None = None #slope
        self.b: float | None = None #intercept
        self.is_fit :bool = False

    def fit(self , x , y):
        """Fits the model parameters from training data using closed form formula
            w = summation((x - x_bar) * (y - y_bar)) / summation((x - x_bar)^2)
            b = y_bar - (w * x_bar)
        """
        x_arr = np.asarray(x , dtype=float).flatten()
        y_arr = np.asarray(y , dtype=float).flatten()

        if len(x_arr) == 0 or len(y_arr) == 0:
            raise DataError("Empty Training arrays recieved")
        if len(x_arr) != len(y_arr):
            raise DataError("Feature and target size not the same")

        x_mean , y_mean =np.mean(x_arr) , np.mean(y_arr) 
        numerator = np.sum( (x_arr - x_mean) *(y_arr - y_mean))
        denominator = np.sum(( x_arr -x_mean) ** 2)

        if denominator == 0:
            raise ZeroDivisionError("Variance is 0")

        self.w = float(numerator/denominator)
        self.b = float(y_mean-(self.w *x_mean))
        self.is_fit = True
        logger.info("Training complete: w (slope) = %.4f,b(intercept)= %.4f", self.w, self.b)
        return self

    def predict(self , x:float | list | int | np.ndarray):
        """Computes prediction for given hosue sizes"""
        if not self.is_fit:
            raise ModelNotFitError("Call fit() before prediction")
        x_arr = np.asarray(x , dtype=float)
        return (self.w * x_arr + self.b) # type: ignore


    def save(self, fpath):
        """saves everything into a json"""
        if not self.is_fit:
            raise ModelNotFitError("Cannot save unfit model")
        saved = {"w":self.w , "b":self.b , "formula": "Predicted Price = w * Size + b","units": {"size": "sq_ft", "price": "lakhs"}}
        with open(fpath,'w') as f:
            json.dump(saved,f)
        logger.info("Model weights saved to '%s'",fpath)

def load_data():
    """Loads size and price of houses in arrays"""
    house_sizes = np.array([        #random prices and sizes taken
        500, 800, 1000, 1200, 1500,
        1800, 2000, 2200, 2500, 2800
    ], dtype=float)

    house_prices = np.array([
        15, 24, 30, 36, 45,
        54, 60, 66, 75, 84
    ], dtype=float)
    return house_sizes , house_prices

def explore_data(sizes,prices): #why we even doin this?
    """Computes and logs summary statistics for the dataset."""
    logger.info("---Explore the data---")
    logger.info("Dataset size: %d samples", len(sizes))
    logger.info(
        "Sizes (sq ft) -> Mean: %.1f | Min: %.1f | Max: %.1f | Median: %.1f | Std: %.2f",
        np.mean(sizes), np.min(sizes), np.max(sizes), np.median(sizes), np.std(sizes)
    )
    logger.info(
        "Prices (lakh) -> Mean: %.1f | Min: %.1f | Max: %.1f | Median: %.1f | Std: %.2f",
        np.mean(prices), np.min(prices), np.max(prices), np.median(prices), np.std(prices)
    )

def split_data(sizes , prices, train_count = 8 ,shuffle = False):
    if shuffle:
        rng = np.random.default_rng(seed = 45)
        indices = rng.permutation(len(sizes))
        sizes = sizes[indices]
        prices = prices[indices]
    x_train , x_test = sizes[:train_count] , sizes[train_count:]
    y_train , y_test = prices[:train_count] , prices[train_count:]

    return x_train, y_train, x_test, y_test

def evaluation(model , x_train , x_test , y_train , y_test):
    y_pred =  model.predict(x_test)
    mae = MAE(y_test , y_pred)
    mse = MSE(y_test , y_pred)

    logger.info("--Evaluation--")

    for size , actual , pred in zip(x_test, y_test , y_pred):
        logger.info(f"Size:{size} | Actual:{actual} | prediction:{pred}")

    logger.info(f"MAE = {mae} , MSE = {mse}")

    # challenge C
    baseline_prediction = np.mean(y_train)
    baseline_preds = np.full_like(y_test, fill_value=baseline_prediction)
    baseline_mae = MAE(y_test, baseline_preds)


def interface(model):
    print("House Price prediction tool")
    print("Enter house size in square feet to predict price")
    print("Enter exit/quit to exit")
    while True:
        try:
            inp = input("Enter house size in sqr feet:").strip()
            if inp.lower() in ('exit','quit','q'):
                print("Bye")
                break

            size = float(inp)
            if size<=0:
                print("Invalid size")
                continue
            prediction = model.predict(size)
            print(f"Predicted price = {prediction:.3f} Lakhs")

        except ValueError:
            print("Invalid input")
        except (KeyboardInterrupt , EOFError):
            print("Unexpected error occured")
            break


def main():
    """main function"""
    try:
        house_sizes , house_prices = load_data()

        explore_data(house_sizes,house_prices)

        x_train, y_train, x_test, y_test = split_data(house_sizes, house_prices, train_count=8, shuffle=False)

        logger.info("--Model Training--")
        model = LinearRegression()
        model.fit(x_train,y_train)

        #predicting 1300 sqft jhouse
        sample_size = 1300.0
        sample_pred = float(model.predict(sample_size))
        logger.info(f"Sample Verification: {sample_size:.3f} sqft size house is predicted at {sample_pred:.3f} Laksh")

        evaluation(model , x_train , x_test , y_train , y_test)

        model.save(fpath)

        interface(model)

        return 0
    except Exception as e:
        logger.exception(f"an error occured {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
