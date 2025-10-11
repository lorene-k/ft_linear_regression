import os
import json
from utils import load_file
import numpy as np
from train import LinearRegression


class Precision:
    """Checks accuracy of trained Linear Regression model."""

    def __init__(self, thetas, df):
        """Initializes class with loaded data."""
        self.regression = LinearRegression()
        self.w = np.array(list(thetas))
        self.theta0, self.theta1 = thetas.values()
        self.X = df[["km"]].values.astype(float) / 1e5
        self.y = df[["price"]].values.astype(float) / 1e5

    def check_R2(self):
        """Computes and prints coefficient of determination (R²)."""
        y_predict = self.theta0 + self.theta1 * self.X
        y_mean = np.mean(self.y)
        SSR = np.sum((self.y - y_predict) ** 2)
        SST = np.sum((self.y - y_mean) ** 2)
        R2 = 1 - SSR / SST
        print(f"R² = {R2}")

    def check_RMSE(self):
        """Computes and prints Root Mean Squared Error (RMSE)."""
        x1 = np.ones((self.X.shape[0], 1))
        input_X = np.hstack((x1, self.X))
        print(input_X.shape)
        yhat = self.regression.y_hat(input_X, self.w)
        RMSE = self.regression.MSE(yhat, self.y)
        RMSE **= 0.5
        print("RMSE =", RMSE)

    def check_precision(self):
        """Computes model accuracy."""
        pass

    def check_precision(self):
        """Evaluate the overall precision of the model."""
        self.check_R2()
        self.check_RMSE()


def main():
    """Checks model precision."""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(dir_path)
    file_path = os.path.join(base_path, "data", "thetas.json")
    with open(file_path) as file:
        thetas = json.loads(file.read())
    df = load_file("data", "data.csv")
    if df is None:
        print("ERROR")
        return
    p = Precision(thetas, df)
    p.check_precision()


if __name__ == "__main__":
    main()


# Leave-One-Out Cross Validation (LOOCV)
