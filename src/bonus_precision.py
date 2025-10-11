import os
import json
from utils import load_file
import numpy as np
from train import LinearRegression


class Precision:
    """Checks accuracy of trained Linear Regression model."""

    def __init__(self, thetas, X, y):
        """Initializes class with loaded data."""
        self.regression = LinearRegression()
        self.theta0, self.theta1 = thetas.values()
        self.w = np.array(list(thetas.values()))
        self.X = X
        self.y = y
        self.m = X.shape[0]

    def get_yhat(self):
        x1 = np.ones((self.m, 1))
        input_X = np.hstack((x1, self.X))
        self.yhat = np.expand_dims(self.regression.y_hat(input_X, self.w), axis=1)

    def check_R2(self):
        """Computes and prints coefficient of determination (R²)."""
        y_predict = self.theta0 + self.theta1 * self.X
        y_mean = np.mean(self.y)
        SSR = np.sum((self.y - y_predict) ** 2)
        SST = np.sum((self.y - y_mean) ** 2)
        R2 = 1 - SSR / SST
        print(f"• R² = {R2}")
        print(
            f"The model explains roughly {(R2 * 100):.2f}% of the variance in car prices based on mileage.\n"
        )

    def check_RMSE(self):
        """Computes and prints Root Mean Squared Error (RMSE)."""
        RMSE = self.regression.MSE(self.yhat, self.y, self.m) ** 0.5
        print("• RMSE =", RMSE)
        print(
            f"Average magnitude of the errors between predicted prices and actual prices = {(RMSE * 1e5):.2f}\n"
        )

    def check_MAE(self):
        """Computes Mean Absolute Error.""" ""
        y = self.y * 1e5
        yhat = self.yhat * 1e5
        MAE = np.sum(abs(y - yhat)) / self.m
        print(f"• MAE = {MAE}")
        print(f"Average absolute difference between predicted car prices and actual prices = {MAE:.2f}$\n")

    def check_MAPE(self):
        """Computes Mean Absolute Percentage Error."""
        MAPE = np.sum(abs(self.y - self.yhat) / self.y) / self.m * 100
        print(f"• MAPE = {MAPE}")
        print(f"Average percentage error relative to actual values = {MAPE:.2f}%\n")

    def check_precision(self):
        """Computes model accuracy using Leave-One-Out Cross-Validation."""
        set_size = len(self.X) - 1
        loss = 0
        for it in range(set_size):
            mask = np.ones(len(self.X), dtype=bool)
            mask[it] = False
            X_training = self.X[mask]
            y_training = self.y[mask]
            X_test = self.X[it : it + 1]
            y_test = self.y[it : it + 1].astype(float)
            thetas = self.regression.exec(X_training, y_training)
            y_pred = thetas[0] + thetas[1] * X_test
            loss += np.sum((y_test - y_pred) ** 2)
        RMSE = np.sqrt(loss / set_size) * 1e5
        mean_price = np.mean(self.y) * 1e5
        relative_error = (RMSE / mean_price) * 100
        print(f"• LOOCV RMSE = {RMSE}:.2f")
        print(
            f"The model's predictions are about {relative_error:.2f}% off from the true price."
        )

    def evaluate(self):
        """Evaluate the overall precision of the model."""
        self.get_yhat()
        print("--- Metrics ---")
        self.check_R2()
        self.check_RMSE()
        self.check_MAE()
        self.check_MAPE()
        print("--- Precision ---")
        self.check_precision()


def main():
    """Checks model precision."""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(dir_path)
    file_path = os.path.join(base_path, "data", "thetas.json")
    with open(file_path) as file:
        thetas = json.loads(file.read())
    file_path = os.path.join(base_path, "data", "data.csv")
    df = load_file(file_path)
    if df is None:
        print("ERROR")
        return
    X = df[["km"]].values.astype(float) / 1e5
    y = df[["price"]].values.astype(float) / 1e5
    p = Precision(thetas, X, y)
    p.evaluate()


if __name__ == "__main__":
    main()
