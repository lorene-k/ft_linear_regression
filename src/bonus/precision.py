from src.utils import load_file
from src.train import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import json
import os

BLUE = "\033[38;5;26m"
CYAN = "\033[1;36m"
MAUVE = "\033[38;5;177m"
DEFAULT = "\033[0m"
ITALIC = "\033[3m"


class Precision:
    """Checks accuracy of trained Linear Regression model."""

    def __init__(self, thetas, X, y):
        """Initializes class with loaded data."""
        self.regression = LinearRegression()
        self.thetas = thetas
        self.theta0, self.theta1 = thetas.values()
        self.w = np.array(list(thetas.values()))
        self.X = X
        self.y = y
        self.m = X.shape[0]

    def get_yhat(self):
        """Computes yhat using Linear Regression method."""
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
        print(f"• R² = {R2:.2f}")
        print(
            f"{ITALIC}Percentage of variance explained by model = {(R2 * 100):.2f}%{DEFAULT}"
        )

    def check_RMSE(self):
        """Computes and prints Root Mean Squared Error (RMSE)."""
        RMSE = self.regression.MSE(self.yhat, self.y, self.m) ** 0.5
        print(f"• RMSE = {RMSE:.2f}")
        print(
            f"{ITALIC}On average, predictions deviate by {(RMSE * 1e5):.2f}${DEFAULT}\n"
        )

    def check_MAE(self):
        """Computes and prints Mean Absolute Error.""" ""
        y = self.y * 1e5
        yhat = self.yhat * 1e5
        MAE = np.sum(abs(y - yhat)) / self.m
        print(f"• MAE = {MAE:.2f}")
        print(f"{ITALIC}On average, predictions are off by {MAE:.2f}${DEFAULT}\n")

    def check_MAPE(self):
        """Computes Mean Absolute Percentage Error."""
        MAPE = np.sum(abs(self.y - self.yhat) / self.y) / self.m * 100
        print(f"• MAPE = {MAPE:.2f}")
        print(f"{ITALIC}On average, predictions are off by {MAPE:.2f}%{DEFAULT}\n")

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
            thetas = self.regression.exec(X_training, y_training, self.thetas, True)
            y_pred = thetas[0] + thetas[1] * X_test
            loss += np.sum((y_test - y_pred) ** 2)
        RMSE = np.sqrt(loss / set_size) * 1e5
        mean_price = np.mean(self.y) * 1e5
        relative_error = (RMSE / mean_price) * 100
        print(f"• LOOCV RMSE = {RMSE:.2f}")
        print(
            f"On average, when tested on unseen data, predictions are off by {relative_error:.2f}%\n"
        )

    def check_residuals(self):
        try:
            residuals = self.y - self.yhat
            mean_residuals = np.sum(residuals) / len(residuals)
            variance = np.sum((residuals - mean_residuals) ** 2) / (len(residuals) - 1)
            print(f"• Residuals Mean = {(mean_residuals * 1e5):.10f}")
            print("Average error of the model's predictions\n")
            print(f"• Variance = {(variance * 1e5):.10f}")
            print(f"Spread of errors around the mean\n")
            plt.scatter(self.X, residuals, label="Residuals")
            plt.title("Residuals")
            plt.xlabel("km")
            plt.ylabel("Residuals")
            plt.axhline(0, color="cyan", linestyle="--")
            if input(f'{MAUVE}Enter "yes" to show residuals graph: {DEFAULT}') == "yes":
                plt.show()
        except EOFError:
            print("\n")
        except KeyboardInterrupt:
            print("\n")
        except Exception as e:
            print("Error:", e)

    def evaluate(self):
        """Evaluates the overall precision of the model."""
        self.get_yhat()
        print(f"\n{CYAN}--- Model Fit ---{DEFAULT}")
        self.check_R2()
        print(f"\n{CYAN}--- Error Metrics ---{DEFAULT}")
        self.check_RMSE()
        self.check_MAE()
        self.check_MAPE()
        print(f"{CYAN}--- Cross Validation ---{DEFAULT}")
        self.check_precision()
        print(f"{CYAN}--- Residual Check ---{DEFAULT}")
        self.check_residuals()


def main():
    """Loads data and checks model precision."""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(dir_path)
    base_path = os.path.dirname(base_path)
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
