import numpy as np
import json
from utils import load_file
import os


class LinearRegression:
    """Linear regression class with:
    • m = training examples (nb of pts)
    • n = number of features (intercept 1 & mileage)
    • y = prices ; shape (1, m)
    • X = input matrix (intercept + mileage) ; shape (m, n)
    • w = thetas : shape (n, 1)
    """

    def __init__(self):
        """Initializes model with default parameters."""
        self.learning_rate = 0.01
        self.total_iterations = 10000

    def y_hat(self, X, w):
        """
        Computes predictions for input features using current weights.

        Args:
            X (np.ndarray): input matrix
            w (np.ndarray): weight vector

        Returns:
            np.ndarray: Predicted values of shape (m, 1)
        """
        return np.dot(X, w)

    def MSE(self, yhat, y, m):
        """
        Computes Mean Squared Error (MSE) between predictions and true values.

        Args:
            yhat (np.ndarray): predicted values (m, 1)
            y (np.ndarray): true target values (m, 1)

        Returns:
            float: Mean Squared Error
        """
        loss = 1 / m * np.sum(np.power(y - yhat, 2))
        return loss

    def gradient_descent(self, w, X, y, yhat):
        """
        Performs one step of gradient descent to update weights.

        Args:
            w (np.ndarray): current weight vector
            X (np.ndarray): input matrix
            y (np.ndarray): true target values
            yhat (np.ndarray): predicted values

        Returns:
            np.ndarray: updated weight vector
        """
        dLdw = X.T @ (yhat - y)
        w = w - self.learning_rate * dLdw
        return w

    def exec(self, X, y, precision=False):
        """Trains linear regression model using gradient descent.

        Args:
            X (np.ndarray): input features (km)
            y (np.ndarray): target values (price)

        Returns:
            np.ndarray: final learned weight vector (thetas)
        """
        x1 = np.ones((X.shape[0], 1))
        X = np.hstack((x1, X))
        m = X.shape[0]
        n = X.shape[1]
        w = np.zeros((n, 1))
        prev_loss = 0
        for it in range(self.total_iterations + 1):
            yhat = self.y_hat(X, w)
            loss = self.MSE(yhat, y, m)
            if precision == False:
                if it % 50 == 0:
                    print(f"Cost at iteration {it} is {loss.item()}.")
                if abs(prev_loss - loss.item()) < 1e-21:
                    print(
                        f"\nConverged after {it} iterations with loss {loss.item()}\n"
                    )
                    break
            prev_loss = loss.item()
            w = self.gradient_descent(w, X, y, yhat)
        return w


def main():
    """
    Loads and scales data, trains Linear Regression model,
    and save learned theta values to a JSON file.
    """
    dir_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(dir_path)
    file_path = os.path.join(base_path, "data", "data.csv")
    df = load_file(file_path)
    if df is None:
        ("ERROR")
        return
    X = df[["km"]].values.astype(float) / 1e5
    y = df[["price"]].values.astype(float) / 1e5
    regression = LinearRegression()
    w = regression.exec(X, y)
    thetas = {"theta0": w[0].item(), "theta1": w[1].item()}
    file_path = os.path.join(base_path, "data", "thetas.json")
    with open(file_path, "w") as file:
        json.dump(thetas, file)


if __name__ == "__main__":
    main()
