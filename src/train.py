import json
from utils import load_file
from pandas import DataFrame
import numpy as np
import os

RED = "\033[0;31m"
GREEN = "\033[0;32m"
DEFAULT = "\033[0m"


def ft_linear_regression(df: DataFrame):
    """Implements Normal Equation : B=(XTX)−1XTy"""
    price = df[["price"]].copy()  # Y
    km = df[["km"]].copy()  # X
    km["intercept"] = 1
    km = km[["intercept", "km"]]
    km_T = km.T  # transpose
    B = (
        np.linalg.inv(km_T @ km) @ km_T @ price
    )  # multiply matrix by its transpose and take the inverse of the result
    B.index = km.columns
    predictions = km @ B
    SSR = ((price - predictions) ** 2).sum()
    SST = ((price - price.mean()) ** 2).sum()
    R2 = 1 - (SSR / SST)
    print(R2)


def main():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(dir_path)
    file_path = os.path.join(base_path, "data", "data.csv")
    df = load_file(file_path)
    if df is None:
        ("ERROR")
        return
    ft_linear_regression(df)
    thetas = {"theta0": 0, "theta1": 1}
    file_path = os.path.join(base_path, "data", "thetas.json")
    with open(file_path, "w") as file:
        json.dump(thetas, file)


if __name__ == "__main__":
    main()


# Collect data - OK
# Divide the data into training data (about 80%) and testing data (about 20%)
# Feed the training data (inputs and their labels) to Linear Regression model
# Evaluate the model using testing data it has never seen before
# The model predicts outputs and these predictions are compared with the actual labels
# to calculate accuracy or error
