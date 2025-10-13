from src.utils import load_file
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import json
import os


def ft_plot(df: DataFrame, line_data, mil):
    """Plots the original data points and the regression line."""
    km = df[["km"]]
    price = df[["price"]]
    plt.scatter(km, price, color="red", label="Extracted data")
    plt.plot(mil, line_data, label="Predictions")


def ft_render():
    """Renders graph."""
    plt.title("Car price according to mileage")
    plt.legend(loc="upper right")
    plt.xlabel("Km")
    plt.ylabel("Price")
    plt.show()


def get_line_data(file_path, df):
    """Computes the predicted regression line using saved theta values."""
    with open(file_path) as file:
        thetas = json.loads(file.read())
    theta0, theta1 = thetas.values()
    max_mil = df[["km"]].max().item()
    mil = np.linspace(0, max_mil, 100)
    mil_scaled = mil / 1e5
    price_scaled = theta0 + (theta1 * mil_scaled)
    price = price_scaled * 1e5
    return price, mil


def main():
    """Loads dataset and predicted theta values, then renders the regression graph."""
    try:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.dirname(dir_path)
        file_path = os.path.join(base_path, "data", "data.csv")
        df = load_file(file_path)
        if df is None:
            ("ERROR")
            return
        file_path = os.path.join(base_path, "data", "thetas.json")
        line_data, mil = get_line_data(file_path, df)
        ft_plot(df, line_data, mil)
        ft_render()
    except EOFError:
        print("\nEOF (Ctrl+D).")
    except KeyboardInterrupt:
        print("\nInterrupt (Ctrl+C).")
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
