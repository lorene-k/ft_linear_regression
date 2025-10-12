from utils import load_file
import os
import json
import numpy as np


RED = "\033[0;31m"
GREEN = "\033[0;32m"
DEFAULT = "\033[0m"


def test_coeffs(file_path, thetas, df):
    """Tests theta values againt np.polyfit coeffs."""
    theta0_scaled, theta1_scaled = thetas.values()
    theta0 = theta0_scaled * 1e5
    theta1 = theta1_scaled
    X = df[["km"]].values.astype(float)
    y = df[["price"]].values.astype(float)
    coeffs = np.polyfit(X.flatten(), y.flatten(), 1)
    print("\n--- THETAS TEST ---")
    print(f"• np.polyfit : theta0 = {coeffs[1]:e} ; theta1 = {coeffs[0]:e} ")
    print(f"• my coeffs : theta0 = {theta0:e} ; my_theta1 = {theta1:e}\n")


def main():
    """Extracts data and tests predictions."""
    try:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.dirname(dir_path)
        file_path = os.path.join(base_path, "data", "data.csv")
        df = load_file(file_path)
        if df is None:
            ("ERROR")
            return
        file_path = os.path.join(base_path, "data", "thetas.json")
        with open(file_path) as file:
            thetas = json.loads(file.read())
        test_coeffs(file_path, thetas, df)
    except EOFError:
        print("\nEOF (Ctrl+D).")
    except KeyboardInterrupt:
        print("\nInterrupt (Ctrl+C).")
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
