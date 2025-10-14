from src.utils import load_file
import os
import json
import numpy as np
import unittest


RED = "\033[0;31m"
GREEN = "\033[0;32m"
DEFAULT = "\033[0m"


class TestLinearRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            dir_path = os.path.dirname(os.path.abspath(__file__))
            base_path = os.path.dirname(dir_path)
            file_path = os.path.join(base_path, "data", "data.csv")
            df = load_file(file_path)
            if df is None:
                print("ERROR")
                return
            file_path = os.path.join(base_path, "data", "thetas.json")
            with open(file_path) as file:
                thetas = json.loads(file.read())
            cls.X = df[["km"]].values.astype(float)
            cls.y = df[["price"]].values.astype(float)
            cls.theta0, cls.theta1 = thetas.values()
            cls.theta0 *= 1e5
        except EOFError:
            print("\nEOF (Ctrl+D).")
        except KeyboardInterrupt:
            print("\nInterrupt (Ctrl+C).")
        except Exception as e:
            print("Error:", e)

    def test_thetas(self):
        """Tests theta values againt np.polyfit coeffs."""
        coeffs = np.polyfit(self.X.flatten(), self.y.flatten(), 1)
        np.testing.assert_almost_equal(self.theta0, coeffs[1], decimal=1)
        np.testing.assert_almost_equal(self.theta1, coeffs[0], decimal=3)


if __name__ == "__main__":
    unittest.main()
