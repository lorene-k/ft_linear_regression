from utils import load_file
from pandas import DataFrame
import matplotlib.pyplot as plt
import os


def ft_plot(df: DataFrame):
    """Plots points for graph."""
    km = df[["km"]]
    price = df[["price"]]
    plt.scatter(km, price, color="red", label="Extracted data")


def ft_render():
    """Renders graph."""
    plt.title("Car price according to mileage")
    plt.legend(loc="upper right")
    plt.xlabel("Km")
    plt.ylabel("Price")
    plt.show()


def main():
    """Extracts data and renders graph."""
    try:
        dir_path = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.dirname(dir_path)
        file_path = os.path.join(base_path, "data", "data.csv")
        df = load_file(file_path)
        if df is None:
            ("ERROR")
            return
        ft_plot(df)
        ft_render()
    except EOFError:
        print("\nEOF (Ctrl+D).")
    except KeyboardInterrupt:
        print("\nInterrupt (Ctrl+C).")
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()
