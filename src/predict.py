import json
import os

RED = "\033[0;31m"
GREEN = "\033[0;32m"
DEFAULT = "\033[0m"

BAD_INPUT_TYPE = f"{RED}Please enter a number.{DEFAULT}"
EXIT = f"{RED}\nProgram exited.\n{DEFAULT}"


def predict(mil: int):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(dir_path)
    file_path = os.path.join(base_path, "data", "thetas.json")
    with open(file_path) as file:
        thetas = json.loads(file.read())
    theta0, theta1 = thetas.values()
    mil_scaled = mil / 1e5
    price_scaled = theta0 + (theta1 * mil_scaled)
    price = price_scaled * 1e5
    print(f"{GREEN}The estimate price is : {price}{DEFAULT}\n")


def main():
    while True:
        try:
            inp = input("\nPlease enter a mileage: ").strip()
            if inp.isdigit():
                predict(float(inp))
                return
            else:
                print(BAD_INPUT_TYPE)
        except (EOFError, KeyboardInterrupt):
            print(EXIT)
            return
        except Exception as e:
            print("Error: ", e)
            return


if __name__ == "__main__":
    main()
