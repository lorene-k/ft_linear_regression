RED = "\033[0;31m"
GREEN = "\033[0;32m"
DEFAULT = "\033[0m"

BAD_INPUT_TYPE = f"{RED}Please enter a number.{DEFAULT}"
EXIT = f"{RED}\nProgram exited.\n{DEFAULT}"


def predict(mil):
    print(f"{GREEN}TEST: mil = {mil} {DEFAULT}\n")


def main():
    while True:
        try:
            inp = input("\nPlease enter a mileage: ").strip()
            if inp.isdigit():
                predict(int(inp))
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
