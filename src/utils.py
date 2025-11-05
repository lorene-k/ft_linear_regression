from pandas import DataFrame
import pandas as pd

BAD_TYPE = "TypeError : path must be a string"
BAD_PERMISSION = "PermissionError: cannot read the file"
FILE_NOT_FOUND = "FileNotFoundError: file not found at the given path"


def load_file(path: str) -> DataFrame:
    """Loads csv file and returns DataFrame."""
    try:
        if not isinstance(path, str):
            raise TypeError(BAD_TYPE)
        df = pd.read_csv(path)
        return df
    except PermissionError:
        print(BAD_PERMISSION)
    except FileNotFoundError:
        print(FILE_NOT_FOUND)
    except TypeError as e:
        print(e)
    except Exception as e:
        print(f"Error: {e}")
    return None
