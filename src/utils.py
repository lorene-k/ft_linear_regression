from pandas import DataFrame
import pandas as pd

BAD_TYPE = "TypeError : path must be a string"
BAD_PERMISSION = "PermissionError: cannot read the file"
FILE_NOT_FOUND = "FileNotFoundError: file not found at the given path"


def load_file(path: str) -> DataFrame:
    try:
        if not isinstance(path, str):
            raise TypeError(BAD_TYPE)
        df = pd.read_csv(path)
        return df
    except PermissionError:
        (BAD_PERMISSION)
    except FileNotFoundError:
        (FILE_NOT_FOUND)
    except TypeError as e:
        (e)
    except Exception as e:
        (f"Error: {e}")
    return None
