from src.utils import load_file


def test_load_file(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("km,price\n10000,5000\n20000,7000\n")

    df = load_file(str(file))
    assert df is not None
    assert df.shape == (2, 2)
    assert list(df.columns) == ["km", "price"]


def test_load_file_bad_path():
    df = load_file(123)
    assert df is None


def test_load_file_not_found():
    df = load_file("non_existent_file.csv")
    assert df is None


def test_load_file_no_permission(tmp_path):
    file = tmp_path / "data.csv"
    file.write_text("km,price\n10000,5000\n20000,7000\n")
    file.chmod(0o000)

    df = load_file(str(file))
    assert df is None
