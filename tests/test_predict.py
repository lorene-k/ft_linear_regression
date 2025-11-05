import io
import json
from unittest.mock import mock_open, patch
from src.predict import predict, main, GREEN, DEFAULT


def test_predict():
    """Tests the predict function with mock thetas and mileage."""
    dummy_thetas = {"theta0": 0.5, "theta1": 2.0}
    m = 65000
    expected_price = (
        dummy_thetas["theta0"] + (dummy_thetas["theta1"] * (m / 1e5))
    ) * 1e5
    with patch("builtins.open", mock_open(read_data=json.dumps(dummy_thetas))):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            predict(m)
            output = mock_stdout.getvalue()
    assert f"{GREEN}The estimate price is : {expected_price}{DEFAULT}\n" in output


def test_smoke_main():
    with patch("builtins.input", return_value="8000"), patch(
        "src.predict.predict"
    ) as mock_predict:
        main()
        mock_predict.assert_called_once()
