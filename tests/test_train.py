import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch
from src.train import LinearRegression, main
from src.utils import load_file


@pytest.fixture(scope="module")
def data():
    """Fixture that generates dummy data and thetas for testing."""
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([[2], [4], [6], [8], [10]], dtype=float)
    theta0 = 0.0
    theta1 = 2.0
    w = np.array([[theta0], [theta1]])
    m = X.shape[0]
    model = LinearRegression()
    x1 = np.ones((X.shape[0], 1))
    X_b = np.hstack((x1, X))
    return X, y, theta0, theta1, X_b, w, m, model


def test_y_hat(data):
    """Tests predicted values against expected values."""
    _, y, _, _, X_b, w, _, model = data
    yhat = model.y_hat(X_b, w)
    np.testing.assert_allclose(yhat, y, rtol=1e-8, atol=1e-8)


def test_MSE(data):
    """Tests MSE calculation against expected value."""
    _, y, theta0, _, X_b, w, m, model = data
    yhat = model.y_hat(X_b, w)
    mse = model.MSE(yhat, y, m)
    expected = theta0
    assert np.isclose(mse, expected, rtol=1e-8, atol=1e-8)


def test_gradient_descent(data):
    """Tests gradient descent weight update against expected values."""
    _, y, theta0, theta1, X_b, w, _, model = data
    yhat = model.y_hat(X_b, w)
    coefs = model.gradient_descent(w, X_b, y, yhat)
    expected = [[theta0], [theta1]]
    np.testing.assert_allclose(coefs, expected, rtol=1e-8, atol=1e-8)


def test_exec(data):
    """Tests exec method for training against expected thetas."""
    X, y, theta0, theta1, _, _, _, model = data
    expected = np.array([[theta0], [theta1]])
    thetas = model.exec(X, y, {"theta0": 0.0, "theta1": 0.0}, precision=True)
    np.testing.assert_allclose(thetas, expected, rtol=1e-8, atol=1e-8)


def test_smoke_main():
    """Smoke test for main training execution."""
    dummy_df = pd.DataFrame({"km": [75000, 100000], "price": [8000, 10000]})
    dummy_w = np.array([[0.0], [2.0]])
    with patch(
        "src.train.LinearRegression.exec", return_value=dummy_w
    ) as mock_exec, patch(
        "src.train.load_file", return_value=dummy_df
    ) as mock_load_file:
        main()
        mock_exec.assert_called_once()
        mock_load_file.assert_called_once()
