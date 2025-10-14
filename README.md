# ft_linear_regression

Predict car prices from mileage using a simple linear regression model trained with gradient descent.

## Overview
- **train.py**: trains the model on `data.csv` using gradient descent and saves gradients to `thetas.json`
- **predict.py**: prompts user for mileage and predicts price using gradients from `thetas.json`
- **bonus_render_graph.py**: plots data points and regression line into graph
- **bonus_precision.py**: calculates model precision

## Usage
1. Setup venv & packages
    ```bash
    git clone https://github.com/lorene-k/ft_linear_regression.git
    cd ft_linear_regression
    chmod +x run.sh
    ```
2. Train model
    ```bash
        ./run.sh train
    ```
3. Predict price
    ```bash
        ./run.sh predict
    ```
4.  Evaluate model and visualize data
    ```bash
        ./run.sh precision
        ./run.sh graph
    ```

5.  Run unit tests
    ```bash
        ./run.sh test
    ```