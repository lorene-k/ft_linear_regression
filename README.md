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
    source setup.py
    ```
2. Train model
    ```bash
        src/train.py
    ```
3. Predict price
    ```bash
        src/predict.py
    ```
4.  Evaluate model and visualize data
    ```bash
        src/bonus_precision.py
        src/bonus_render_graph.py
    ```