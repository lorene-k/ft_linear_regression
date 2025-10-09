from sklearn.linear_model import LinearRegression

def test(km, price):
    """Linear Regression : finds thetas that minimize errors"""
    lr = LinearRegression()
    lr.fit(km, price)
    print("Intercept (theta0):", lr.intercept_[0])
    print("Slope (theta1):", lr.coef_[0][0])
    print("R²:", lr.score(km, price))