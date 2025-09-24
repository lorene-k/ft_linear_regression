## NOTES

x = mileage
y = price

1. linear regression equation :

   - **y = mx+b**

2. total squared errors = Sumofsquarederrors(SSE) :

   - **function = Σ(yᵢ − ŷᵢ)²**
   - (ŷᵢ = predicted value ; yᵢ = true value)

3. average squared errors = Mean Squared Error :

   - **Cost function J = (MSE) = 1/n \* SSE**

4. hypothesis function :
   - **h(x) = β₀ + β₁x**
   - (β₀ is the intercept, representing the value of y when x is 0.
   - β₁ is the slope, indicating how much y changes for each unit change in x.)

> - Utilizing the MSE function, the iterative process of **gradient descent** is applied to update the values of θ0 & θ1
> - This ensures that the MSE value converges to the global minima, signifying the most accurate fit of the linear regression line to the dataset.
> - This process involves continuously adjusting the parameters θ0 & θ1 based on the gradients calculated from the MSE.
> - Steps : 1. calculate loss function, 2. optimize model to mitigate error = apply gradient descent

5. Gradient descent :
   - start with random values for slope (0) & intercept (1) = 0
   - calculate error : **MSE**
   - Find how much each parameter contributes to the error : gradient
   - Update the parameters in the direction that reduces the error
   - Repeat until the error is as small as possible

## LINKS

- [ML Linear regression - geeksforgeeks](https://www.geeksforgeeks.org/machine-learning/ml-linear-regression/)
- [Linear regression - ex](https://mlu-explain.github.io/linear-regression/)
- [Gradient descent](https://www.geeksforgeeks.org/machine-learning/gradient-descent-in-linear-regression/)

## TODO

1. Predict price of car for given mileage

   - Prompts user with mileage
   - Prints estimated price for that mileage
   - hypothesis function : **estimatePrice(mileage) = θ0 + (θ1 ∗ mileage)**
   - before running training prog, θ0 = θ1 = 0

2. Train model

   - Reads dataset file
   - Performs linear regression on data
   - Once completed, save θ0 & θ1 for prog 1
   - Note that the estimatePrice is the same as in our first program, but here it uses
     your temporary, lastly computed theta0 and theta1.
   - Also, don’t forget to simultaneously update theta0 and theta1

3. BONUS
   - plot data into graph (scatter)
   - plot best-fit line into same graph
   - calculate precision of algorithm
     - using Mean Absolute Error - low MAE = better perf
     - or RMSE (penalizes outliers - not normalized >> bof)
     - Coefficient of Determination (R-squared)
