import numpy as np
import matplotlib.pyplot as plt
import public_tests
from utils import load_data


# Function of the cost:
def compute_cost(x, y, w, b):
    # Calculating the number of data points:
    m = len(x)
    # Calculating the cost of linear regression:
    total_cost = np.sum((w * x + b - y) ** 2) / (2 * m)
    # Returning the results:
    return total_cost


# Function of the gradient:
def compute_gradient(x, y, w, b):
    # Calculating the predicted values f of the linear regression:
    f = w * x + b
    # Calculating the error between predicted and actual values:
    error = f - y
    # Calculating the gradient of the cost function with b parameter:
    dj_db = np.sum(error) / len(x)
    # Calculating the gradient of the cost function with w parameter:
    dj_dw = np.sum(error * x) / len(x)
    # Returning the results:
    return dj_dw, dj_db


# Function of the gradient descent:
def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    # Making an empty list to store history of cost values:
    J_history = []
    # Initalizing parameters
    w = w_in
    b = b_in
    # Looping through parameters w and b,
    # using gradient descent for num_iters number of times,
    # adjusting w and b to minimize cost function,
    # recording each step in J_history:
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        J_history.append(cost_function(x, y, w, b))
    return w, b, J_history


X, y = load_data()

# Plotting the results:
plt.plot(X, y, 'rx')
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000")
plt.title("Profits vs. Population per city")
plt.show()

result = compute_cost(x=X, y=y, w=2, b=1)
print("Cost function = ", result)

public_tests.compute_cost_test(compute_cost)

result = compute_gradient(x=X, y=y, w=0.2, b=0.2)
print("Gradient = ", result)

public_tests.compute_gradient_test(compute_gradient)

w, b, history = gradient_descent(X, y, 0, 0, compute_cost, compute_gradient, 0.01, 1500)
print("W, b = ", w, b)
plt.plot(X, y, 'rx', X, w * X + b)
plt.xlabel("Population of City in 10,000s")
plt.ylabel("Profit in $10,000")
plt.title("Profits vs. Population per city")
plt.show()
