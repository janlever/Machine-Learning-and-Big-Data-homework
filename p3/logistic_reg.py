import numpy as np
import copy
import math
from utils import *


def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    """
    g = 1 / (1 + np.exp(-z))

    return g


#########################################################################
# logistic regression
#
def compute_cost(X, y, w, b, lambda_=None):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value
      w : (array_like Shape (n,)) Values of parameters of the model
      b : scalar Values of bias parameter of the model
      lambda_: unused placeholder
    Returns:
      total_cost: (scalar)         cost
    """
    m = len(y)
    est = np.dot(X, w) + b
    est = sigmoid(est)

    loss = (np.dot(np.log(est), -y)) - np.dot(np.log(1 - est), 1 - y)

    total_cost = np.sum(loss) / m

    return total_cost


def compute_gradient(X, y, w, b, lambda_=None):
    """
    Computes the gradient for logistic regression

    Args:
      X : (ndarray Shape (m,n)) variable such as house size
      y : (array_like Shape (m,1)) actual value
      w : (array_like Shape (n,1)) values of parameters of the model
      b : (scalar)                 value of parameter of the model
      lambda_: unused placeholder
    Returns
      dj_db: (scalar)                The gradient of the cost w.r.t. the parameter b.
      dj_dw: (array_like Shape (n,1)) The gradient of the cost w.r.t. the parameters w.
    """
    m = len(y)
    est = np.dot(X, w) + b
    est = sigmoid(est)

    dj_db = np.sum(est - y) / m
    dj_dw = np.dot(est - y, X) / m

    return dj_db, dj_dw


#########################################################################
# regularized logistic regression
#
def compute_cost_reg(X, y, w, b, lambda_=1):
    """
    Computes the cost over all examples
    Args:
      X : (array_like Shape (m,n)) data, m examples by n features
      y : (array_like Shape (m,)) target value 
      w : (array_like Shape (n,)) Values of parameters of the model      
      b : (array_like Shape (n,)) Values of bias parameter of the model
      lambda_ : (scalar, float)    Controls amount of regularization
    Returns:
      total_cost: (scalar)         cost 
    """
    m = len(y)
    est = np.dot(X, w) + b
    est = sigmoid(est)

    loss = ((np.dot(np.log(est), -y)) - np.dot(np.log(1 - est), 1 - y)) * (1 / m)
    loss = np.sum(loss)
    lambda_part = np.sum((lambda_ / (2 * m)) * np.square(w))

    total_cost = loss + lambda_part

    return total_cost


def compute_gradient_reg(X, y, w, b, lambda_=1):
    """
    Computes the gradient for linear regression 

    Args:
      X : (ndarray Shape (m,n))   variable such as house size 
      y : (ndarray Shape (m,))    actual value 
      w : (ndarray Shape (n,))    values of parameters of the model      
      b : (scalar)                value of parameter of the model  
      lambda_ : (scalar,float)    regularization constant
    Returns
      dj_db: (scalar)             The gradient of the cost w.r.t. the parameter b. 
      dj_dw: (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 

    """
    m = len(y)
    est = np.dot(X, w) + b
    est = sigmoid(est)

    dj_db = np.sum(est - y) / m
    dj_dw = np.dot(est - y, X) / m + lambda_ * (1 / m) * w

    return dj_db, dj_dw


#########################################################################
# gradient descent
#
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_=None):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha

    Args:
      X :    (array_like Shape (m, n)
      y :    (array_like Shape (m,))
      w_in : (array_like Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)                 Initial value of parameter of the model
      cost_function:                  function to compute cost
      alpha : (float)                 Learning rate
      num_iters : (int)               number of iterations to run gradient descent
      lambda_ (scalar, float)         regularization constant

    Returns:
      w : (array_like Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
      J_history : (ndarray): Shape (num_iters,) J at each iteration,
          primarily for graphing later
    """
    w = w_in
    b = b_in
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X=X, y=y, w=w, b=b, lambda_=lambda_)
        w = w - np.multiply(alpha, dj_dw)
        b = b - np.multiply(alpha, dj_db)
        J_history[i] = cost_function(X, y, w, b, lambda_)

    return w, b, J_history


#########################################################################
# predict
#
def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w and b

    Args:
    X : (ndarray Shape (m, n))
    w : (array_like Shape (n,))      Parameters of the model
    b : (scalar, float)              Parameter of the model

    Returns:
    p: (ndarray (m,1))
        The predictions for X using a threshold at 0.5
    """
    est_prob = sigmoid(np.dot(X, w) + b)

    p = (est_prob >= 0.5).astype(int)
    return p

def load_data():
    data = np.loadtxt('./data/ex2data1.txt', delimiter=',')

    # Extract X and y
    X = data[:, :-1]  # All rows, all columns except the last one
    y = data[:, -1]  # All rows, only the last column

    return np.array(X), np.array(y)


X, y = load_data()

# Plotting
w_in, b_in, J_history = gradient_descent(X=X, y=y, alpha=0.001, b_in=-8, w_in=np.zeros(2), cost_function=compute_cost,
                                         gradient_function=compute_gradient, num_iters=10000, lambda_=0)
print([J_history[-1]])
plot_decision_boundary(w=w_in, b=b_in, X=X, y=y)
plt.show()

plt.plot(J_history, marker='o', linestyle='-')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function over Iterations')
plt.show()
