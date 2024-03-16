import public_tests
from multi_linear_reg import compute_cost, compute_gradient, zscore_normalize_features, gradient_descent
import numpy as np
import matplotlib.pyplot as plt


data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
X_train = data[:, :4]
y_train = data[:, 4]
X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
fig, ax = plt.subplots(1, 4, figsize=(25, 5), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()

public_tests.compute_cost_test(compute_cost)
public_tests.compute_gradient_test(compute_gradient)

X_norm, mu, sigma = zscore_normalize_features(X_train)
# print(X_norm.shape, X_train.shape)
w, b, _ = gradient_descent(X_norm, y_train, np.zeros(len(X_norm[0])), 0, compute_cost, compute_gradient, 0.1, 1_000)

X = np.array([[1200, 3, 1, 40]])
X = (X - mu) / sigma
f = w @ X.T + b
print(f * 1000, "$")


X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']
y_predicted = w @ X_norm.T + b
fig, ax = plt.subplots(1, 4, figsize=(25, 5), sharey=True)
for i in range(len(ax)):
    ax[i].plot(X_train[:, i], y_train, 'bo')
    ax[i].plot(X_train[:, i], y_predicted, 'ro')
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()