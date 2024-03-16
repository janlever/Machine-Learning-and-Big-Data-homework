import numpy as np
from scipy.io import loadmat
import utils


#########################################################################
# one-vs-all
#
def oneVsAll(X, y, n_labels, lambda_):
    """
     Trains n_labels logistic regression classifiers and returns
     each of these classifiers in a matrix all_theta, where the i-th
     row of all_theta corresponds to the classifier for label i.

     Parameters
     ----------
     X : array_like
         The input dataset of shape (m x n). m is the number of
         data points, and n is the number of features. 

     y : array_like
         The data labels. A vector of shape (m, ).

     n_labels : int
         Number of possible labels.

     lambda_ : float
         The logistic regularization parameter.

     Returns
     -------
     all_theta : array_like
         The trained parameters for logistic regression for each class.
         This is a matrix of shape (K x n+1) where K is number of classes
         (ie. `n_labels`) and n is number of features without the bias.
     """

    w_matrix = np.zeros((n_labels, len(X[0])))  
    b_matrix = np.zeros(n_labels)  

    y = np.array([np.where(y == c, 1, 0) for c in range(10)])

    for i in range(1000):
        est = np.dot(w_matrix, X.T) + b_matrix[:, np.newaxis]
        est = 1 / (1 + np.exp(-est))

        dj_db = np.sum(est - y) / len(y)
        dj_dw = np.dot(est - y, X) / len(y) + lambda_ * (1 / len(y)) * w_matrix

        w_matrix = w_matrix - np.multiply(0.006, dj_dw)
        b_matrix = b_matrix - np.multiply(0.006, dj_db)

    all_theta = np.hstack((b_matrix.reshape(-1, 1), w_matrix))

    return all_theta


def predictOneVsAll(all_theta, X):
    """
    Return a vector of predictions for each example in the matrix X. 
    Note that X contains the examples in rows. all_theta is a matrix where
    the i-th row is a trained logistic regression theta vector for the 
    i-th class. You should set p to a vector of values from 0..K-1 
    (e.g., p = [0, 2, 0, 1] predicts classes 0, 2, 0, 1 for 4 examples) .

    Parameters
    ----------
    all_theta : array_like
        The trained parameters for logistic regression for each class.
        This is a matrix of shape (K x n+1) where K is number of classes
        and n is number of features without the bias.

    X : array_like
        Data points to predict their labels. This is a matrix of shape 
        (m x n) where m is number of data points to predict, and n is number 
        of features without the bias term. Note we add the bias term for X in 
        this function. 

    Returns
    -------
    p : array_like
        The predictions for each data point in X. This is a vector of shape (m, ).
    """

    '''print("all_theat: ")
    print(all_theta)
'''
    X_with_bias = np.insert(X, 0, 1, axis=1)

    # Computing the probabilities for all examples in each class
    probs = np.dot(all_theta, X_with_bias.T)

    # Getting the index of max prob. for each example
    p = np.argmax(probs, axis=0)

    return p


#########################################################################
# NN
#
def predict(theta1, theta2, X):
    """
    Predict the label of an input given a trained neural network.

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size)

    X : array_like
        The image inputs having shape (number of examples x image dimensions).

    Return 
    ------
    p : array_like
        Predictions vector containing the predicted label for each example.
        It has a length equal to the number of examples.
    """

    m = X.shape[0]
    X1s = np.hstack([np.ones((m, 1)), X])

    a1 = 1 / (1 + np.exp(-np.dot(theta1, X1s.T)))
    m = a1.shape[0]
    a1s = np.vstack([np.ones((1, a1.shape[1])), a1])
    # print(len(a1s), len(a1s[0]))
    pp = 1 / (1 + np.exp(-np.dot(theta2, a1s)))

    p = np.argmax(pp, axis=0)

    return p


data = loadmat('data/ex3data1.mat', squeeze_me=True)

X = data['X']
y = data['y']

rand_indices = np.random.choice(X.shape[0], 100, replace=False)
utils.displayData(X[rand_indices, :])

all_theta = oneVsAll(X=X, y=y, n_labels=10, lambda_=0.003)
predict_log_reg = predictOneVsAll(all_theta=all_theta, X=X)

weights = loadmat('data/ex3weights.mat')
theta1, theta2 = weights['Theta1'], weights['Theta2']
predict_neural_netw = predict(theta1=theta1, theta2=theta2, X=X)


print("y : ", len(y))
print("log reg: " + str(len(predict_log_reg)) + ", ")
print("neural net: " + str(len(predict_neural_netw)) + ", ")

# Number of matching elements
predictions_log_reg = np.sum(y == predict_log_reg)
predictions_neural_netw = np.sum(y == predict_neural_netw)

# Total number of elements
total_examples = len(y)

# Accuracy
accuracy_log_reg = (predictions_log_reg / total_examples) * 100.0
accuracy_neural_netw = (predictions_neural_netw / total_examples) * 100.0

print("Accuracy - logistic regression: ", accuracy_log_reg, "%")
print("Accuracy - neural network: ", accuracy_neural_netw, "%")