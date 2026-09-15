
import numpy as np


def hypothesis(X, theta):
    return X @ theta


def cost(X, y, theta):
    predictions = hypothesis(X, theta)
    errors = predictions - y
    return 0.5 * np.sum(errors ** 2)


def fit_normal(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta


def fit_batch_gd(X, y, alpha, iterations):

    theta = np.zeros(X.shape[1])
    costs = []

    for i in range(iterations):

        predictions = hypothesis(X, theta)

        errors = y - predictions

        theta = theta + alpha * (X.T @ errors)

        costs.append(cost(X, y, theta))

    return theta, costs


def fit_sgd(X, y, alpha, epochs):

    theta = np.zeros(X.shape[1])
    costs = []

    for epoch in range(epochs):

        for i in range(len(X)):

            prediction = X[i] @ theta

            error = y[i] - prediction

            theta = theta + alpha * error * X[i]

        costs.append(cost(X, y, theta))

    return theta, costs


def rmse(y_true, y_pred):

    return np.sqrt(
        np.mean((y_true - y_pred) ** 2)
    )
