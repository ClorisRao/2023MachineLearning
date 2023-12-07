import numpy as np
from scipy.special import softmax
import pickle
def logreg_nll_gd(K: int, eta: float, T: int, X: np.ndarray, y: np.array) -> np.ndarray:

    # change y format
    Y = np.zeros((len(y), K))
    for i in range(len(y)):
        Y[i, y[i]] = 1

    # initialize W
    W = np.zeros((X.shape[1], K))

    # iteration
    for _ in range(T):
        W -= eta * (-X.T @ Y + X.T @ softmax(X@W, axis=1))

    return W


def error_rate(W: np.ndarray, X: np.ndarray, y: np.array) -> float:

    y_pred = np.argmax(X @ W, axis=1)
    error = np.mean(y_pred != y)

    return error


if __name__ == '__main__':

    hw5p1 = pickle.load(open('hw5\hw5p1.pkl', 'rb'))

    X_train = hw5p1['data']
    y_train = hw5p1['labels']
    X_test = hw5p1['testdata']
    y_test = hw5p1['testlabels']

    n, d= X_train.shape
    K = len(np.unique(y_train))

    eta = 2 / n
    T = 10000

    # (a)
    # Training model
    W = logreg_nll_gd(K, eta, T, X_train, y_train)

    # error rate
    training_error_rate = error_rate(W, X_train, y_train)
    testing_error_rate = error_rate(W, X_test, y_test)

