import numpy as np
import pickle

def gradient(X, y, w, b):

    n, d = X.shape
    d += 1

    # transform y
    y = y.reshape((len(y), 1))
    # transform X
    X = np.concatenate([X, np.ones((n, 1))], axis=1)
    # transform w
    w = np.append(w, b).reshape((d, 1))

    # calculate gradient
    g = X.T @ (1 / (1 + np.exp(-X @ w))- y)

    return g

def error_rate(X, y, w, b):

    y_pred = (np.sum(X * w, axis=1) + b) > 0
    rate = np.mean(y_pred != y)

    return rate


if __name__ == '__main__':

    click = pickle.load(open('hw5\hw5click.pkl', 'rb'))

    X_train = click['data']
    y_train = click['labels']
    X_test = click['testdata']
    y_test = click['testlabels']
    w = click['w_mle']
    b = click['b_mle']

    # (a)
    g = gradient(X_train, y_train, w, b)
    norm = np.sqrt(np.sum(g * g))
    print('Euclidean norm is', norm)

    # (b)
    training_error_rate = error_rate(X_train, y_train, w, b)
    testing_error_rate = error_rate(X_test, y_test, w ,b)
    print('Training error rate is ', training_error_rate)
    print('Testing error rate is ', testing_error_rate)