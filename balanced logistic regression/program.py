import numpy as np
import pickle

def logreg_nll_gd(lamda: float, eta: float, T: int, X: np.ndarray, y: np.array, w_mle: np.array, b_mle: float) -> np.ndarray:

    n, d = X.shape
    d += 1

    # change X
    X = np.concatenate([X, np.ones((n, 1))], axis=1)
    X_0 = X[y == 0]
    X_1 = X[y == 1]
    n0 = X_0.shape[0]
    n1 = X_1.shape[0]

    # initialize W
    init_w = np.append(w_mle, b_mle).reshape((d, 1))
    w = init_w.copy()

    # iteration
    for _ in range(T):
        w -= eta * (lamda * (w-init_w) + 1/(2*n0) * X_0.T @ (1/(1+np.exp(-X_0 @ w))) - 1/(2*n1) * X_1.T @ (1/(1+np.exp(X_1 @ w))))

    return w

def error_rate(w: np.ndarray, X: np.ndarray, y: np.array) -> float:

    # change X
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

    y_pred = X @ w > 0

    balanced_rate = 1 / 2 * np.mean(y_pred[y==0] == 1) + 1 / 2 * np.mean(y_pred[y==1] == 0)

    return balanced_rate

if __name__ == '__main__':

    click = pickle.load(open('hw5\hw5click.pkl', 'rb'))

    X_train = click['data']
    y_train = click['labels']
    X_test = click['testdata']
    y_test = click['testlabels']
    w = click['w_mle']
    b = click['b_mle']

    lamda = 0.01
    T = 50000
    eta = 0.001

    # train model
    w_model = logreg_nll_gd(lamda, eta, T, X_train, y_train, w, b)

    # balanced error rate
    training_error_rate = error_rate(w_model, X_train, y_train)
    testing_error_rate = error_rate(w_model, X_test, y_test)

