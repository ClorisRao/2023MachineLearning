import torch
import numpy as np
import pickle


def logreg_nll_gd(K: int, eta: float, T: int, X: np.ndarray, y: np.array) -> torch.tensor:

    def J(W):

        res = -torch.sum(torch.dot(torch.flatten(torch.log(torch.softmax(torch.matmul(X, W), dim=1))), torch.flatten(Y)))

        return res

    # change y format
    Y = np.zeros((len(y), K))
    for i in range(len(y)):
        Y[i, y[i]] = 1

    X = torch.tensor(X, dtype=torch.double)
    Y = torch.tensor(Y, dtype=torch.double)
    W = torch.zeros(X.shape[1], K, dtype=torch.double)
    W.requires_grad = True

    for _ in range(T):
        objective_value = J(W)
        objective_value.backward()

        with torch.no_grad():
            W -= eta * W.grad
            W.grad.zero_()

    return W.detach()

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

    W = logreg_nll_gd(K, eta, T, X_train, y_train)