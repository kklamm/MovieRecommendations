import numpy as np
import q
from tqdm import tqdm, trange


def loss(R, X, Y, lambda_):
    loss_ = 0
    for u, i in zip(*R.nonzero()):
        loss_ += (R[u, i] - X[u] @ Y[i])**2
    loss_ += lambda_ * (np.sum(X**2) + np.sum(Y**2))
    return loss_


def als_step(R, X, Y, lambda_, alpha=40):
    P = (R > 0).astype(int)
    YtY = Y.T @ Y
    for u in trange(P.shape[0]):
        Pu = P[u].toarray()[0]
        Ru = R[u].toarray()[0]
        Cu = 1 + alpha * Ru
        Cu = np.diag(Cu)
        I = np.diag(np.ones(X.shape[1]))
        Cu_minus_I = Cu - 1
        YtCu_minus_IY = Y.T @ Cu_minus_I @ Y
        X[u] = np.linalg.inv(YtY + YtCu_minus_IY + lambda_ * I) @ Y.T @ Cu @ Pu

    XtX = X.T @ X
    for i in trange(P.shape[1]):
        Pi = P[:, i].toarray()[:, 0]
        Ri = R[:, i].toarray()[:, 0]
        Ci = 1 + 40 * Ri
        Ci = np.diag(Ci)
        I = np.diag(np.ones(X.shape[1]))
        Ci_minus_I = Ci - 1
        XtCu_minus_IX = X.T @ Ci_minus_I @ X
        Y[i] = np.linalg.inv(XtX + XtCu_minus_IX + lambda_ * I) @ X.T @ Ci @ Pi


def alternate_least_squares(R, X, Y, lambda_, *, n_optimize=15, show_loss=False):
    if show_loss:
        print(loss(R, X, Y, lambda_))
    for n in trange(n_optimize):
        als_step(R, X, Y, lambda_)
        if show_loss:
            print(loss(R, X, Y, lambda_))
