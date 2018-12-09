import numpy as np
import q
from tqdm import tqdm, trange


def loss(R, X, Y, lambda_):
    loss_ = 0
    for u, i in zip(*R.nonzero()):
        loss_ += (R[u, i] - X[u] @ Y[i])**2
    loss_ += lambda_ * (np.sum(X**2) + np.sum(Y**2))
    return loss_


def als_step(R, X, Y, lambda_):
    P = (R > 0).astype(int)
    YtY = Y.T @ Y
    for u in range(P.shape[0]):
        Pu = P[u].toarray()[0]
        Ru = R[u].toarray()[0]
        Cu = np.diag(Ru)
        I = np.diag(np.ones(X.shape[1]))
        YtCuY = YtY + Y.T @ (Cu - np.diag(np.ones(Cu.shape[0]))) @ Y
        X[u] = np.linalg.inv(YtCuY + lambda_ * I) @ Y.T @ Cu @ Pu


def alternate_least_squares(R, X, Y, lambda_, *, n_optimize=15, show_loss=False):
    with tqdm(total=n_optimize) as pbar:
        for n in range(n_optimize):
            als_step(R, X, Y, lambda_)
            pbar.update(0.5)
            als_step(R.T, Y, X, lambda_)
            pbar.update(0.5)
            if show_loss:
                l = loss(R, X, Y, lambda_)
                pbar.set_postfix({"loss": l})
