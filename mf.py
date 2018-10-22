import numpy as np
import q


def optimize(P, C, X, Y, lambda_):
    YtY = Y.T @ Y

    for u in range(P.shape[0]):
        Cu = np.diag(C[u])
        I = np.diag(np.ones(X.shape[1]))
        Cu_minus_I = np.diag(C[u] - 1)
        YtCu_minus_IY = Y.T @ Cu_minus_I @ Y
        X[u] = np.linalg.inv(YtY + YtCu_minus_IY + lambda_ * I) @ Y.T @ Cu @ P[u]
