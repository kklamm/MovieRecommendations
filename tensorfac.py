import numpy as np


def least_squares(X, Y, row, col, val):
    C = X * Y
    w = np.maximum(val, 1)


def tensorfac(u: np.ndarray, i: np.ndarray, c: np.ndarray,
              n_users: int,
              n_items: int,
              n_context: int,
              n_latent: int,
              n_epochs: int):
    """
    Implementation of ALS-based tensor factorization for three dimensions (user-item
    matrix with one additional context dimension)
    """

    M0 = np.random.rand(n_latent, n_users)
    M1 = np.random.rand(n_latent, n_items)
    M2 = np.random.rand(n_latent, n_context)

    MM0 = M0 @ M0
    MM1 = M1 @ M1
    MM2 = M2 @ M2

    for _ in range(n_epochs):
        least_squares(MM1, MM2)

        C1 = M0 * M2
        C2 = M0 * M1

