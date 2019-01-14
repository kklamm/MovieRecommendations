import fire
import numpy as np
import sparse
from tqdm import tqdm, trange


def loss(T, M0, M1, M2, lambda_):
    total = 0
    for i, j, k in zip(*T.nonzero()):
        total += T[i, j, k] * (1 - np.sum(M0[:, i] * M1[:, j] * M2[:, k]))**2
    total += lambda_ * np.sum(M0**2 + M1**2 + M2**2)
    return total


def alternate_least_squares(T, M0, M1, M2, lambda_):
    MM0 = M0 @ M0.T
    MM1 = M1 @ M1.T
    MM2 = M2 @ M2.T

    K = M0.shape[0]
    I = np.eye(K)
    Ci = MM1 * MM2

    for i in range(T.shape[0]):
        Cji = Ci.copy()
        Oj = np.zeros((1, K))
        for j, k in zip(*T[i].nonzero()):
            w = T[i, j, k]
            if w == 0:
                w += 1
            v = M1[:, j] * M2[:, k]
            Cji += w * v * v[:, None]
            Oj += w * v
        M0[:, i] = np.linalg.inv(Cji + lambda_ * I) @ Oj[0]


def implicit_tensor_factorization(tensor,
                                  n_users: int,
                                  n_items: int,
                                  n_context: int,
                                  n_latent: int,
                                  lambda_: float,
                                  n_epochs: int = 15):
    """
    Implementation of ALS-based tensor factorization for three dimensions (user-item
    matrix with one additional context dimension)
    """

    M0 = np.random.rand(n_latent, n_users)
    M1 = np.random.rand(n_latent, n_items)
    M2 = np.random.rand(n_latent, n_context)

    tensor0 = tensor
    tensor1 = tensor.transpose((1, 0, 2))
    tensor2 = tensor.transpose((2, 0, 1))


    for _ in trange(n_epochs):
        alternate_least_squares(tensor0, M0, M1, M2, lambda_)
        alternate_least_squares(tensor1, M1, M0, M1, lambda_)
        alternate_least_squares(tensor2, M2, M0, M1, lambda_)


def test_tensor_fac(n_users=100, n_items=20, n_context=3, n_latent=10, n_epochs=15, lambda_=0.1):
    tensor = sparse.random((n_users, n_items, n_context), density=0.01)

    implicit_tensor_factorization(tensor, n_users, n_items, n_context, n_latent, lambda_)


fire.Fire()