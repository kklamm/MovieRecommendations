import fire
import numpy as np
import sparse
from tqdm import tqdm, trange


def to_indptr(row_indices, n_rows):
    indptr = np.zeros(n_rows + 1, dtype=int)
    rows, counts = np.unique(row_indices, return_counts=True)

    cumul = 0
    for row_idx in range(n_rows):
        loc, = np.where(row_idx == rows)
        indptr[row_idx] = cumul
        if loc.size > 0:
            cumul += counts[loc]
    indptr[-1] = cumul
    return indptr


def loss(T, M0, M1, M2, lambda_):
    total = 0
    for i, j, k in zip(*T.nonzero()):
        total += T[i, j, k] * (1 - np.sum(M0[:, i] * M1[:, j] * M2[:, k]))**2
    total += lambda_ * (np.sum(M0**2) + np.sum(M1**2) + np.sum(M2**2))
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
                w = 1
            v = M1[:, j] * M2[:, k]
            Cji += w * v * v[:, None]
            Oj += w * v
        M0[:, i] = np.linalg.inv(Cji + lambda_ * I) @ Oj[0]


def alternate_least_squares_optimized(indptr, y_indices, z_indices, data, M0, M1, M2, lambda_):
    MM0 = M0 @ M0.T
    MM1 = M1 @ M1.T
    MM2 = M2 @ M2.T

    K = M0.shape[0]
    I = np.eye(K)
    Ci = MM1 * MM2

    n0 = M0.shape[0]

    for i in range(n0):
        Cji = Ci.copy()
        Oj = np.zeros((1, K))
        for idx in range(indptr[i], indptr[i+1]):
            j = y_indices[idx]
            k = z_indices[idx]
            confidence = data[idx]
            v = M1[:, j] * M2[:, k]
            Cji += confidence * v * v[:, None]
            Oj += confidence * v
        M0[:, i] = np.linalg.inv(Cji + lambda_ * I) @ Oj[0]


def implicit_tensor_factorization(tensor,
                                  n_users: int,
                                  n_items: int,
                                  n_context: int,
                                  n_latent: int,
                                  lambda_: float,
                                  n_epochs: int = 15,
                                  show_loss: bool = False):
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

    with tqdm(total=n_epochs) as pbar:
        for n in range(n_epochs):
            alternate_least_squares(tensor0, M0, M1, M2, lambda_)
            pbar.n = n + 1/3
            pbar.refresh()
            alternate_least_squares(tensor1, M1, M0, M2, lambda_)
            pbar.n = n + 2/3
            pbar.refresh()
            alternate_least_squares(tensor2, M2, M0, M1, lambda_)
            pbar.n = n + 1

            if show_loss:
                loss_ = loss(tensor, M0, M1, M2, lambda_)
                pbar.set_postfix(loss=loss_)
            pbar.refresh()




def test_tensor_fac(n_users=1000, n_items=200, n_context=3, n_latent=10, n_epochs=15, lambda_=0.1):
    tensor = (sparse.random((n_users, n_items, n_context), density=0.01) * 30).astype(int)
    implicit_tensor_factorization(tensor, n_users, n_items, n_context, n_latent, lambda_, show_loss=True)


fire.Fire()