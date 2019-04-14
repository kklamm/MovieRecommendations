import fire
import numba
import numpy as np
import sparse
from tqdm import tqdm, trange


@numba.njit
def to_indptr(row_indices, n_rows):
    indptr = np.zeros(n_rows + 1, dtype=np.int32)
    rows = np.unique(row_indices)
    counts = np.zeros(n_rows, dtype=np.int32)
    for row in row_indices:
        counts[row] += 1
    counts = counts[counts > 0]

    cumul = 0
    for row_idx in range(n_rows):
        loc, = np.where(row_idx == rows)
        indptr[row_idx] = cumul
        if loc.size > 0:
            cumul += counts[loc[0]]
    indptr[-1] = cumul
    return indptr


def loss(T, M0, M1, M2, lambda_):
    total = 0
    for i, j, k in zip(*T.nonzero()):
        total += T[i, j, k] * (1 - np.sum(M0[:, i] * M1[:, j] * M2[:, k]))**2
    total += lambda_ * (np.sum(M0**2) + np.sum(M1**2) + np.sum(M2**2))
    return total


@numba.njit
def loss_optimized(indptr, coord1, coord2, data, M0, M1, M2, lambda_):
    total = 0
    for i in range(indptr.size - 1):
        for idx in range(indptr[i], indptr[i+1]):
            j = coord1[idx]
            k = coord2[idx]
            val = data[idx]
            if val > 0:
                confidence = val
                val = 1
            else:
                val = 0
                confidence = 1

            total += confidence  * (val - np.sum(M0[:, i] * M1[:, j] * M2[:, k]))**2
    total += lambda_ * (np.sum(M0**2) + np.sum(M1**2) + np.sum(M2**2))
    return total


@numba.njit
def conjugate_gradient(A, b, x, n_iter=3):
    r = b - A @ x
    p = r
    rsold = r @ r

    for _ in range(n_iter):
        Ap = A @ p
        alpha = rsold / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = r @ r
        if rsnew < 1e-20:
            break
        p = r + rsnew / rsold * p
        rsold = rsnew


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


@numba.njit(parallel=False)
def alternate_least_squares_optimized(indptr, y_indices, z_indices, data, M0, M1, M2, lambda_):
    MM0 = M0 @ M0.T
    MM1 = M1 @ M1.T
    MM2 = M2 @ M2.T

    K = M0.shape[0]
    I = np.eye(K)
    Ci = MM1 * MM2

    n0 = M0.shape[1]

    for i in numba.prange(n0):
        Cji = Ci.copy()
        Oj = np.zeros((1, K))

        if indptr[i] == indptr[i+1]:
            M0[:, i] = 0
            continue

        for idx in range(indptr[i], indptr[i+1]):
            j = y_indices[idx]
            k = z_indices[idx]
            v = np.empty((1, K))
            confidence = data[idx]
            v[0] = M1[:, j] * M2[:, k]
            Cji += confidence * v * v.T
            Oj += confidence * v

        conjugate_gradient(Cji + lambda_ * I, Oj[0], M0[:, i])


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


def implicit_tensor_factorization_optimized(tensor,
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

    indptr0 = to_indptr(tensor.coords[0], tensor.nnz)
    indptr1 = to_indptr(tensor.coords[1], tensor.nnz)
    indptr2 = to_indptr(tensor.coords[2], tensor.nnz)

    with tqdm(total=n_epochs) as pbar:
        for n in range(n_epochs):
            alternate_least_squares_optimized(indptr0, tensor0.coords[1], tensor0.coords[2], tensor0.data,
                                              M0, M1, M2, lambda_)
            pbar.n = n + 1/3
            pbar.refresh()
            alternate_least_squares_optimized(indptr1, tensor1.coords[1], tensor1.coords[2], tensor1.data,
                                              M1, M0, M2, lambda_)
            pbar.n = n + 2/3
            pbar.refresh()
            alternate_least_squares_optimized(indptr2, tensor2.coords[1], tensor2.coords[2], tensor2.data,
                                              M2, M0, M1, lambda_)
            pbar.n = n + 1

            if show_loss:
                loss_ = loss_optimized(indptr0, tensor.coords[1], tensor.coords[2], tensor.data,
                                       M0, M1, M2, lambda_)
                pbar.set_postfix(loss=loss_)
            pbar.refresh()


def test_tensor_fac(n_users=100_000, n_items=2000, n_context=3, n_latent=10, n_epochs=15, lambda_=0.1):
    tensor = (sparse.random((n_users, n_items, n_context), density=0.001) * 30).astype(int)
    implicit_tensor_factorization_optimized(tensor, n_users, n_items, n_context, n_latent, lambda_, show_loss=True)


fire.Fire()