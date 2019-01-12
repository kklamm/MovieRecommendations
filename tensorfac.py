import numpy as np
import sparse
from tqdm import tqdm


def alternate_least_squares(T, M0, M1, M2):
    MM0 = M0 @ M0
    MM1 = M1 @ M1
    MM2 = M2 @ M2

    K = M0.shape[0]
    Ci = MM1 * MM2

    for i in range(T.shape[0]):
        Cji = Ci.copy()
        Oj = np.zeros((1, K))
        for j, k in zip(*T[i].nonzero()):
            w = T[i, j, k]
            if w == 0:
                w += 1
            v = M1 * M2



def implicit_tensor_factorization(tensor,
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

    tensor0 = tensor
    tensor1 = tensor.transpose((1, 0, 2))
    tensor2 = tensor.transpose((2, 0, 1))


    for _ in range(n_epochs):
        alternate_least_squares(tensor0, M0, M1, M2)
        alternate_least_squares(tensor1, M1, M0, M1)
        alternate_least_squares(tensor2, M2, M0, M1)

        C1 = M0 * M2
        C2 = M0 * M1

