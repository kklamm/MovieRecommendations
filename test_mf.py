import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mf import optimize, loss


def test_optimize():
    R = np.random.rand(300, 50)
    R *= (R < 0.01)
    R = csr_matrix(R)


    latent = 20
    X = np.random.rand(R.shape[0], latent)
    Y = np.random.rand(R.shape[1], latent)
    l = loss(R, X, Y, 0.1)
    optimize(R, X, Y, 0.1)
    l2 = loss(R, X, Y, 0.1)
    assert l2 < l
