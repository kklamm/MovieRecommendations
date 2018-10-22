import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mf import optimize


def test_optimize():
    R = np.random.random((300, 50))
    R *= (R < 0.01)
    R = csr_matrix(R)

    latent = 20
    X = np.random.rand(R.shape[0], latent)
    Y = np.random.rand(R.shape[1], latent)

    optimize(R, X, Y, 0.1)
