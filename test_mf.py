import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mf import als_step, alternate_least_squares, loss


@pytest.fixture
def R():
    r = np.random.rand(300, 50)
    r *= (r < 0.01)
    r = csr_matrix(r)
    return r


def test_als_step(R):
    latent = 20
    X = np.random.rand(R.shape[0], latent)
    Y = np.random.rand(R.shape[1], latent)
    l = loss(R, X, Y, 0.1)
    als_step(R, X, Y, 0.1)
    l2 = loss(R, X, Y, 0.1)
    assert l2 < l


def test_alternate_least_squares(R):
    latent = 20
    X = np.random.rand(R.shape[0], latent)
    Y = np.random.rand(R.shape[1], latent)

    alternate_least_squares(R, X, Y, 0.1, show_loss=True)


