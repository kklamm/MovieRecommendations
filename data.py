from joblib import Memory
import numpy as np
from scipy.sparse import lil_matrix


memory = Memory(location="~/.joblib/")


@memory.cache
def df_to_implicit_sparse(df):
    pass