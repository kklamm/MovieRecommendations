from dataclasses import dataclass
from datetime import datetime

from joblib import Memory
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from tqdm import tqdm

from movies import MovieHandler

memory = Memory(location="~/.joblib/")


@dataclass
class Rating:
    index: int
    user_id: int
    rating: float
    timestamp: datetime


@memory.cache
def _df_to_implicit_sparse(df, user_index_mapping, movie_index_mapping):
    n_users = len(user_index_mapping)
    n_items = len(movie_index_mapping)
    mat = lil_matrix((n_users, n_items))
    users = df.userId.map(user_index_mapping)
    movies = df.movieId.map(movie_index_mapping)
    values = np.ones_like(users)

    return coo_matrix((values, (movies, values)))


class MovieRatings:
    def __init__(self, ratings_df, ):
        self._df = ratings_df
        self._users = ratings_df.userId.sort_values().unique()

    @classmethod
    def from_file(cls, filename):
        ratings_df = pd.read_csv(filename)
        return cls(ratings_df)

    @property
    def user_index_mapping(self):
        if not hasattr(self, "_user_index_mapping"):
            self._user_index_mapping = {u: i for (i, u) in enumerate(self._users)}
        return self._user_index_mapping

    @property
    def index_user_mapping(self):
        if not hasattr(self, "_index_user_mapping"):
            self._index_user_mapping = {i: u for (i, u) in self._user_index_mapping.items()}
        return self._index_user_mapping



def main():
