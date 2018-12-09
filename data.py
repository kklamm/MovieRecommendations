from dataclasses import dataclass
from datetime import datetime
import logging
import pathlib

import fire
from joblib import Memory
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from tqdm import tqdm

from movies import MovieHandler
from mf import alternate_least_squares


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

memory = Memory(location="~/.joblib/", verbose=True)


@dataclass
class Rating:
    index: int
    user_id: int
    rating: float
    timestamp: datetime


def user_item_matrix(df, user_index_mapping, movie_index_mapping):
    n_users = len(user_index_mapping)
    n_items = len(movie_index_mapping)
    users = df["UserID"].map(user_index_mapping)
    movies = df["MovieID"].map(movie_index_mapping)
    values = df["Rating"]
    return coo_matrix((values, (users, movies)), shape=(n_users, n_items))


class MovieRatings:
    def __init__(self, ratings_df):
        self._df = ratings_df
        self._users = ratings_df["UserID"].sort_values().unique()

    @classmethod
    def from_file(cls, filename):
        ratings_df = pd.read_csv(filename, sep="::", names=["UserID", "MovieID",
                                                            "Rating", "Timestamp"])
        return cls(ratings_df)

    @property
    def df(self):
        return self._df

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


def main(path):
    path = pathlib.Path(path)
    movie_handler = MovieHandler.from_file(path / "movies.dat")
    movie_ratings = MovieRatings.from_file(path / "ratings.dat")
    ui_mat = user_item_matrix(movie_ratings.df, movie_ratings.user_index_mapping, movie_handler.id_index_mapping)

    n_latent = 20
    R = ui_mat.tocsr()
    X = np.random.rand(R.shape[0], n_latent)
    Y = np.random.rand(R.shape[1], n_latent)

    alternate_least_squares(R, X, Y, lambda_=0.1, show_loss=True)

    memory.clear()


if __name__ == '__main__':
    fire.Fire(main)
