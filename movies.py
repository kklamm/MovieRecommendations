from dataclasses import dataclass
import logging
from typing import List

from joblib import Memory
import pandas as pd


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

memory = Memory(location="~/joblib", verbose=True)


@dataclass
class Movie:
    index: int
    movie_id: int
    title: str
    genres: List[str]


def df_to_movies(df):
    df = df.set_index("movieId").sort_index()
    movies = []
    for i, (id_, title, genres) in enumerate(df.itertuples()):
        movies.append(Movie(i, id_, title, genres))
    return movies


class MovieHandler:
    def __init__(self, movies_df):
        self.movies = df_to_movies(movies_df)

    @classmethod
    def from_file(cls, filename):
        df = pd.read_csv(filename, sep="::", names=["movieId", "title", "genre"])
        return cls(df)

    @property
    def index_id_mapping(self):
        if not hasattr(self, "_index_id_mapping"):
            self._index_id_mapping = {m.index: m.movie_id for m in self.movies}
        return self._index_id_mapping

    @property
    def id_index_mapping(self):
        if not hasattr(self, "_id_index_mapping"):
            self._id_index_mapping = {movie_id: index for (index, movie_id) in self.index_id_mapping.items()}
        return self._id_index_mapping


def main():
    mh = MovieHandler.from_file("ml-10M100K/movies.dat")


if __name__ == '__main__':
    main()
    memory.clear()