from dataclasses import dataclass
from typing import List


@dataclass
class Movie:
    index: int
    title: str
    genres: List[str]


def df_to_movies(df):
    pass
