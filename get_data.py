import argparse
import pathlib
import urllib.request
import zipfile


URLS = {
    "small": "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "medium": "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "large": "http://files.grouplens.org/datasets/movielens/ml-10m.zip"
}


def get_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("size", choices=["small", "medium", "large"],
                        help="Dataset size")
    args = parser.parse_args()

    url = URLS[args.size]
    fname = url.split("/")[-1]
    if not pathlib.Path(fname).exists():
        urllib.request.urlretrieve(url, fname)
    with zipfile.ZipFile(fname) as zf:
        zf.extractall()


get_dataset()

