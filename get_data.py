import argparse
import urllib.request


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

    req = urllib.request.urlopen(URLS[args.size])
    fname = req.url.split("/")[-1]
    breakpoint()


get_dataset()

