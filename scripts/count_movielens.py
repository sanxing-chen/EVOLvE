#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd


def infer_ratings_path(path: str) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")

    # Try typical MovieLens filenames in order
    candidates = [
        p / "ratings.csv",
        p / "ratings.dat",
        p / "u.data",
        p / "u1.base",  # alternate split names in 100k
        p / "u.base",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Fallback: first file starting with 'ratings' or exact 'u.data'
    for child in p.iterdir():
        name = child.name.lower()
        if child.is_file() and (name.startswith("ratings") or name == "u.data"):
            return child

    raise FileNotFoundError(
        f"Could not locate ratings file in directory: {path}. Expected one of ratings.csv, ratings.dat, u.data"
    )


def load_from_file(ratings_file: Path) -> pd.DataFrame:
    name = ratings_file.name.lower()
    if name.endswith(".dat") or name == "ratings.dat":
        # MovieLens 1M format: userId::movieId::rating::timestamp
        return pd.read_csv(
            ratings_file,
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
            header=None,
        )
    if name.endswith(".data") or name == "u.data" or name.endswith(".tsv"):
        # MovieLens 100k format: tab-separated w/o header
        return pd.read_csv(
            ratings_file,
            sep="\t",
            names=["userId", "movieId", "rating", "timestamp"],
            header=None,
        )
    if name.endswith(".csv"):
        df = pd.read_csv(ratings_file)
        return df

    # Fallback: try auto-detect with whitespace
    try:
        return pd.read_csv(ratings_file, sep="\s+", header=None, names=["userId", "movieId", "rating", "timestamp"])
    except Exception as e:
        raise ValueError(f"Unsupported file format for {ratings_file}: {e}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    # Common variants
    user_col = cols.get("userid") or cols.get("user_id") or cols.get("user")
    movie_col = cols.get("movieid") or cols.get("movie_id") or cols.get("movie")
    rating_col = cols.get("rating") or cols.get("user_rating")

    if not user_col or not movie_col:
        # Perhaps DataFrame used our default names via header=None
        for candidate in ("userId", "user_id"):
            if candidate in df.columns:
                user_col = candidate
                break
        for candidate in ("movieId", "movie_id"):
            if candidate in df.columns:
                movie_col = candidate
                break
        for candidate in ("rating", "user_rating"):
            if candidate in df.columns:
                rating_col = candidate
                break

    # Final validation
    if not user_col or not movie_col:
        raise ValueError(
            f"Could not identify user/movie columns. Available columns: {list(df.columns)}"
        )

    # Ensure we have a rating column even if not strictly required for counts
    if not rating_col:
        # Create a dummy rating column to simplify downstream logic
        df = df.copy()
        df["rating"] = 1
        rating_col = "rating"

    return df.rename(columns={user_col: "userId", movie_col: "movieId", rating_col: "rating"})


def load_from_hf(hf_id_or_path: str) -> pd.DataFrame:
    try:
        import datasets  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Hugging Face 'datasets' is not installed. Install with 'pip install datasets' or provide --ratings-file."
        ) from e

    if os.path.isdir(hf_id_or_path):
        ds = datasets.load_from_disk(hf_id_or_path)
        # Expect a split dict when loading a dataset dict; handle both cases
        if isinstance(ds, dict) or hasattr(ds, "keys"):
            ds = ds["train"]
    else:
        ds = datasets.load_dataset(hf_id_or_path)
        ds = ds["train"]

    df = ds.to_pandas()
    # Expected HF schema in this repo uses 'user_id', 'movie_id', 'user_rating'
    cols = {c.lower(): c for c in df.columns}
    mapping = {}
    if "user_id" in cols:
        mapping[cols["user_id"]] = "userId"
    if "movie_id" in cols:
        mapping[cols["movie_id"]] = "movieId"
    if "user_rating" in cols:
        mapping[cols["user_rating"]] = "rating"
    df = df.rename(columns=mapping)
    return normalize_columns(df)


def compute_counts(df: pd.DataFrame) -> Tuple[int, int, int]:
    df = normalize_columns(df)
    num_users = df["userId"].nunique()
    num_movies = df["movieId"].nunique()
    num_ratings = len(df)
    return num_users, num_movies, num_ratings


def main():
    parser = argparse.ArgumentParser(
        description="Count unique users, movies, and total ratings in MovieLens data"
    )
    parser.add_argument(
        "--path",
        help="Path to ratings file or directory containing MovieLens (e.g., ratings.csv/ratings.dat/u.data)",
    )
    parser.add_argument(
        "--hf-id",
        help="Optional Hugging Face dataset id or local saved_to_disk path (e.g., DukeNLPGroup/movielens-100k)",
    )
    args = parser.parse_args()

    if not args.path and not args.hf_id:
        parser.error("Provide --path to a file/dir or --hf-id to load from HF")

    if args.hf_id:
        df = load_from_hf(args.hf_id)
    else:
        ratings_path = infer_ratings_path(args.path)
        df = load_from_file(ratings_path)

    users, movies, ratings = compute_counts(df)
    print(f"users\t{users}")
    print(f"movies\t{movies}")
    print(f"ratings\t{ratings}")


if __name__ == "__main__":
    main()

