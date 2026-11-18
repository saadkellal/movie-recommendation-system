"""Popularity, content-based and user-user recommendation baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

MOVIE_COLUMNS = {"movie_id", "title", "genres", "keywords", "cast", "director"}
RATING_COLUMNS = {"user_id", "movie_id", "rating"}


def _validate_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def popularity_recommendations(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    n: int = 10,
    minimum_vote_quantile: float = 0.25,
) -> pd.DataFrame:
    """Rank sufficiently rated movies with a Bayesian weighted mean."""

    _validate_columns(movies, MOVIE_COLUMNS, "movies")
    _validate_columns(ratings, RATING_COLUMNS, "ratings")
    if not 0 <= minimum_vote_quantile <= 1:
        raise ValueError("minimum_vote_quantile must be between 0 and 1")

    stats = ratings.groupby("movie_id")["rating"].agg(["count", "mean"]).reset_index()
    minimum_votes = float(stats["count"].quantile(minimum_vote_quantile))
    global_mean = float(ratings["rating"].mean())
    qualified = stats[stats["count"] >= minimum_votes].copy()
    qualified["score"] = (
        qualified["count"].div(qualified["count"] + minimum_votes) * qualified["mean"]
        + minimum_votes / (qualified["count"] + minimum_votes) * global_mean
    )
    return (
        qualified.merge(movies[["movie_id", "title", "genres"]], on="movie_id", how="left")
        .sort_values(["score", "count", "title"], ascending=[False, False, True])
        .head(n)
        .loc[:, ["movie_id", "title", "genres", "count", "mean", "score"]]
        .reset_index(drop=True)
    )


def _content_soup(movies: pd.DataFrame) -> pd.Series:
    fields = movies[["genres", "keywords", "cast", "director"]].fillna("").astype(str)
    return fields.agg(" ".join, axis=1).str.lower()


def content_recommendations(movies: pd.DataFrame, title: str, n: int = 10) -> pd.DataFrame:
    """Return movies with the most similar metadata to a selected title."""

    _validate_columns(movies, MOVIE_COLUMNS, "movies")
    matching = movies.index[movies["title"].str.casefold() == title.casefold()].tolist()
    if not matching:
        raise KeyError(f"Unknown title: {title}")

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(_content_soup(movies))
    query_index = matching[0]
    scores = linear_kernel(matrix[query_index : query_index + 1], matrix).ravel()
    order = scores.argsort()[::-1]
    order = [index for index in order if index != query_index][:n]

    result = movies.iloc[order][["movie_id", "title", "genres"]].copy()
    result["similarity"] = scores[order]
    return result.reset_index(drop=True)


def user_user_recommendations(
    movies: pd.DataFrame, ratings: pd.DataFrame, user_id: int, n: int = 10, neighbors: int = 10
) -> pd.DataFrame:
    """Recommend unseen titles from cosine-similar users."""

    _validate_columns(movies, MOVIE_COLUMNS, "movies")
    _validate_columns(ratings, RATING_COLUMNS, "ratings")
    matrix = ratings.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)
    if user_id not in matrix.index:
        raise KeyError(f"Unknown user: {user_id}")

    similarity = cosine_similarity(matrix)
    similarity_frame = pd.DataFrame(similarity, index=matrix.index, columns=matrix.index)
    similar_users = similarity_frame[user_id].drop(user_id).sort_values(ascending=False).head(neighbors)
    unseen = matrix.columns[matrix.loc[user_id].eq(0)]

    predictions: list[tuple[int, float]] = []
    for movie_id in unseen:
        neighbor_ratings = matrix.loc[similar_users.index, movie_id]
        mask = neighbor_ratings.gt(0) & similar_users.gt(0)
        if mask.any():
            weights = similar_users[mask]
            score = float((neighbor_ratings[mask] * weights).sum() / weights.sum())
            predictions.append((int(movie_id), score))

    ranked = pd.DataFrame(predictions, columns=["movie_id", "predicted_rating"])
    if ranked.empty:
        return pd.DataFrame(columns=["movie_id", "title", "genres", "predicted_rating"])
    return (
        ranked.merge(movies[["movie_id", "title", "genres"]], on="movie_id", how="left")
        .sort_values(["predicted_rating", "title"], ascending=[False, True])
        .head(n)
        .loc[:, ["movie_id", "title", "genres", "predicted_rating"]]
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movies", type=Path, default=Path("data/demo_movies.csv"))
    parser.add_argument("--ratings", type=Path, default=Path("data/demo_ratings.csv"))
    parser.add_argument("--title", default="Orbit Colony")
    parser.add_argument("--user", type=int, default=1)
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()

    movies = pd.read_csv(args.movies)
    ratings = pd.read_csv(args.ratings)
    print("Popularity")
    print(popularity_recommendations(movies, ratings, args.count).to_string(index=False))
    print(f"\nSimilar to {args.title}")
    print(content_recommendations(movies, args.title, args.count).to_string(index=False))
    print(f"\nFor user {args.user}")
    print(user_user_recommendations(movies, ratings, args.user, args.count).to_string(index=False))


if __name__ == "__main__":
    main()
