from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.recommender import (
    content_recommendations,
    popularity_recommendations,
    user_user_recommendations,
)

ROOT = Path(__file__).resolve().parents[1]


class RecommenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.movies = pd.read_csv(ROOT / "data/demo_movies.csv")
        cls.ratings = pd.read_csv(ROOT / "data/demo_ratings.csv")

    def test_content_results_exclude_the_query(self) -> None:
        result = content_recommendations(self.movies, "Orbit Colony", n=3)
        self.assertNotIn("Orbit Colony", result["title"].tolist())
        self.assertTrue({"Red Planet Relay", "Quantum Harbor"}.intersection(result["title"]))
        self.assertTrue(result["similarity"].between(0, 1).all())

    def test_popularity_result_is_ranked(self) -> None:
        result = popularity_recommendations(self.movies, self.ratings, n=4)
        self.assertEqual(len(result), 4)
        self.assertTrue(result["score"].is_monotonic_decreasing)

    def test_collaborative_results_exclude_rated_movies(self) -> None:
        result = user_user_recommendations(self.movies, self.ratings, user_id=1, n=4)
        rated = set(self.ratings.loc[self.ratings["user_id"] == 1, "movie_id"])
        self.assertTrue(set(result["movie_id"]).isdisjoint(rated))
        self.assertIn("Red Planet Relay", result["title"].tolist())


if __name__ == "__main__":
    unittest.main()
