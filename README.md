# Movie Recommendation System

Portfolio project implementing three recommendation baselines: weighted popularity, content similarity and user-user collaborative filtering.

## Overview

The original Google Colab notebook was located and preserved in `notebooks/`. It targets The Movies Dataset/TMDB-style files (`movies_metadata.csv`, `ratings_small.csv`, `credits.csv`, `keywords.csv` and `links_small.csv`). Those external datasets are not included because redistribution rights were not established during the audit.

A small, tested Python module and clearly synthetic demonstration dataset are included so recruiters can run the core algorithms without Google Drive or private files.

## Implemented approaches

### Weighted popularity

Ranks movies with a Bayesian weighted mean using rating count, movie mean rating and the catalogue-wide mean. A vote-count quantile prevents single-rating titles from dominating the list.

### Content-based similarity

Builds a TF-IDF representation from genres, keywords, cast and director fields, then returns the highest cosine-similarity titles for a selected movie.

### User-user collaborative filtering

Creates a user-item matrix, finds cosine-similar users and predicts ratings for unseen movies with a similarity-weighted average.

## Repository structure

```text
.
├── data/
│   ├── demo_movies.csv       # Synthetic, redistributable catalogue
│   └── demo_ratings.csv      # Synthetic, redistributable interactions
├── notebooks/
│   └── movie_recommendation_system.ipynb
├── src/
│   └── recommender.py
├── tests/
│   └── test_recommender.py
└── requirements.txt
```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the demo

```powershell
python -m src.recommender --title "Orbit Colony" --user 1 --count 5
```

Run the tests:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Notebook audit

The original notebook contains 21 code cells and no Markdown cells. Only the first two cells have execution counts, so it does not contain reproducible model outputs or verified metrics.

The source implements the three approaches above, plus popularity RMSE/MAE, coverage reporting, rating-level K-fold analysis and visualizations. The following issues should be fixed before treating notebook metrics as valid:

- It uses Python `eval` to parse dataset strings; use `ast.literal_eval` or structured JSON parsing instead.
- The user-item matrix and similarity matrix are created from all ratings before the train/test evaluation, leaking test interactions into collaborative predictions.
- A random rating-level split can place the same users and movies in train and test; user/time-aware evaluation is preferable.
- Content coverage is reported, but ranking quality is not measured with precision@k, recall@k or NDCG@k.
- Full dense similarity matrices may become expensive as the catalogue or user count grows.

No performance metric from the notebook is claimed in this README.

## Demo data

The included titles, people and ratings are fictional and created only to exercise the algorithms. They are not TMDB records and do not represent real user behavior.

## Technical contribution

This portfolio draft preserves the original coursework notebook, documents the algorithmic approach and its evaluation limitations, and adds a safe, dependency-light implementation with deterministic unit tests and a redistributable demonstration dataset.

## Limitations and next steps

- Rebuild collaborative evaluation with train-only matrices and per-user holdout.
- Add precision@k, recall@k, MAP@k and NDCG@k.
- Replace dense pairwise matrices with nearest-neighbor search for larger data.
- Add cold-start handling and clearer duplicate-title behavior.
- Re-run a cleaned notebook top-to-bottom on a legally obtained dataset.

## Suggested topics

`python` · `pandas` · `scikit-learn` · `recommendation-system` · `collaborative-filtering` · `content-based-filtering` · `cosine-similarity` · `tfidf`

## License

Original source code and original project documentation are available under
the [MIT License](LICENSE). The license does not cover MovieLens, TMDB, The
Movies Dataset, downloaded metadata or ratings, artwork, trademarks, or other
third-party material; see [LICENSE_SCOPE.md](LICENSE_SCOPE.md). Third-party
datasets are not distributed here.
