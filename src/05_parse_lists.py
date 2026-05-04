"""
Stage 5 - Parse list-type columns into modelling features.

When loaded from JSON and saved to parquet, list columns (genres,
categories, supported_languages, tags, developers, publishers) are
JSON-encoded strings. We parse them back, then:

  1. Derive cardinality features (num_genres, num_languages, ...)
  2. Multi-hot encode the top-N most frequent genres and categories.

Top-N is chosen empirically via a cumulative coverage plot (default 15).
"""
# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import Counter
from utils import DATA_INTERIM, FIGURES

TOP_N_GENRES = 15
TOP_N_CATEGORIES = 15

# %% Load
df = pd.read_parquet(DATA_INTERIM / "with_targets.parquet")
print(f"Loaded: {df.shape}")

# %% Parse JSON-encoded list columns back into Python lists
list_source_cols = ["genres", "categories", "supported_languages",
                    "tags", "developers", "publishers"]

def parse_list(x):
    """Turn a JSON-encoded list string back into a list.
    Tolerates None, empty string, and already-list inputs."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() == "null":
            return []
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return v
            if isinstance(v, dict):
                return list(v.keys())    # tags sometimes come as {name: weight}
            return [str(v)]
        except Exception:
            # Fall back: comma-split
            return [p.strip() for p in s.split(",") if p.strip()]
    return []

for col in list_source_cols:
    if col in df.columns:
        df[col + "_list"] = df[col].map(parse_list)
    else:
        print(f"  (note: column '{col}' missing; skipping)")
        df[col + "_list"] = [[] for _ in range(len(df))]

# %% Cardinality features
df["num_genres"]     = df["genres_list"].str.len()
df["num_categories"] = df["categories_list"].str.len()
df["num_languages"]  = df["supported_languages_list"].str.len()
df["num_tags"]       = df["tags_list"].str.len()
df["num_developers"] = df["developers_list"].str.len()
df["num_publishers"] = df["publishers_list"].str.len()
df["num_platforms"]  = df[["windows", "mac", "linux"]].sum(axis=1)

# %% Genre coverage plot to justify top-N cutoff
genre_counts = Counter(g for gs in df["genres_list"] for g in gs)
genre_series = pd.Series(genre_counts).sort_values(ascending=False)

if len(genre_series) > 0:
    cum_coverage = genre_series.cumsum() / genre_series.sum()
    n_show = min(30, len(genre_series))
    fig, ax = plt.subplots(figsize=(10, 4))
    cum_coverage.head(n_show).plot(ax=ax, marker="o", color="steelblue")
    ax.axhline(0.95, color="r", linestyle="--", label="95% coverage")
    ax.axvline(min(TOP_N_GENRES, n_show) - 1, color="g", linestyle="--",
               label=f"Top-{TOP_N_GENRES} cutoff")
    ax.set_title("Cumulative genre coverage vs. top-N")
    ax.set_ylabel("Cumulative share of genre occurrences")
    plt.xticks(rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "05_genre_coverage.png", dpi=120)
    plt.close()

top_genres = genre_series.head(TOP_N_GENRES).index.tolist()
print(f"\nTop {len(top_genres)} genres: {top_genres}")
for g in top_genres:
    safe = (str(g).replace(" ", "_").replace("-", "_")
            .replace("&", "and").replace("/", "_"))
    df[f"genre_{safe}"] = df["genres_list"].map(lambda gs, g=g: int(g in gs))

# %% Categories the same way
cat_counts = Counter(c for cs in df["categories_list"] for c in cs)
cat_series = pd.Series(cat_counts).sort_values(ascending=False)
top_cats = cat_series.head(TOP_N_CATEGORIES).index.tolist()
print(f"\nTop {len(top_cats)} categories: {top_cats}")
for c in top_cats:
    safe = (str(c).replace(" ", "_").replace("-", "_")
            .replace("&", "and").replace("/", "_"))
    df[f"cat_{safe}"] = df["categories_list"].map(lambda cs, c=c: int(c in cs))

# %% Quick plot: top genres + top categories
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
if len(genre_series) > 0:
    genre_series.head(15).plot(kind="barh", ax=axes[0], color="steelblue")
    axes[0].set_title("Top 15 genres (frequency)")
    axes[0].invert_yaxis()
if len(cat_series) > 0:
    cat_series.head(15).plot(kind="barh", ax=axes[1], color="seagreen")
    axes[1].set_title("Top 15 categories (frequency)")
    axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig(FIGURES / "05_top_lists.png", dpi=120)
plt.close()

# %% Drop the raw list columns (parquet struggles with ragged nested lists)
df = df.drop(columns=[c for c in df.columns if c.endswith("_list")])
# And drop the original string versions of list columns - we no longer need them
df = df.drop(columns=[c for c in list_source_cols if c in df.columns])

# %% Save
out = DATA_INTERIM / "featurised.parquet"
df.to_parquet(out, index=False)
print(f"\nSaved -> {out}")
print(f"Final shape: {df.shape}")
print(f"Added {len(top_genres) + len(top_cats)} multi-hot columns "
      f"+ 7 cardinality features.")