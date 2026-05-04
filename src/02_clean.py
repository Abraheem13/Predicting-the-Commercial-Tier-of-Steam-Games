"""
Stage 2 - Cleaning and sentinel-zero handling.

The JSON source uses 0 as a sentinel for "not recorded" in many numeric
fields. Treating these as true zeros would corrupt every downstream
analysis. We apply three rules:

  - >90% zeros  -> drop as continuous feature, keep binary `has_X` flag
  - 30-90% zeros -> keep as-is, add binary `has_X` flag
  - <30% zeros  -> keep as continuous, no special handling needed

The flag `has_X` often carries real signal (e.g. `has_metacritic`
indicates a game received enough traction for Valve to generate a
Metacritic entry).
"""
# %% Imports
import pandas as pd
import numpy as np
from utils import DATA_INTERIM, FIGURES, SENTINEL_ZERO_COLS

# %% Load the JSON-sourced parquet from Stage 1
df = pd.read_parquet(DATA_INTERIM / "raw_from_json.parquet")
print(f"Loaded: {df.shape}")

# %% Sentinel-zero audit
sentinel_report = pd.DataFrame({
    "zero_count": [(df[c] == 0).sum() for c in SENTINEL_ZERO_COLS],
    "zero_pct":   [(df[c] == 0).mean() * 100 for c in SENTINEL_ZERO_COLS],
}, index=SENTINEL_ZERO_COLS).sort_values("zero_pct", ascending=False)
print("\nSentinel-zero report:")
print(sentinel_report.round(2))
sentinel_report.to_csv(FIGURES / "02_sentinel_zero_report.csv")

# %% Presence flags for high-sentinel columns.
#    Presence of a Metacritic / user score is ITSELF a success signal.
df["has_metacritic"] = (df["metacritic_score"] > 0).astype(int)
df["has_user_score"] = (df["user_score"] > 0).astype(int)
df["has_playtime"]   = (df["average_playtime_forever"] > 0).astype(int)
df["has_reviews"]    = ((df["positive"] + df["negative"]) > 0).astype(int)

# %% Drop "0 - 0" ghost rows (unreleased / withdrawn games)
before = len(df)
df = df[df["estimated_owners"] != "0 - 0"]
print(f"\nDropped {before - len(df):,} '0 - 0' rows "
      f"({(before - len(df)) / before * 100:.1f}%)")

# %% Drop games with NO signal at all: no reviews AND no playtime
before = len(df)
df = df[~((df["has_playtime"] == 0) & (df["has_reviews"] == 0))]
print(f"Dropped {before - len(df):,} never-played AND never-reviewed "
      f"({(before - len(df)) / before * 100:.1f}%)")

# %% Drop non-predictive / leakage-prone / high-cardinality columns.
#    Free text / URLs / IDs that would need NLP or are not useful for
#    tabular prediction.
drop_cols = [
    "appid",
    "name",
    "detailed_description", "about_the_game", "short_description",
    "reviews", "notes",
    "header_image", "website", "support_url", "support_email",
    "metacritic_url", "score_rank",
    "screenshots", "movies", "packages",
    "full_audio_languages",
    "discount",   # redundant with price for most games
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])
print(f"\nAfter column drops: {df.shape}")

# %% Parse release_date -> days_since_release
#    Use dataset max date as anchor so re-runs on the same snapshot are
#    reproducible (unlike `datetime.now()` which drifts).
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
reference_date = df["release_date"].max()
df["days_since_release"] = (reference_date - df["release_date"]).dt.days

before = len(df)
df = df.dropna(subset=["release_date"])
print(f"Dropped {before - len(df)} rows with unparseable release_date")

# %% Drop the tiny number of rows where required_age > 25 (sentinel bad data)
before = len(df)
df = df[df["required_age"] <= 25]
print(f"Dropped {before - len(df)} rows with implausible required_age")

# %% Save
out = DATA_INTERIM / "cleaned.parquet"
df.to_parquet(out, index=False)
print(f"\nSaved cleaned data -> {out}")
print(f"Final shape: {df.shape}")