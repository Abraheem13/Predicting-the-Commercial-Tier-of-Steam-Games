"""
Stage 3 - Target construction.

Classification target: collapse Steam's `estimated_owners` string bucket
  (e.g. "20000 - 50000") into 4 ordered commercial tiers
  {Low, Medium, High, Very High}.

Regression target: sentiment_ratio = positive / (positive + negative),
  restricted to games with total_reviews >= MIN_REVIEWS.
"""
# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import DATA_INTERIM, FIGURES

MIN_REVIEWS = 50

# %% Load
df = pd.read_parquet(DATA_INTERIM / "cleaned.parquet")
print(f"Loaded: {df.shape}")

# %% Parse "20000 - 50000" -> (lo, hi)
def parse_owner_range(s):
    if pd.isna(s):
        return np.nan, np.nan
    try:
        lo, hi = s.replace(",", "").split(" - ")
        return int(lo.strip()), int(hi.strip())
    except Exception:
        return np.nan, np.nan

parsed = df["estimated_owners"].map(parse_owner_range)
df["owners_lo"]  = parsed.map(lambda t: t[0])
df["owners_hi"]  = parsed.map(lambda t: t[1])
df["owners_mid"] = (df["owners_lo"] + df["owners_hi"]) / 2

df = df.dropna(subset=["owners_lo"])

# %% Raw ordered categorical (all Steam buckets)
bucket_order = (df.groupby("estimated_owners")["owners_lo"]
                  .first().sort_values().index.tolist())
df["owner_bucket"] = pd.Categorical(
    df["estimated_owners"], categories=bucket_order, ordered=True
)

# %% Collapse to 4 commercial tiers
def collapse(lo):
    if lo < 20_000:     return "Low"
    if lo < 100_000:    return "Medium"
    if lo < 1_000_000:  return "High"
    return "Very High"

df["owner_class"] = df["owners_lo"].map(collapse)
df["owner_class"] = pd.Categorical(
    df["owner_class"],
    categories=["Low", "Medium", "High", "Very High"],
    ordered=True,
)

# %% Class distribution plot (collapsed)
fig, ax = plt.subplots(figsize=(8, 4))
counts = df["owner_class"].value_counts().sort_index()
counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
ax.set_title("Class distribution - collapsed owner_class")
ax.set_ylabel("Number of games")
ax.set_xlabel("Owner class")
for i, v in enumerate(counts.values):
    ax.text(i, v, f"{v:,}\n({v / len(df) * 100:.1f}%)",
            ha="center", va="bottom", fontsize=9)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES / "03_class_distribution.png", dpi=150)
plt.close()

print("\nCollapsed class distribution:")
print(counts)
print(f"Imbalance ratio (max/min): {counts.max() / counts.min():.1f}")

# %% Raw bucket distribution (all 10 Steam tiers)
fig, ax = plt.subplots(figsize=(12, 4))
df["owner_bucket"].value_counts().sort_index().plot(
    kind="bar", ax=ax, color="steelblue", edgecolor="black"
)
ax.set_title("Raw Steam buckets - estimated_owners")
ax.set_ylabel("Number of games")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(FIGURES / "03_raw_bucket_distribution.png", dpi=150)
plt.close()

# %% Regression target
df["total_reviews"] = df["positive"] + df["negative"]
mask = df["total_reviews"] >= MIN_REVIEWS
df["sentiment_ratio"] = np.nan
df.loc[mask, "sentiment_ratio"] = (
    df.loc[mask, "positive"] / df.loc[mask, "total_reviews"]
)
print(f"\nRegression retention at MIN_REVIEWS={MIN_REVIEWS}: "
      f"{mask.sum():,} / {len(df):,} games "
      f"({mask.mean() * 100:.1f}%)")

fig, ax = plt.subplots(figsize=(8, 4))
df["sentiment_ratio"].dropna().hist(bins=40, ax=ax,
                                    color="steelblue", edgecolor="black")
ax.set_title(f"Distribution of sentiment_ratio (n={int(mask.sum()):,})")
ax.set_xlabel("positive / (positive + negative)")
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES / "03_sentiment_distribution.png", dpi=150)
plt.close()

# %% Save
out = DATA_INTERIM / "with_targets.parquet"
df.to_parquet(out, index=False)
print(f"\nSaved -> {out}")
print(f"Shape: {df.shape}")