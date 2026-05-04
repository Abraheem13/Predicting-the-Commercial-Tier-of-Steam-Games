"""
Stage 6 - Bivariate analysis against classification and regression targets.

Outputs:
  - Boxplots: numeric features by owner_class (log1p scale)
  - Correlation heatmap (log-transformed numeric + owners_mid)
  - Hexbin scatter: numeric features vs. sentiment_ratio

Findings here justify feature-selection and multicollinearity treatment
decisions in the PCA stage and in the modelling sections.
"""
# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import DATA_INTERIM, FIGURES

# %% Load
df = pd.read_parquet(DATA_INTERIM / "featurised.parquet")
print(f"Loaded: {df.shape}")

numeric_cols = [
    "price", "peak_ccu", "dlc_count", "achievements", "recommendations",
    "positive", "negative",
    "num_genres", "num_languages", "num_platforms", "num_categories",
    "num_tags",
    "days_since_release", "average_playtime_forever",
]

# %% Boxplots of numeric features by owner_class
fig, axes = plt.subplots(4, 4, figsize=(18, 16))
for ax, col in zip(axes.flat, numeric_cols):
    data = df[[col, "owner_class"]].copy()
    data[col] = np.log1p(data[col].clip(lower=0))
    sns.boxplot(data=data, x="owner_class", y=col, ax=ax,
                showfliers=False, color="steelblue")
    ax.set_title(f"log1p({col}) by owner class")
    ax.tick_params(axis="x", rotation=20)

# Hide any unused axes
for ax in axes.flat[len(numeric_cols):]:
    ax.axis("off")

plt.tight_layout()
plt.savefig(FIGURES / "06_boxplots_by_class.png", dpi=120)
plt.close()

# %% Correlation heatmap (log-transformed)
corr_df = df[numeric_cols].apply(lambda x: np.log1p(x.clip(lower=0)))
corr_df["owners_mid_log"] = np.log1p(df["owners_mid"])
corr = corr_df.corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, vmin=-1, vmax=1)
ax.set_title("Correlation heatmap (log1p features)")
plt.tight_layout()
plt.savefig(FIGURES / "06_correlation_heatmap.png", dpi=120)
plt.close()
corr.to_csv(FIGURES / "06_correlation_matrix.csv")

# %% High-correlation pairs
high = (corr.abs()
        .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack().sort_values(ascending=False))
high = high[high > 0.7]
print("\nHigh-correlation pairs (|r|>0.7):")
print(high)

# %% Regression: hexbin scatter vs sentiment_ratio
reg_df = df.dropna(subset=["sentiment_ratio"])
print(f"\nRegression subset size: {len(reg_df):,}")

fig, axes = plt.subplots(4, 4, figsize=(18, 16))
for ax, col in zip(axes.flat, numeric_cols):
    x = np.log1p(reg_df[col].clip(lower=0))
    ax.hexbin(x, reg_df["sentiment_ratio"],
              gridsize=30, cmap="viridis", mincnt=1)
    ax.set_xlabel(f"log1p({col})")
    ax.set_ylabel("sentiment_ratio")
for ax in axes.flat[len(numeric_cols):]:
    ax.axis("off")
plt.tight_layout()
plt.savefig(FIGURES / "06_scatter_vs_sentiment.png", dpi=120)
plt.close()

# %% Correlations with regression target
reg_corr = pd.Series({
    col: np.log1p(reg_df[col].clip(lower=0)).corr(reg_df["sentiment_ratio"])
    for col in numeric_cols
}).sort_values(key=abs, ascending=False)
print("\nCorrelation with sentiment_ratio:")
print(reg_corr.round(3))
reg_corr.to_csv(FIGURES / "06_sentiment_correlations.csv")

print(f"\nFigures written to {FIGURES}")