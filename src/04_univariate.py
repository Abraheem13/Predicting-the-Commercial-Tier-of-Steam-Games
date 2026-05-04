"""
Stage 4 - Univariate analysis.

Measure skewness for every numeric feature and decide which need log1p
transformation before PCA / clustering / linear models. Decision rule:
|skew_raw| > 2 flags the column for log1p.
"""
# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import DATA_INTERIM, FIGURES

# %% Load
df = pd.read_parquet(DATA_INTERIM / "with_targets.parquet")

numeric_cols = [
    "price", "peak_ccu", "dlc_count", "achievements",
    "recommendations", "positive", "negative",
    "average_playtime_forever", "median_playtime_forever",
    "average_playtime_2weeks", "median_playtime_2weeks",
    "days_since_release",
]

# %% Skew measurements (on positive values for the raw column)
skew_raw = df[numeric_cols].apply(
    lambda x: x[x > 0].skew() if (x > 0).any() else np.nan
).sort_values(ascending=False)
skew_log = df[numeric_cols].apply(
    lambda x: np.log1p(x.clip(lower=0)).skew()
).sort_values(ascending=False)

skew_df = pd.DataFrame({"skew_raw": skew_raw, "skew_log1p": skew_log})
skew_df.to_csv(FIGURES / "04_skew_report.csv")
print("Skew report:")
print(skew_df.round(2))

# %% Distribution grid - raw vs log1p
fig, axes = plt.subplots(len(numeric_cols), 2,
                         figsize=(11, 2.6 * len(numeric_cols)))
for i, col in enumerate(numeric_cols):
    data = df[col].dropna()
    axes[i, 0].hist(data, bins=50, color="steelblue", edgecolor="black")
    axes[i, 0].set_title(f"{col} (raw), skew={skew_raw.get(col, np.nan):.2f}")

    logged = np.log1p(data.clip(lower=0))
    axes[i, 1].hist(logged, bins=50, color="seagreen", edgecolor="black")
    axes[i, 1].set_title(f"log1p({col}), skew={skew_log.get(col, np.nan):.2f}")
plt.tight_layout()
plt.savefig(FIGURES / "04_distributions_raw_vs_log.png", dpi=110)
plt.close()

# %% Persist log-transform decision for later stages
to_log = skew_raw[skew_raw.abs() > 2].index.tolist()
print(f"\nColumns to log1p-transform (|skew|>2): {to_log}")
with open(FIGURES / "04_log_decisions.txt", "w") as f:
    f.write("Columns flagged for log1p transformation (|skew|>2):\n")
    f.write("\n".join(to_log))

# %% Boxplot overview on log scale
fig, axes = plt.subplots(3, 4, figsize=(16, 9))
for ax, col in zip(axes.flat, numeric_cols):
    ax.boxplot(np.log1p(df[col].dropna().clip(lower=0)), vert=True)
    ax.set_title(f"log1p({col})")
plt.tight_layout()
plt.savefig(FIGURES / "04_boxplots_log.png", dpi=120)
plt.close()

print(f"\nFigures written to {FIGURES}")