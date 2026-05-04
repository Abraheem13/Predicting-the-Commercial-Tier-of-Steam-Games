"""
Stage 7 - PCA (first unsupervised method, for the B-grade rubric).

Applied to the standardised, log-transformed feature matrix to:
  * quantify dimensionality / redundancy  (scree plot)
  * visualise class separability           (PC1 vs PC2 scatter)
  * identify latent factors                (loadings table)
"""
# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils import DATA_INTERIM, FIGURES, LOG_CANDIDATES

# %% Load
df = pd.read_parquet(DATA_INTERIM / "featurised.parquet")

# Build the canonical feature list
base_numeric = [
    "price", "peak_ccu", "dlc_count", "achievements", "recommendations",
    "positive", "negative",
    "num_genres", "num_languages", "num_platforms", "num_categories",
    "num_tags",
    "days_since_release", "average_playtime_forever",
    "has_metacritic", "has_user_score", "has_playtime", "has_reviews",
    "required_age",
]
multi_hot = [c for c in df.columns
             if c.startswith("genre_") or c.startswith("cat_")]
feature_cols = [c for c in base_numeric if c in df.columns] + multi_hot
print(f"Feature matrix width: {len(feature_cols)}")

X = df[feature_cols].copy()
for c in LOG_CANDIDATES:
    if c in X.columns:
        X[c] = np.log1p(X[c].clip(lower=0))
X = X.fillna(0)

y = df["owner_class"]

# %% Scale (required before PCA)
scaler = StandardScaler()
Xs = scaler.fit_transform(X)

# %% Full PCA for scree
pca_full = PCA().fit(Xs)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n80 = int(np.argmax(cum_var >= 0.80) + 1)
n95 = int(np.argmax(cum_var >= 0.95) + 1)
print(f"\nComponents for 80% variance: {n80}")
print(f"Components for 95% variance: {n95}")
print(f"Reduction factor at 80%: {len(feature_cols) / n80:.1f}x")

# %% Scree + 2D projection
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, len(cum_var) + 1), cum_var,
             marker="o", color="steelblue")
axes[0].axhline(0.8, color="r", linestyle="--", label="80% variance")
axes[0].axhline(0.95, color="orange", linestyle="--", label="95% variance")
axes[0].axvline(n80, color="g", linestyle=":", label=f"n={n80} (80%)")
axes[0].set_xlabel("Number of components")
axes[0].set_ylabel("Cumulative explained variance")
axes[0].set_title("Scree plot")
axes[0].legend()
axes[0].grid(alpha=0.3)

pca2 = PCA(n_components=2).fit_transform(Xs)
colours = {"Low": "#4C72B0", "Medium": "#55A868",
           "High": "#C44E52", "Very High": "#8172B2"}
for cls in y.cat.categories:
    m = (y == cls).values
    axes[1].scatter(pca2[m, 0], pca2[m, 1],
                    s=2, alpha=0.3, label=cls,
                    color=colours.get(cls))
axes[1].set_xlabel(f"PC1 ({pca_full.explained_variance_ratio_[0] * 100:.1f}%)")
axes[1].set_ylabel(f"PC2 ({pca_full.explained_variance_ratio_[1] * 100:.1f}%)")
axes[1].set_title("PCA projection coloured by owner class")
axes[1].legend(markerscale=4, loc="best")
plt.tight_layout()
plt.savefig(FIGURES / "07_pca.png", dpi=120)
plt.close()

# %% Loadings - which features drive each PC?
n_pcs_to_show = min(5, pca_full.n_components_)
loadings = pd.DataFrame(
    pca_full.components_[:n_pcs_to_show].T,
    columns=[f"PC{i+1}" for i in range(n_pcs_to_show)],
    index=feature_cols,
)
loadings.to_csv(FIGURES / "07_pca_loadings.csv")

print("\nTop 10 features by |PC1|:")
print(loadings.reindex(loadings["PC1"].abs()
                       .sort_values(ascending=False).index).head(10))
print("\nTop 10 features by |PC2|:")
print(loadings.reindex(loadings["PC2"].abs()
                       .sort_values(ascending=False).index).head(10))

# %% Persist for Stage 8
np.save(DATA_INTERIM / "Xs_scaled.npy", Xs)
pd.Series(feature_cols).to_csv(DATA_INTERIM / "feature_cols.csv", index=False)
with open(DATA_INTERIM / "pca_n80.txt", "w") as f:
    f.write(str(n80))

print(f"\nSaved scaled matrix and feature list to {DATA_INTERIM}")