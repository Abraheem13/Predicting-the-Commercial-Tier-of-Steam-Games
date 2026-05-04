"""
Stage 8 - K-Means clustering (second unsupervised method).

Uses the same scaled feature matrix as Stage 7 (persisted to disk).

Flow:
  1. Elbow + silhouette sweep over k in [2, 10]
  2. Fit at K_FINAL (edit after reviewing 08_k_selection.png)
  3. Profile clusters by mean feature values
  4. Cross-tab clusters against owner_class - do unsupervised
     structures recover commercial tier?
  5. Plot clusters in PC1-PC2 space for visual comparison with Stage 7.
"""
# %% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from utils import DATA_INTERIM, FIGURES

K_FINAL = 5   # review 08_k_selection.png and update this if the elbow is elsewhere

# %% Load
Xs = np.load(DATA_INTERIM / "Xs_scaled.npy")
feature_cols = pd.read_csv(DATA_INTERIM / "feature_cols.csv").iloc[:, 0].tolist()
df = pd.read_parquet(DATA_INTERIM / "featurised.parquet").reset_index(drop=True)
print(f"Scaled matrix: {Xs.shape}")
print(f"Source df:     {df.shape}")
assert len(df) == len(Xs), "Row count mismatch between df and Xs"

# %% Elbow + silhouette sweep
inertias, silhouettes = [], []
K_RANGE = range(2, 11)
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(Xs), size=min(10_000, len(Xs)), replace=False)

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(Xs[sample_idx], km.labels_[sample_idx]))
    print(f"k={k}: inertia={km.inertia_:.0f}  silhouette={silhouettes[-1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(list(K_RANGE), inertias, marker="o", color="steelblue")
axes[0].set_title("Elbow plot (inertia)")
axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
axes[0].grid(alpha=0.3)

axes[1].plot(list(K_RANGE), silhouettes, marker="o", color="seagreen")
axes[1].set_title("Silhouette score (10k sample)")
axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES / "08_k_selection.png", dpi=120)
plt.close()

pd.DataFrame({"k": list(K_RANGE),
              "inertia": inertias,
              "silhouette": silhouettes}).to_csv(
    FIGURES / "08_k_selection.csv", index=False
)

# %% Fit final k
print(f"\nFitting final KMeans with k={K_FINAL}")
km = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10).fit(Xs)
df["cluster"] = km.labels_

sizes = pd.Series(km.labels_).value_counts().sort_index()
print("\nCluster sizes:")
print(sizes)

# %% Cluster profiles (mean of SCALED features)
X_df = pd.DataFrame(Xs, columns=feature_cols)
X_df["cluster"] = km.labels_
profile = X_df.groupby("cluster").mean().T
profile.to_csv(FIGURES / "08_cluster_profiles.csv")

print("\nTop distinguishing features per cluster:")
for c in range(K_FINAL):
    top = profile[c].abs().sort_values(ascending=False).head(5)
    print(f"\nCluster {c}:")
    print(profile.loc[top.index, c].round(3))

# %% Cross-tab clusters vs owner_class
ct = pd.crosstab(df["cluster"], df["owner_class"], normalize="index")
print("\nRow-normalised cross-tab (cluster -> owner_class share):")
print(ct.round(3))
ct.to_csv(FIGURES / "08_cluster_vs_class.csv")

fig, ax = plt.subplots(figsize=(9, 5))
ct.plot(kind="bar", stacked=True, ax=ax, colormap="viridis",
        edgecolor="black")
ax.set_title("Owner-class composition of each cluster")
ax.set_ylabel("Proportion of cluster")
ax.set_xlabel("Cluster")
ax.legend(title="Owner class", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES / "08_cluster_vs_class.png", dpi=120)
plt.close()

# %% Clusters on PC1-PC2
pca2 = PCA(n_components=2).fit_transform(Xs)
fig, ax = plt.subplots(figsize=(9, 7))
cmap = plt.cm.get_cmap("tab10")
for c in range(K_FINAL):
    m = km.labels_ == c
    ax.scatter(pca2[m, 0], pca2[m, 1], s=2, alpha=0.3,
               label=f"Cluster {c}", color=cmap(c))
ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
ax.set_title(f"K-Means clusters (k={K_FINAL}) in PC1-PC2 space")
ax.legend(markerscale=4)
plt.tight_layout()
plt.savefig(FIGURES / "08_clusters_pca.png", dpi=120)
plt.close()

# %% Save
df.to_parquet(DATA_INTERIM / "with_clusters.parquet", index=False)
print(f"\nSaved clustered df to {DATA_INTERIM / 'with_clusters.parquet'}")