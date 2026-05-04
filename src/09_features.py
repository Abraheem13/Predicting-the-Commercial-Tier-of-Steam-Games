"""
Stage 9 - Final feature matrix and train/test splits.

Exports artefacts consumed by the ML and DL modelling scripts:

  data/processed/
    clf_Xtrain.parquet  clf_Xtest.parquet
    clf_ytrain.parquet  clf_ytest.parquet
    reg_Xtrain.parquet  reg_Xtest.parquet
    reg_ytrain.parquet  reg_ytest.parquet
    feature_cols.txt

Design choices:
  * 80/20 split, random_state=42 for reproducibility
  * STRATIFIED split on owner_class for classification (heavy imbalance)
  * log1p is applied to skewed features BEFORE split so the train/test
    transformation is identical. Scaling is NOT done here - it belongs
    in the modelling pipeline where the scaler is fit on train only.
"""
# %% Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils import DATA_INTERIM, DATA_PROCESSED, LOG_CANDIDATES

# %% Load
df = pd.read_parquet(DATA_INTERIM / "featurised.parquet")

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

# %% Apply log1p to skewed features
X_full = df[feature_cols].copy()
for c in LOG_CANDIDATES:
    if c in X_full.columns:
        X_full[c] = np.log1p(X_full[c].clip(lower=0))
X_full = X_full.fillna(0)

# %% Classification split (stratified)
clf_mask = df["owner_class"].notna()
Xc = X_full[clf_mask]
yc = df.loc[clf_mask, "owner_class"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, stratify=yc, random_state=42,
)
print(f"Classification:  train={len(Xc_train):,}  test={len(Xc_test):,}")
print("Train class distribution:")
print(yc_train.value_counts(normalize=True).sort_index().round(3))

Xc_train.to_parquet(DATA_PROCESSED / "clf_Xtrain.parquet")
Xc_test .to_parquet(DATA_PROCESSED / "clf_Xtest.parquet")
# Cast categorical -> string for parquet compatibility
yc_train.astype(str).to_frame("owner_class").to_parquet(
    DATA_PROCESSED / "clf_ytrain.parquet")
yc_test.astype(str).to_frame("owner_class").to_parquet(
    DATA_PROCESSED / "clf_ytest.parquet")

# %% Regression split
reg_mask = df["sentiment_ratio"].notna()
Xr = X_full[reg_mask]
yr = df.loc[reg_mask, "sentiment_ratio"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    Xr, yr, test_size=0.2, random_state=42,
)
print(f"\nRegression:      train={len(Xr_train):,}  test={len(Xr_test):,}")

Xr_train.to_parquet(DATA_PROCESSED / "reg_Xtrain.parquet")
Xr_test .to_parquet(DATA_PROCESSED / "reg_Xtest.parquet")
yr_train.to_frame("sentiment_ratio").to_parquet(
    DATA_PROCESSED / "reg_ytrain.parquet")
yr_test.to_frame("sentiment_ratio").to_parquet(
    DATA_PROCESSED / "reg_ytest.parquet")

# %% Persist feature list
with open(DATA_PROCESSED / "feature_cols.txt", "w") as f:
    f.write("\n".join(feature_cols))

print(f"\nAll splits written to {DATA_PROCESSED}")
print(f"Feature count: {len(feature_cols)}")