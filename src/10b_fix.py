"""
Stage 10b - Fixes and re-runs.

Two issues surfaced in the first run of src/10_ml_rf.py:

  1. ROC-AUC one-vs-rest came out below 0.5, which is inconsistent with
     the macro-F1 of 0.749. The cause is a label-ordering mismatch when
     binarising the target. Fixed by using the classifier's own
     `classes_` attribute to align binarisation with predict_proba.

  2. Regression R^2 = 0.9987 is target leakage. sentiment_ratio is
     defined as positive / (positive + negative); both quantities are
     in the feature set. The regressor was reconstructing the target
     arithmetically. Fixed by excluding positive, negative, and
     recommendations (highly collinear with positive) from the
     regression feature set.

This script:
  * Re-evaluates the TUNED classification model with the corrected
    ROC-AUC on the same best_params (no grid search needed again).
  * Completely re-runs the regression with the leakage fix.

It uses the same splits from Stage 9, so nothing upstream changes.
"""
# %% Imports
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score, precision_recall_fscore_support,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.preprocessing import label_binarize

from utils import DATA_PROCESSED, FIGURES

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CLASS_ORDER = ["Low", "Medium", "High", "Very High"]

# Best params found in the Stage 10 grid search. Reusing avoids re-running
# the 10-minute CV — they were already selected by cross-validation so
# re-fitting on the same train split gives an identical classifier.
BEST_CLF_PARAMS = {
    "max_depth": 20,
    "max_features": 0.3,
    "min_samples_leaf": 1,
    "n_estimators": 400,
}

# Features to EXCLUDE from the regression because they computationally
# define the target: sentiment_ratio = positive / (positive + negative).
# `recommendations` correlates 0.74 with positive, so including it would
# partially leak too (it's a post-launch popularity signal).
REG_LEAKY_FEATURES = ["positive", "negative", "recommendations"]


# %% ------------------- Classification: re-score AUC only ------------------
def refit_and_score_clf():
    print("#" * 70)
    print("# CLASSIFICATION: re-score tuned RF with corrected ROC-AUC")
    print("#" * 70)

    X_train = pd.read_parquet(DATA_PROCESSED / "clf_Xtrain.parquet")
    X_test  = pd.read_parquet(DATA_PROCESSED / "clf_Xtest.parquet")
    y_train = pd.read_parquet(DATA_PROCESSED / "clf_ytrain.parquet")["owner_class"]
    y_test  = pd.read_parquet(DATA_PROCESSED / "clf_ytest.parquet")["owner_class"]
    y_train = pd.Categorical(y_train, categories=CLASS_ORDER, ordered=True)
    y_test  = pd.Categorical(y_test,  categories=CLASS_ORDER, ordered=True)

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Refitting tuned RF with best params: {BEST_CLF_PARAMS}")

    model = RandomForestClassifier(
        **BEST_CLF_PARAMS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"Trained in {time.time() - t0:.1f}s")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Align binarisation with the classifier's own class ordering
    classifier_classes = list(model.classes_)
    print(f"Classifier classes_ ordering: {classifier_classes}")

    # proba columns are in `classifier_classes` order; binarise y_test the same way
    y_test_bin = label_binarize(y_test, classes=classifier_classes)

    auc_ovr_macro = roc_auc_score(
        y_test_bin, y_proba, multi_class="ovr", average="macro"
    )
    auc_ovr_weighted = roc_auc_score(
        y_test_bin, y_proba, multi_class="ovr", average="weighted"
    )

    # Per-class one-vs-rest AUC
    per_class_auc = {}
    for i, cls in enumerate(classifier_classes):
        per_class_auc[cls] = roc_auc_score(y_test_bin[:, i], y_proba[:, i])

    # Other metrics (should match the earlier run exactly)
    f1_macro = f1_score(y_test, y_pred, average="macro",    labels=CLASS_ORDER)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", labels=CLASS_ORDER)

    print("\n" + "=" * 60)
    print(" RF tuned (full features) - CORRECTED METRICS")
    print("=" * 60)
    print(f"Macro-F1        : {f1_macro:.4f}")
    print(f"Weighted-F1     : {f1_weighted:.4f}")
    print(f"ROC-AUC OvR macro   : {auc_ovr_macro:.4f}   <-- was reported as 0.459, bug fixed")
    print(f"ROC-AUC OvR weighted: {auc_ovr_weighted:.4f}")
    print(f"\nPer-class one-vs-rest AUC:")
    for cls in CLASS_ORDER:
        print(f"  {cls:10s}: {per_class_auc[cls]:.4f}")

    # Update the summary CSV with corrected AUC
    summary = pd.read_csv(FIGURES / "10_classification_summary.csv")
    summary.loc[summary["Model"] == "RF tuned (full features)",
                "ROC-AUC OvR"] = auc_ovr_macro
    # Also refresh the baseline AUC since the same bug applied
    # (we refit just the baseline quickly here)
    print("\n--- Refitting baseline for corrected AUC ---")
    baseline = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    baseline.fit(X_train, y_train)
    bp = baseline.predict_proba(X_test)
    bin_b = label_binarize(y_test, classes=list(baseline.classes_))
    baseline_auc = roc_auc_score(bin_b, bp, multi_class="ovr", average="macro")
    summary.loc[summary["Model"] == "RF baseline (full features)",
                "ROC-AUC OvR"] = baseline_auc
    print(f"Baseline AUC (corrected): {baseline_auc:.4f}")

    # Regenerate ablation AUCs
    print("\n--- Refitting ablations for corrected AUC ---")

    # Pre-launch ablation
    PRE_LAUNCH_EXACT = {
        "price", "dlc_count", "num_genres", "num_languages", "num_platforms",
        "num_categories", "num_tags", "num_developers", "num_publishers",
        "required_age", "days_since_release", "achievements",
    }
    pl_cols = [c for c in X_train.columns if c in PRE_LAUNCH_EXACT
               or c.startswith(("genre_", "cat_"))]
    rf_pl = RandomForestClassifier(
        **BEST_CLF_PARAMS, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_pl.fit(X_train[pl_cols], y_train)
    pl_proba = rf_pl.predict_proba(X_test[pl_cols])
    bin_pl = label_binarize(y_test, classes=list(rf_pl.classes_))
    pl_auc = roc_auc_score(bin_pl, pl_proba, multi_class="ovr", average="macro")
    summary.loc[summary["Model"] == "RF tuned (pre-launch features only)",
                "ROC-AUC OvR"] = pl_auc
    print(f"Pre-launch AUC (corrected): {pl_auc:.4f}")

    # PCA ablation
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_train)
    Xs_te = scaler.transform(X_test)
    pca = PCA(n_components=24, random_state=RANDOM_STATE)
    Xp_tr = pca.fit_transform(Xs_tr)
    Xp_te = pca.transform(Xs_te)
    rf_pca = RandomForestClassifier(
        **BEST_CLF_PARAMS, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_pca.fit(Xp_tr, y_train)
    pca_proba = rf_pca.predict_proba(Xp_te)
    bin_pca = label_binarize(y_test, classes=list(rf_pca.classes_))
    pca_auc = roc_auc_score(bin_pca, pca_proba, multi_class="ovr", average="macro")
    summary.loc[summary["Model"] == "RF tuned (PCA-24 inputs)",
                "ROC-AUC OvR"] = pca_auc
    print(f"PCA-24 AUC (corrected): {pca_auc:.4f}")

    summary.to_csv(FIGURES / "10_classification_summary.csv", index=False)
    print("\n=== CORRECTED CLASSIFICATION SUMMARY ===")
    print(summary.to_string(index=False))

    # Save per-class AUC for the report
    pd.DataFrame(
        [{"class": c, "auc_ovr": per_class_auc[c]} for c in CLASS_ORDER]
    ).to_csv(FIGURES / "10_rf_tuned_per_class_auc.csv", index=False)

    return summary


# %% ------------------- Regression: re-run without leakage -----------------
def run_regression_no_leakage():
    print("\n" + "#" * 70)
    print("# REGRESSION: sentiment_ratio - leakage-fixed re-run")
    print("#" * 70)

    X_train = pd.read_parquet(DATA_PROCESSED / "reg_Xtrain.parquet")
    X_test  = pd.read_parquet(DATA_PROCESSED / "reg_Xtest.parquet")
    y_train = pd.read_parquet(DATA_PROCESSED / "reg_ytrain.parquet")["sentiment_ratio"]
    y_test  = pd.read_parquet(DATA_PROCESSED / "reg_ytest.parquet")["sentiment_ratio"]

    # Drop features that computationally define the target
    feat_cols = [c for c in X_train.columns if c not in REG_LEAKY_FEATURES]
    X_train = X_train[feat_cols]
    X_test  = X_test [feat_cols]
    print(f"Dropped leaky features: {REG_LEAKY_FEATURES}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"y_train mean={y_train.mean():.3f} std={y_train.std():.3f}")

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 5],
    }
    grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=5, n_jobs=-1, verbose=1,
    )
    t0 = time.time()
    grid.fit(X_train, y_train)
    print(f"Regressor grid search in {time.time() - t0:.1f}s")
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV RMSE: {-grid.best_score_:.4f}")

    model = grid.best_estimator_
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    # Baseline: predict train mean for every test row (tells us what R^2 > 0 really means)
    y_baseline = np.full_like(y_test, y_train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))

    print("\n" + "=" * 60)
    print(" RF Regressor - LEAKAGE FIXED")
    print("=" * 60)
    print(f"Test RMSE      : {rmse:.4f}")
    print(f"Test MAE       : {mae:.4f}")
    print(f"Test R^2       : {r2:.4f}")
    print(f"Mean-baseline RMSE (for context): {baseline_rmse:.4f}")
    print(f"RMSE improvement vs mean baseline: "
          f"{(baseline_rmse - rmse) / baseline_rmse * 100:.1f}%")

    pd.DataFrame([{
        "Model": "RF Regressor (tuned, no-leakage)",
        "RMSE": rmse, "MAE": mae, "R2": r2,
        "Mean_baseline_RMSE": baseline_rmse,
        "Improvement_vs_baseline_pct": (baseline_rmse - rmse) / baseline_rmse * 100,
        "best_params": str(grid.best_params_),
    }]).to_csv(FIGURES / "10_regression_summary.csv", index=False)

    # Residual diagnostics
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pred, residuals, s=3, alpha=0.3, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted sentiment_ratio")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"Residuals vs predicted  (RMSE={rmse:.3f}, R^2={r2:.3f})")
    axes[0].grid(alpha=0.3)

    axes[1].hexbin(y_test, y_pred, gridsize=40, cmap="viridis", mincnt=1)
    lims = [0, 1]
    axes[1].plot(lims, lims, "r--", label="y = y_pred")
    axes[1].set_xlabel("Actual sentiment_ratio")
    axes[1].set_ylabel("Predicted sentiment_ratio")
    axes[1].set_title("Predicted vs actual (leakage-fixed)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "10_regression_diagnostics.png", dpi=130)
    plt.close()

    imp = pd.Series(model.feature_importances_, index=X_train.columns)
    imp = imp.sort_values(ascending=False)
    imp.head(20).to_csv(FIGURES / "10_regression_importance.csv")
    print("\nTop 15 features (regression, impurity):")
    print(imp.head(15).to_string())

    return {"rmse": rmse, "mae": mae, "r2": r2,
            "baseline_rmse": baseline_rmse,
            "best_params": grid.best_params_}


# %% Main
if __name__ == "__main__":
    clf_summary = refit_and_score_clf()
    reg_results = run_regression_no_leakage()

    # Update the consolidated JSON
    with open(FIGURES / "10_summary.json") as f:
        summary = json.load(f)
    # Overwrite regression block
    summary["regression"] = {
        "rmse": reg_results["rmse"],
        "mae":  reg_results["mae"],
        "r2":   reg_results["r2"],
        "mean_baseline_rmse": reg_results["baseline_rmse"],
        "best_params": str(reg_results["best_params"]),
        "note": ("positive, negative, recommendations excluded to prevent "
                 "target leakage: sentiment_ratio = positive / (positive + negative)")
    }
    with open(FIGURES / "10_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nDone. Updated files:")
    print("  reports/figures/10_classification_summary.csv (corrected AUC column)")
    print("  reports/figures/10_regression_summary.csv    (leakage fixed)")
    print("  reports/figures/10_regression_diagnostics.png")
    print("  reports/figures/10_regression_importance.csv")
    print("  reports/figures/10_rf_tuned_per_class_auc.csv")
    print("  reports/figures/10_summary.json")