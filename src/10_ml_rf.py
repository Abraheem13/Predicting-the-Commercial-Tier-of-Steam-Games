"""
Stage 10 - Random Forest modelling (classification + regression).

Classification (primary task):
  * RF classifier on owner_class (4-way, imbalanced 43.9:1)
  * 5-fold stratified CV with GridSearchCV on a small, defensible grid
  * Ablations:
      A. Full features (baseline + tuned)
      B. Pre-launch-only features (addresses the leakage flag from EDA)
      C. PCA-reduced inputs (n_components=24, captures 80% variance)
  * Reported metrics: macro-F1, weighted-F1, per-class precision/recall/F1,
    confusion matrix, ROC-AUC one-vs-rest
  * Interpretability: impurity-based + permutation-based feature importance

Regression (secondary task):
  * RF regressor on sentiment_ratio (continuous, ~30.6k rows)
  * Metrics: RMSE, MAE, R^2, residual plot

All artefacts are written to reports/figures/ so they can be referenced
directly in the final report. Random seeds are fixed throughout.
"""
# %% Imports
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, roc_auc_score, precision_recall_fscore_support,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight

from utils import DATA_PROCESSED, FIGURES

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Pre-launch feature identifier: anything known at release day, BEFORE
# reviews / playtime / CCU accumulate. Everything else is post-launch.
PRE_LAUNCH_ALLOWED_PREFIXES = ("genre_", "cat_")
PRE_LAUNCH_ALLOWED_EXACT = {
    "price", "dlc_count",
    "num_genres", "num_languages", "num_platforms", "num_categories",
    "num_tags", "num_developers", "num_publishers",
    "required_age", "days_since_release",
    # achievements: arguably pre-launch (announced at release). Include it.
    "achievements",
}

CLASS_ORDER = ["Low", "Medium", "High", "Very High"]


# %% Load the processed splits
def load_clf():
    X_train = pd.read_parquet(DATA_PROCESSED / "clf_Xtrain.parquet")
    X_test  = pd.read_parquet(DATA_PROCESSED / "clf_Xtest.parquet")
    y_train = pd.read_parquet(DATA_PROCESSED / "clf_ytrain.parquet")["owner_class"]
    y_test  = pd.read_parquet(DATA_PROCESSED / "clf_ytest.parquet")["owner_class"]
    # Preserve class ordering for reporting
    y_train = pd.Categorical(y_train, categories=CLASS_ORDER, ordered=True)
    y_test  = pd.Categorical(y_test,  categories=CLASS_ORDER, ordered=True)
    return X_train, X_test, y_train, y_test


def load_reg():
    X_train = pd.read_parquet(DATA_PROCESSED / "reg_Xtrain.parquet")
    X_test  = pd.read_parquet(DATA_PROCESSED / "reg_Xtest.parquet")
    y_train = pd.read_parquet(DATA_PROCESSED / "reg_ytrain.parquet")["sentiment_ratio"]
    y_test  = pd.read_parquet(DATA_PROCESSED / "reg_ytest.parquet")["sentiment_ratio"]
    return X_train, X_test, y_train, y_test


# %% Evaluation helpers
def evaluate_classification(name, model, X_test, y_test, class_order=CLASS_ORDER):
    """Compute metrics, return dict, print summary."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    f1_macro    = f1_score(y_test, y_pred, average="macro",    labels=class_order)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", labels=class_order)

    # ROC-AUC one-vs-rest (requires numeric labels)
    y_test_bin = label_binarize(y_test, classes=class_order)
    try:
        auc_ovr = roc_auc_score(y_test_bin, y_proba, multi_class="ovr",
                                average="macro")
    except ValueError:
        auc_ovr = np.nan

    p, r, f, s = precision_recall_fscore_support(
        y_test, y_pred, labels=class_order, zero_division=0
    )
    per_class = pd.DataFrame({
        "class": class_order, "precision": p, "recall": r, "f1": f, "support": s
    })

    print(f"\n{'=' * 60}")
    print(f" {name}")
    print(f"{'=' * 60}")
    print(f"Macro-F1    : {f1_macro:.4f}")
    print(f"Weighted-F1 : {f1_weighted:.4f}")
    print(f"ROC-AUC OvR : {auc_ovr:.4f}")
    print("\nPer-class:")
    print(per_class.to_string(index=False))

    return {
        "name": name, "f1_macro": f1_macro, "f1_weighted": f1_weighted,
        "roc_auc_ovr": auc_ovr, "per_class": per_class,
        "y_pred": y_pred, "y_proba": y_proba,
    }


def save_confusion_matrix(name, y_test, y_pred, filename):
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax,
                cbar_kws={"label": "Row-normalised proportion"})
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix (row-normalised) - {name}")
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=130)
    plt.close()


# %% Classification: baseline
def run_classification():
    print("\n" + "#" * 70)
    print("# CLASSIFICATION: owner_class (4-way)")
    print("#" * 70)

    X_train, X_test, y_train, y_test = load_clf()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Train class distribution:")
    print(pd.Series(y_train).value_counts().reindex(CLASS_ORDER))

    results = []

    # -------- Baseline: default RF with class_weight --------
    print("\n--- Baseline RF (default params, class_weight='balanced') ---")
    t0 = time.time()
    baseline = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    baseline.fit(X_train, y_train)
    print(f"Trained in {time.time() - t0:.1f}s")
    res_baseline = evaluate_classification(
        "RF baseline (full features)", baseline, X_test, y_test
    )
    save_confusion_matrix("RF baseline", y_test, res_baseline["y_pred"],
                          "10_cm_rf_baseline.png")
    results.append(res_baseline)

    # -------- Tuned: small defensible grid --------
    # We deliberately keep the grid small to keep runtime reasonable
    # and because Random Forest is forgiving of hyperparameters.
    print("\n--- Tuned RF via 5-fold stratified CV + GridSearchCV ---")
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 20, 40],
        "min_samples_leaf": [1, 5],
        "max_features": ["sqrt", 0.3],
    }
    # ~24 combinations * 5 folds = 120 fits. On 68k rows this takes ~10-15 min.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,    # parallelism goes to GridSearchCV instead
        ),
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    t0 = time.time()
    grid.fit(X_train, y_train)
    print(f"Grid search completed in {time.time() - t0:.1f}s")
    print(f"Best params    : {grid.best_params_}")
    print(f"Best CV macro-F1: {grid.best_score_:.4f}")

    tuned = grid.best_estimator_
    res_tuned = evaluate_classification(
        "RF tuned (full features)", tuned, X_test, y_test
    )
    save_confusion_matrix("RF tuned", y_test, res_tuned["y_pred"],
                          "10_cm_rf_tuned.png")
    results.append(res_tuned)

    # Persist CV results for the report
    cv_df = pd.DataFrame(grid.cv_results_)
    keep = ["params", "mean_test_score", "std_test_score",
            "mean_train_score", "std_train_score", "mean_fit_time"]
    cv_df[keep].sort_values("mean_test_score", ascending=False).to_csv(
        FIGURES / "10_rf_cv_results.csv", index=False
    )

    # -------- Ablation B: pre-launch features only --------
    pre_launch_cols = [c for c in X_train.columns
                       if c in PRE_LAUNCH_ALLOWED_EXACT
                       or c.startswith(PRE_LAUNCH_ALLOWED_PREFIXES)]
    print(f"\n--- Ablation B: pre-launch features only ({len(pre_launch_cols)} cols) ---")
    print(f"Dropped {X_train.shape[1] - len(pre_launch_cols)} post-launch features")
    Xp_train = X_train[pre_launch_cols]
    Xp_test  = X_test [pre_launch_cols]

    rf_prelaunch = RandomForestClassifier(
        **grid.best_params_,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_prelaunch.fit(Xp_train, y_train)
    res_prelaunch = evaluate_classification(
        "RF tuned (pre-launch features only)",
        rf_prelaunch, Xp_test, y_test
    )
    save_confusion_matrix("RF pre-launch", y_test, res_prelaunch["y_pred"],
                          "10_cm_rf_prelaunch.png")
    results.append(res_prelaunch)

    # -------- Ablation C: PCA-reduced inputs (n=24, 80% variance) --------
    print("\n--- Ablation C: PCA-reduced inputs (n_components=24) ---")
    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_test  = scaler.transform(X_test)

    pca = PCA(n_components=24, random_state=RANDOM_STATE)
    Xp_train = pca.fit_transform(Xs_train)
    Xp_test  = pca.transform(Xs_test)
    print(f"PCA retained variance: {pca.explained_variance_ratio_.sum():.3f}")

    rf_pca = RandomForestClassifier(
        **grid.best_params_,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_pca.fit(Xp_train, y_train)
    res_pca = evaluate_classification(
        "RF tuned (PCA-24 inputs)", rf_pca, Xp_test, y_test
    )
    save_confusion_matrix("RF PCA-24", y_test, res_pca["y_pred"],
                          "10_cm_rf_pca.png")
    results.append(res_pca)

    # -------- Summary table across ablations --------
    summary = pd.DataFrame([{
        "Model": r["name"],
        "Macro-F1": r["f1_macro"],
        "Weighted-F1": r["f1_weighted"],
        "ROC-AUC OvR": r["roc_auc_ovr"],
    } for r in results])
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
    summary.to_csv(FIGURES / "10_classification_summary.csv", index=False)

    # Per-class metrics for the tuned model (used in the report)
    res_tuned["per_class"].to_csv(
        FIGURES / "10_rf_tuned_per_class.csv", index=False
    )
    res_prelaunch["per_class"].to_csv(
        FIGURES / "10_rf_prelaunch_per_class.csv", index=False
    )

    # -------- Feature importance on the TUNED full-feature model --------
    print("\n--- Feature importance (tuned model) ---")

    # A. Impurity-based (fast but biased toward high-cardinality features)
    imp_impurity = pd.Series(
        tuned.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)
    imp_impurity.to_csv(FIGURES / "10_importance_impurity.csv")
    print("\nTop 15 by impurity importance:")
    print(imp_impurity.head(15).to_string())

    # B. Permutation-based (slower, unbiased, better for the report)
    # Use a subsample of the test set for speed; 5 repeats is standard.
    print("\nComputing permutation importance (~2-5 min)...")
    rng = np.random.RandomState(RANDOM_STATE)
    sub_idx = rng.choice(len(X_test), size=min(5000, len(X_test)), replace=False)
    perm = permutation_importance(
        tuned, X_test.iloc[sub_idx], y_test[sub_idx],
        n_repeats=5, random_state=RANDOM_STATE,
        n_jobs=-1, scoring="f1_macro",
    )
    imp_perm = pd.DataFrame({
        "feature": X_train.columns,
        "importance_mean": perm.importances_mean,
        "importance_std":  perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    imp_perm.to_csv(FIGURES / "10_importance_permutation.csv", index=False)
    print("\nTop 15 by permutation importance (drop in macro-F1):")
    print(imp_perm.head(15).to_string(index=False))

    # Plot top 20 permutation-importance features
    top = imp_perm.head(20).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(top["feature"], top["importance_mean"],
            xerr=top["importance_std"], color="steelblue", edgecolor="black")
    ax.set_xlabel("Mean decrease in macro-F1 when feature is permuted")
    ax.set_title("Top 20 features by permutation importance (RF tuned)")
    plt.tight_layout()
    plt.savefig(FIGURES / "10_importance_permutation.png", dpi=130)
    plt.close()

    return results, grid.best_params_


# %% Regression: sentiment_ratio
def run_regression():
    print("\n" + "#" * 70)
    print("# REGRESSION: sentiment_ratio")
    print("#" * 70)

    X_train, X_test, y_train, y_test = load_reg()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"y_train stats: mean={y_train.mean():.3f}, std={y_train.std():.3f}")

    # Small grid for the regressor too (smaller than classification grid
    # because regression on 24k rows is faster and RF is robust).
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 20],
        "min_samples_leaf": [1, 5],
    }
    grid = GridSearchCV(
        estimator=RandomForestRegressor(
            random_state=RANDOM_STATE, n_jobs=1
        ),
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
    print(f"\nTest RMSE: {rmse:.4f}")
    print(f"Test MAE : {mae:.4f}")
    print(f"Test R^2 : {r2:.4f}")

    pd.DataFrame([{
        "Model": "RF Regressor (tuned)",
        "RMSE": rmse, "MAE": mae, "R2": r2,
        "best_params": str(grid.best_params_),
    }]).to_csv(FIGURES / "10_regression_summary.csv", index=False)

    # Residual plot
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_pred, residuals, s=3, alpha=0.3, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted sentiment_ratio")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals vs predicted")
    axes[0].grid(alpha=0.3)

    axes[1].hexbin(y_test, y_pred, gridsize=40, cmap="viridis", mincnt=1)
    lims = [0, 1]
    axes[1].plot(lims, lims, "r--", label="y = y_pred")
    axes[1].set_xlabel("Actual sentiment_ratio")
    axes[1].set_ylabel("Predicted sentiment_ratio")
    axes[1].set_title("Predicted vs actual")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "10_regression_diagnostics.png", dpi=130)
    plt.close()

    # Feature importance (impurity) for quick reference
    imp = pd.Series(
        model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)
    imp.head(20).to_csv(FIGURES / "10_regression_importance.csv")
    print("\nTop 10 features (regression, impurity):")
    print(imp.head(10).to_string())

    return {"rmse": rmse, "mae": mae, "r2": r2,
            "best_params": grid.best_params_}


# %% Main
if __name__ == "__main__":
    clf_results, clf_best_params = run_classification()
    reg_results = run_regression()

    # Single consolidated summary JSON for the report appendix
    summary = {
        "classification": {
            "best_params": clf_best_params,
            "ablations": [
                {"name": r["name"],
                 "macro_f1": float(r["f1_macro"]),
                 "weighted_f1": float(r["f1_weighted"]),
                 "roc_auc_ovr": (float(r["roc_auc_ovr"])
                                 if not pd.isna(r["roc_auc_ovr"]) else None)}
                for r in clf_results
            ],
        },
        "regression": {
            "rmse": reg_results["rmse"],
            "mae":  reg_results["mae"],
            "r2":   reg_results["r2"],
            "best_params": reg_results["best_params"],
        }
    }
    with open(FIGURES / "10_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\nAll artefacts written to reports/figures/ with prefix '10_'")
    print("\nReady for the Deep Learning stage.")

    