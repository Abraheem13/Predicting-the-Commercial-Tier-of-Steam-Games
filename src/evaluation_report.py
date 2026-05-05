"""
Standalone classification evaluation report for owner_class.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from utils import DATA_PROCESSED

CLASS_ORDER = ["Low", "Medium", "High", "Very High"]
MODEL_NAME = "RandomForestClassifier (tuned, full features)"
PRED_FILE = DATA_PROCESSED / "clf_eval_rf_tuned_predictions.csv"
PROBA_FILE = DATA_PROCESSED / "clf_eval_rf_tuned_proba.csv"

BEST_CLF_PARAMS = {
    "max_depth": 20,
    "max_features": 0.3,
    "min_samples_leaf": 1,
    "n_estimators": 400,
}


def load_or_build_predictions():
    # Prefer saved outputs from Stage 10; fallback refits on saved train/test split.
    if PRED_FILE.exists() and PROBA_FILE.exists():
        pred_df = pd.read_csv(PRED_FILE)
        proba_df = pd.read_csv(PROBA_FILE)
        y_true = pred_df["y_true"].astype(str).to_numpy()
        y_pred = pred_df["y_pred"].astype(str).to_numpy()
        y_proba = proba_df[CLASS_ORDER].to_numpy()
        return y_true, y_pred, y_proba

    X_train = pd.read_parquet(DATA_PROCESSED / "clf_Xtrain.parquet")
    X_test = pd.read_parquet(DATA_PROCESSED / "clf_Xtest.parquet")
    y_train = pd.read_parquet(DATA_PROCESSED / "clf_ytrain.parquet")["owner_class"].astype(str)
    y_test = pd.read_parquet(DATA_PROCESSED / "clf_ytest.parquet")["owner_class"].astype(str)

    model = RandomForestClassifier(
        **BEST_CLF_PARAMS,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_true = y_test.to_numpy()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(PRED_FILE, index=False)
    pd.DataFrame(y_proba, columns=model.classes_).reindex(
        columns=CLASS_ORDER
    ).to_csv(PROBA_FILE, index=False)

    return y_true, y_pred, pd.read_csv(PROBA_FILE)[CLASS_ORDER].to_numpy()


def main():
    y_true, y_pred, y_proba = load_or_build_predictions()

    macro_f1 = f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, labels=CLASS_ORDER, average="weighted")

    y_true_bin = label_binarize(y_true, classes=CLASS_ORDER)
    roc_auc_ovr = roc_auc_score(y_true_bin, y_proba, multi_class="ovr", average="macro")

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        target_names=CLASS_ORDER,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_ORDER)

    print(f"macro-F1 = {macro_f1:.4f}")
    print(f"weighted-F1 = {weighted_f1:.4f}")
    print(f"ROC-AUC OvR = {roc_auc_ovr:.4f}")
    print(f"model = {MODEL_NAME}\n")

    print("Classification report:")
    print(report)

    print("Confusion matrix (rows = true, cols = predicted):")
    print(CLASS_ORDER)
    print(cm.tolist())


if __name__ == "__main__":
    main()
