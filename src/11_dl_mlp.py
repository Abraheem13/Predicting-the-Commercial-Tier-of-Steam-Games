"""
Stage 11 - Multilayer Perceptron (PyTorch) - classification + regression.

Designed for direct paired comparison against the Random Forest in Stage 10:
  * Same train/test splits from data/processed/ (identical rows, identical targets)
  * Same ablations: full features, pre-launch only, PCA-24 inputs
  * Same primary metric: macro-F1 (classification); RMSE/MAE/R2 (regression)

Architecture (classification and regression heads use the same backbone):
  Input  -> Linear(d_in -> 256) -> BatchNorm -> ReLU -> Dropout(0.3)
         -> Linear(256 -> 128)  -> BatchNorm -> ReLU -> Dropout(0.3)
         -> Linear(128 -> d_out)

Training:
  * AdamW optimiser, weight_decay=1e-4
  * Cosine learning rate schedule from 1e-3 down over max_epochs
  * Class-weighted cross-entropy (sklearn compute_class_weight with 'balanced')
  * Early stopping on validation macro-F1 (patience=10)
  * 10% of the training set held out as validation for early stopping
  * StandardScaler fit on train slice ONLY (no test leakage)

Runs CPU-only. On 68k rows this takes ~5-10 minutes per configuration.
"""
# %% Imports
import json
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (f1_score, confusion_matrix, roc_auc_score,
                             precision_recall_fscore_support,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.preprocessing import label_binarize

import seaborn as sns

from utils import DATA_PROCESSED, FIGURES

# ---------- Reproducibility ------------------------------------------------
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ---------- Hyperparameters ------------------------------------------------
BATCH_SIZE     = 512
MAX_EPOCHS     = 120
EARLY_STOP_PAT = 10      # stop if val metric doesn't improve for N epochs
DROPOUT        = 0.30
HIDDEN_SIZES   = (256, 128)
LR             = 1e-3
WEIGHT_DECAY   = 1e-4

CLASS_ORDER = ["Low", "Medium", "High", "Very High"]

PRE_LAUNCH_EXACT = {
    "price", "dlc_count", "num_genres", "num_languages", "num_platforms",
    "num_categories", "num_tags", "num_developers", "num_publishers",
    "required_age", "days_since_release", "achievements",
}
REG_LEAKY_FEATURES = ["positive", "negative", "recommendations"]


# ---------- Model ----------------------------------------------------------
class MLP(nn.Module):
    """Shared backbone: BN -> ReLU -> Dropout. Output head size configurable."""
    def __init__(self, d_in, d_out, hidden=HIDDEN_SIZES, dropout=DROPOUT):
        super().__init__()
        dims = [d_in, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(),
                       nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------- Training loop (classification) ---------------------------------
def train_classifier(X_train, y_train, X_val, y_val, d_out, class_weights,
                     tag="model"):
    d_in = X_train.shape[1]
    model = MLP(d_in, d_out).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR,
                              weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # Data loaders
    tr_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.long))
    va_ds = TensorDataset(torch.tensor(X_val,   dtype=torch.float32),
                          torch.tensor(y_val,   dtype=torch.long))
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                       drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

    history = {"train_loss": [], "val_loss": [],
               "train_f1":   [], "val_f1":   []}
    best_val_f1 = -np.inf
    best_state  = None
    bad_epochs  = 0

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        # -- train --
        model.train()
        tr_loss_sum, tr_preds, tr_targets = 0.0, [], []
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            tr_loss_sum += loss.item() * len(xb)
            tr_preds.append(logits.argmax(dim=1).cpu().numpy())
            tr_targets.append(yb.cpu().numpy())
        sched.step()
        tr_loss = tr_loss_sum / len(tr_ds)
        tr_f1   = f1_score(np.concatenate(tr_targets), np.concatenate(tr_preds),
                           average="macro")

        # -- validate --
        model.eval()
        va_loss_sum, va_preds, va_targets = 0.0, [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss   = loss_fn(logits, yb)
                va_loss_sum += loss.item() * len(xb)
                va_preds.append(logits.argmax(dim=1).cpu().numpy())
                va_targets.append(yb.cpu().numpy())
        va_loss = va_loss_sum / len(va_ds)
        va_f1   = f1_score(np.concatenate(va_targets), np.concatenate(va_preds),
                           average="macro")

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(va_f1)

        # Early stopping
        if va_f1 > best_val_f1 + 1e-4:
            best_val_f1 = va_f1
            best_state  = copy.deepcopy(model.state_dict())
            bad_epochs  = 0
        else:
            bad_epochs += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d} | "
                  f"tr loss {tr_loss:.4f}  tr F1 {tr_f1:.4f} | "
                  f"va loss {va_loss:.4f}  va F1 {va_f1:.4f}")

        if bad_epochs >= EARLY_STOP_PAT:
            print(f"  early stopping at epoch {epoch} "
                  f"(best val F1 = {best_val_f1:.4f})")
            break

    print(f"  trained in {time.time() - t0:.1f}s")
    model.load_state_dict(best_state)
    return model, history


def evaluate_classifier(model, X_test, y_test, scaler=None, tag=""):
    """Return metrics dict and predicted probabilities."""
    model.eval()
    Xt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(Xt)
        proba  = F.softmax(logits, dim=1).cpu().numpy()
    y_pred = proba.argmax(axis=1)

    f1_macro    = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    y_test_bin = label_binarize(y_test, classes=list(range(len(CLASS_ORDER))))
    try:
        auc = roc_auc_score(y_test_bin, proba, multi_class="ovr",
                            average="macro")
    except ValueError:
        auc = np.nan

    p, r, f, s = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(len(CLASS_ORDER))),
        zero_division=0,
    )
    per_class = pd.DataFrame({
        "class": CLASS_ORDER, "precision": p, "recall": r,
        "f1": f, "support": s,
    })

    print(f"\n{tag}")
    print(f"  Macro-F1      : {f1_macro:.4f}")
    print(f"  Weighted-F1   : {f1_weighted:.4f}")
    print(f"  ROC-AUC OvR   : {auc:.4f}")
    print("  Per-class:")
    print(per_class.to_string(index=False))

    return {
        "f1_macro": f1_macro, "f1_weighted": f1_weighted,
        "roc_auc_ovr": auc, "per_class": per_class,
        "y_pred": y_pred, "proba": proba,
    }


# ---------- Training loop (regression) -------------------------------------
def train_regressor(X_train, y_train, X_val, y_val, tag="reg"):
    d_in = X_train.shape[1]
    model = MLP(d_in, 1).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR,
                              weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
    loss_fn = nn.MSELoss()

    tr_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                          torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    va_ds = TensorDataset(torch.tensor(X_val,   dtype=torch.float32),
                          torch.tensor(y_val,   dtype=torch.float32).unsqueeze(1))
    tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False)

    history = {"train_loss": [], "val_loss": [],
               "train_rmse": [], "val_rmse": []}
    best_val_rmse = np.inf
    best_state    = None
    bad_epochs    = 0

    t0 = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        # train
        model.train()
        tr_loss_sum, tr_se_sum, tr_n = 0.0, 0.0, 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss_sum += loss.item() * len(xb)
            tr_se_sum   += ((pred - yb) ** 2).sum().item()
            tr_n        += len(xb)
        sched.step()
        tr_loss = tr_loss_sum / tr_n
        tr_rmse = (tr_se_sum / tr_n) ** 0.5

        # validate
        model.eval()
        va_loss_sum, va_se_sum, va_n = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss_sum += loss.item() * len(xb)
                va_se_sum   += ((pred - yb) ** 2).sum().item()
                va_n        += len(xb)
        va_loss = va_loss_sum / va_n
        va_rmse = (va_se_sum / va_n) ** 0.5

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_rmse"].append(tr_rmse)
        history["val_rmse"].append(va_rmse)

        if va_rmse < best_val_rmse - 1e-5:
            best_val_rmse = va_rmse
            best_state    = copy.deepcopy(model.state_dict())
            bad_epochs    = 0
        else:
            bad_epochs += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d} | tr RMSE {tr_rmse:.4f} | "
                  f"va RMSE {va_rmse:.4f}")

        if bad_epochs >= EARLY_STOP_PAT:
            print(f"  early stopping at epoch {epoch} "
                  f"(best val RMSE = {best_val_rmse:.4f})")
            break

    print(f"  trained in {time.time() - t0:.1f}s")
    model.load_state_dict(best_state)
    return model, history


# ---------- Plotting helpers ----------------------------------------------
def plot_training_curves(history, filename, ylab_left="Loss",
                         ylab_right="Macro-F1",
                         key_left=("train_loss", "val_loss"),
                         key_right=("train_f1", "val_f1"),
                         title=""):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ep = range(1, len(history[key_left[0]]) + 1)
    axes[0].plot(ep, history[key_left[0]],  label="train", color="steelblue")
    axes[0].plot(ep, history[key_left[1]], label="val",   color="orange")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel(ylab_left)
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[0].set_title("Loss curves")

    axes[1].plot(ep, history[key_right[0]],  label="train", color="steelblue")
    axes[1].plot(ep, history[key_right[1]], label="val",   color="orange")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel(ylab_right)
    axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_title(f"{ylab_right} curves")

    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=130)
    plt.close()


def save_confusion_matrix(y_true, y_pred, filename, title):
    cm = confusion_matrix(y_true, y_pred,
                          labels=list(range(len(CLASS_ORDER))),
                          normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_ORDER, yticklabels=CLASS_ORDER, ax=ax,
                cbar_kws={"label": "Row-normalised proportion"})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(FIGURES / filename, dpi=130)
    plt.close()


# ---------- Main: classification ------------------------------------------
def run_classification():
    print("\n" + "#" * 70)
    print("# MLP CLASSIFICATION")
    print("#" * 70)

    X_train_df = pd.read_parquet(DATA_PROCESSED / "clf_Xtrain.parquet")
    X_test_df  = pd.read_parquet(DATA_PROCESSED / "clf_Xtest.parquet")
    y_train_s  = pd.read_parquet(DATA_PROCESSED / "clf_ytrain.parquet")["owner_class"]
    y_test_s   = pd.read_parquet(DATA_PROCESSED / "clf_ytest.parquet")["owner_class"]

    class_to_idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    y_train_full = y_train_s.map(class_to_idx).to_numpy()
    y_test       = y_test_s.map(class_to_idx).to_numpy()

    results = []

    # ------- Config A: full features -------
    print("\n--- Full-feature MLP ---")
    X_tr_full, X_val_full, y_tr, y_val = train_test_split(
        X_train_df.to_numpy(), y_train_full,
        test_size=0.10, stratify=y_train_full,
        random_state=RANDOM_STATE,
    )
    scaler_full = StandardScaler().fit(X_tr_full)
    X_tr_s  = scaler_full.transform(X_tr_full)
    X_val_s = scaler_full.transform(X_val_full)
    X_test_s_full = scaler_full.transform(X_test_df.to_numpy())

    cls_weights = torch.tensor(
        compute_class_weight("balanced", classes=np.arange(len(CLASS_ORDER)),
                             y=y_tr),
        dtype=torch.float32,
    )
    print(f"  Class weights: {cls_weights.numpy().round(3)}")

    model_full, hist_full = train_classifier(
        X_tr_s, y_tr, X_val_s, y_val,
        d_out=len(CLASS_ORDER), class_weights=cls_weights,
        tag="full",
    )
    res_full = evaluate_classifier(model_full, X_test_s_full, y_test,
                                    tag="MLP (full features) on test")
    plot_training_curves(hist_full, "11_training_curves_full.png",
                          title="MLP training curves - full features")
    save_confusion_matrix(y_test, res_full["y_pred"],
                           "11_cm_mlp_full.png",
                           "MLP confusion matrix (full features)")
    results.append({"name": "MLP (full features)", **res_full})

    # ------- Config B: pre-launch only -------
    print("\n--- Pre-launch-only MLP ---")
    pl_cols = [c for c in X_train_df.columns
               if c in PRE_LAUNCH_EXACT or c.startswith(("genre_", "cat_"))]
    print(f"  Pre-launch columns: {len(pl_cols)} "
          f"(dropped {X_train_df.shape[1] - len(pl_cols)} post-launch features)")

    X_tr_pl, X_val_pl, y_tr_pl, y_val_pl = train_test_split(
        X_train_df[pl_cols].to_numpy(), y_train_full,
        test_size=0.10, stratify=y_train_full,
        random_state=RANDOM_STATE,
    )
    scaler_pl = StandardScaler().fit(X_tr_pl)
    X_tr_ps  = scaler_pl.transform(X_tr_pl)
    X_val_ps = scaler_pl.transform(X_val_pl)
    X_te_ps  = scaler_pl.transform(X_test_df[pl_cols].to_numpy())

    cls_weights_pl = torch.tensor(
        compute_class_weight("balanced", classes=np.arange(len(CLASS_ORDER)),
                             y=y_tr_pl),
        dtype=torch.float32,
    )
    model_pl, hist_pl = train_classifier(
        X_tr_ps, y_tr_pl, X_val_ps, y_val_pl,
        d_out=len(CLASS_ORDER), class_weights=cls_weights_pl,
        tag="prelaunch",
    )
    res_pl = evaluate_classifier(model_pl, X_te_ps, y_test,
                                  tag="MLP (pre-launch only) on test")
    plot_training_curves(hist_pl, "11_training_curves_prelaunch.png",
                          title="MLP training curves - pre-launch only")
    save_confusion_matrix(y_test, res_pl["y_pred"],
                           "11_cm_mlp_prelaunch.png",
                           "MLP confusion matrix (pre-launch only)")
    results.append({"name": "MLP (pre-launch only)", **res_pl})

    # ------- Config C: PCA-24 -------
    print("\n--- PCA-24 MLP ---")
    # PCA goes on the SCALED full-feature train slice only
    pca = PCA(n_components=24, random_state=RANDOM_STATE).fit(X_tr_s)
    print(f"  PCA variance retained: {pca.explained_variance_ratio_.sum():.3f}")
    X_tr_pca  = pca.transform(X_tr_s)
    X_val_pca = pca.transform(X_val_s)
    X_te_pca  = pca.transform(X_test_s_full)

    model_pca, hist_pca = train_classifier(
        X_tr_pca, y_tr, X_val_pca, y_val,
        d_out=len(CLASS_ORDER), class_weights=cls_weights,
        tag="pca",
    )
    res_pca = evaluate_classifier(model_pca, X_te_pca, y_test,
                                   tag="MLP (PCA-24) on test")
    plot_training_curves(hist_pca, "11_training_curves_pca.png",
                          title="MLP training curves - PCA-24 inputs")
    save_confusion_matrix(y_test, res_pca["y_pred"],
                           "11_cm_mlp_pca.png",
                           "MLP confusion matrix (PCA-24)")
    results.append({"name": "MLP (PCA-24)", **res_pca})

    # ------- Summary table -------
    summary = pd.DataFrame([{
        "Model": r["name"],
        "Macro-F1":    r["f1_macro"],
        "Weighted-F1": r["f1_weighted"],
        "ROC-AUC OvR": r["roc_auc_ovr"],
    } for r in results])
    print("\n" + "=" * 60)
    print("MLP CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))
    summary.to_csv(FIGURES / "11_classification_summary.csv", index=False)

    # Per-class tables for the report
    for r in results:
        safe = r["name"].replace(" ", "_").replace("(", "").replace(")", "")
        r["per_class"].to_csv(FIGURES / f"11_per_class_{safe}.csv", index=False)

    return results


# ---------- Main: regression ----------------------------------------------
def run_regression():
    print("\n" + "#" * 70)
    print("# MLP REGRESSION (sentiment_ratio)")
    print("#" * 70)

    X_train_df = pd.read_parquet(DATA_PROCESSED / "reg_Xtrain.parquet")
    X_test_df  = pd.read_parquet(DATA_PROCESSED / "reg_Xtest.parquet")
    y_train    = pd.read_parquet(DATA_PROCESSED / "reg_ytrain.parquet")["sentiment_ratio"].to_numpy()
    y_test     = pd.read_parquet(DATA_PROCESSED / "reg_ytest.parquet") ["sentiment_ratio"].to_numpy()

    # Drop leakage features (same as RF script)
    feat_cols = [c for c in X_train_df.columns if c not in REG_LEAKY_FEATURES]
    X_train_df = X_train_df[feat_cols]
    X_test_df  = X_test_df [feat_cols]
    print(f"  Dropped leaky features: {REG_LEAKY_FEATURES}")
    print(f"  Train: {X_train_df.shape}  Test: {X_test_df.shape}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_df.to_numpy(), y_train, test_size=0.10,
        random_state=RANDOM_STATE,
    )
    scaler = StandardScaler().fit(X_tr)
    X_tr_s  = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s  = scaler.transform(X_test_df.to_numpy())

    model, history = train_regressor(X_tr_s, y_tr, X_val_s, y_val)

    # Evaluate on test
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_te_s, dtype=torch.float32).to(DEVICE)
        preds = model(Xt).squeeze(1).cpu().numpy()

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    baseline_rmse = np.sqrt(mean_squared_error(
        y_test, np.full_like(y_test, y_tr.mean())
    ))

    print("\n" + "=" * 60)
    print("MLP REGRESSION SUMMARY")
    print("=" * 60)
    print(f"  Test RMSE            : {rmse:.4f}")
    print(f"  Test MAE             : {mae:.4f}")
    print(f"  Test R^2             : {r2:.4f}")
    print(f"  Mean-baseline RMSE   : {baseline_rmse:.4f}")
    print(f"  Improvement over base: {(baseline_rmse - rmse) / baseline_rmse * 100:.1f}%")

    pd.DataFrame([{
        "Model": "MLP Regressor (no-leakage)",
        "RMSE": rmse, "MAE": mae, "R2": r2,
        "Mean_baseline_RMSE": baseline_rmse,
        "Improvement_pct": (baseline_rmse - rmse) / baseline_rmse * 100,
    }]).to_csv(FIGURES / "11_regression_summary.csv", index=False)

    # Training curves
    plot_training_curves(history, "11_training_curves_regression.png",
                          ylab_left="MSE loss", ylab_right="RMSE",
                          key_right=("train_rmse", "val_rmse"),
                          title="MLP regression training curves")

    # Diagnostics
    residuals = y_test - preds
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(preds, residuals, s=3, alpha=0.3, color="steelblue")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted sentiment_ratio")
    axes[0].set_ylabel("Residual")
    axes[0].set_title(f"Residuals (RMSE={rmse:.3f}, R2={r2:.3f})")
    axes[0].grid(alpha=0.3)

    axes[1].hexbin(y_test, preds, gridsize=40, cmap="viridis", mincnt=1)
    axes[1].plot([0, 1], [0, 1], "r--", label="y = y_pred")
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title("Predicted vs actual (MLP, leakage-fixed)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "11_regression_diagnostics.png", dpi=130)
    plt.close()

    return {"rmse": rmse, "mae": mae, "r2": r2,
            "baseline_rmse": baseline_rmse}


# ---------- Main ----------------------------------------------------------
if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")

    clf_results = run_classification()
    reg_results = run_regression()

    # Consolidated JSON summary for the report appendix
    summary = {
        "classification": {
            "hyperparameters": {
                "hidden_sizes": list(HIDDEN_SIZES),
                "dropout": DROPOUT,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
                "max_epochs": MAX_EPOCHS,
                "early_stopping_patience": EARLY_STOP_PAT,
                "optimiser": "AdamW",
                "lr_schedule": "CosineAnnealingLR",
                "loss": "class-weighted CrossEntropyLoss",
            },
            "ablations": [
                {"name": r["name"],
                 "macro_f1":    float(r["f1_macro"]),
                 "weighted_f1": float(r["f1_weighted"]),
                 "roc_auc_ovr": (float(r["roc_auc_ovr"])
                                 if not np.isnan(r["roc_auc_ovr"]) else None)}
                for r in clf_results
            ],
        },
        "regression": {
            "rmse": reg_results["rmse"],
            "mae":  reg_results["mae"],
            "r2":   reg_results["r2"],
            "mean_baseline_rmse": reg_results["baseline_rmse"],
            "note": "positive/negative/recommendations excluded (target leakage)",
        },
    }
    with open(FIGURES / "11_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nAll MLP artefacts written to reports/figures/ with prefix '11_'")
    print("Ready for the final comparison section.")