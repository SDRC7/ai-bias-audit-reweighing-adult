import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import get_config
from src.data_load import load_adult_openml
from src.preprocess import preprocess_fit_transform
from src.utils_repro import ensure_dir, set_global_seed


def main():
    cfg = get_config()
    set_global_seed(cfg.random_seed)

    # Ensure output dirs
    ensure_dir("results/models")
    ensure_dir("results/predictions")
    ensure_dir("results/metrics")

    # 1) Load data + preprocess (same as Phase 1, but we reuse saved split indices)
    df_raw = load_adult_openml(data_home="data/raw/openml")
    X, y, sex_bin, preprocessor = preprocess_fit_transform(df_raw, cfg)

    # 2) Load saved split indices (DO NOT resplit)
    train_idx_path = Path("results/splits/train_idx.npy")
    test_idx_path = Path("results/splits/test_idx.npy")
    if not train_idx_path.exists() or not test_idx_path.exists():
        raise FileNotFoundError(
            "Missing split files. Run Phase 1 first to create:\n"
            " - results/splits/train_idx.npy\n"
            " - results/splits/test_idx.npy"
        )

    train_idx = np.load(train_idx_path)
    test_idx = np.load(test_idx_path)

    # 3) Build split datasets (alignment guaranteed by shared indices)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    sex_test = sex_bin[test_idx]

    # Guards to prevent silent mismatch bugs later
    assert len(X_train) == len(y_train), "X_train and y_train length mismatch"
    assert len(X_test) == len(y_test) == len(sex_test), "Test split alignment mismatch"

    if np.isnan(X_train).any() or np.isnan(X_test).any():
        raise AssertionError("NaNs found in X after preprocessing/splitting.")
    if not set(np.unique(y_train)).issubset({0, 1}):
        raise AssertionError(f"y_train must be binary {{0,1}}, got {set(np.unique(y_train))}")
    if not set(np.unique(y_test)).issubset({0, 1}):
        raise AssertionError(f"y_test must be binary {{0,1}}, got {set(np.unique(y_test))}")

    # 4) Train baseline Logistic Regression
    clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=cfg.random_seed)
    w_train = np.load("results/metrics/reweighing_sample_weight.npy")
    clf.fit(X_train, y_train, sample_weight=w_train)

    # 5) Predict (need y_prob for audit + ROC-AUC)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # 6) Compute ML metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    rocauc = roc_auc_score(y_test, y_prob)

    metrics = {
        "model": "LogisticRegression(liblinear)",
        "random_seed": int(cfg.random_seed),
        "test_size": float(cfg.test_size),
        "threshold": 0.5,
        "accuracy": float(acc),
        "f1": float(f1),
        "roc_auc": float(rocauc),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }

    # 7) Save artifacts
    joblib.dump(
        {"model": clf, "preprocessor": preprocessor, "config": cfg},
        "results/models/reweighed_logreg.joblib"
    )

    preds_df = pd.DataFrame({
        "y_true": y_test.astype(int),
        "y_pred": y_pred.astype(int),
        "y_prob": y_prob.astype(float),
        "sex_bin": sex_test.astype(int),  # helpful later (Phase 3 audit); ok to store now
    })
    preds_df.to_csv("results/predictions/reweighed_test_preds.csv", index=False)

    with open("results/metrics/reweighed_ml_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Phase 2 complete.")
    print("Saved model: results/models/reweighed_logreg.joblib")
    print("Saved predictions: results/predictions/reweighed_test_preds.csv")
    print("Saved metrics: results/metrics/reweighed_ml_metrics.json")
    print(f"Accuracy={acc:.4f} | F1={f1:.4f} | ROC-AUC={rocauc:.4f}")


if __name__ == "__main__":
    main()

