import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing

from src.config import get_config
from src.data_load import load_adult_openml
from src.preprocess import add_binary_columns, preprocess_fit_transform
from src.utils_repro import ensure_dir


def main():
    cfg = get_config()

    # Ensure output dirs
    ensure_dir("results/metrics")
    ensure_dir("reports/figures")

    # 1) Load raw dataset
    df_raw = load_adult_openml(data_home="data/raw/openml")

    # 2) Create binary columns (income_bin, sex_bin) on FULL df (then select train rows)
    df_bins = add_binary_columns(df_raw, cfg)

    # Guards: must be binary and non-null for the 2 key columns
    for col in [cfg.label_col_bin, cfg.sensitive_col_bin]:
        if df_bins[col].isna().any():
            raise AssertionError(f"NaNs found in {col} after mapping.")
        vals = set(df_bins[col].unique().tolist())
        if not vals.issubset({0, 1}):
            raise AssertionError(f"{col} must be in {{0,1}}; got {sorted(list(vals))}")

    # 3) Load saved split indices (ONLY use train)
    train_idx = np.load("results/splits/train_idx.npy")
    test_idx = np.load("results/splits/test_idx.npy")

    # Safety: ensure no overlap
    if len(set(train_idx).intersection(set(test_idx))) != 0:
        raise AssertionError("train_idx and test_idx overlap; split files corrupted?")

    # 4) Build df_train (ONLY training rows)
    df_train = df_bins.iloc[train_idx].copy().reset_index(drop=True)

    # IMPORTANT:
    # BinaryLabelDataset rejects NA anywhere in the DataFrame,
    # so pass ONLY label + protected attribute columns.
    df_aif = df_train[[cfg.label_col_bin, cfg.sensitive_col_bin]].copy()

    # 5) Convert to AIF360 BinaryLabelDataset
    bld_train = BinaryLabelDataset(
        df=df_aif,
        label_names=[cfg.label_col_bin],
        protected_attribute_names=[cfg.sensitive_col_bin],
        favorable_label=cfg.favorable_label,
        unfavorable_label=cfg.unfavorable_label,
    )

    # 6) Run AIF360 Reweighing on TRAIN ONLY
    rw = Reweighing(
        unprivileged_groups=cfg.unprivileged_groups,  # [{'sex_bin':0}]
        privileged_groups=cfg.privileged_groups,      # [{'sex_bin':1}]
    )
    bld_train_rw = rw.fit(bld_train).transform(bld_train)

    # 7) Export instance_weights -> sample_weight aligned with df_train rows
    sample_weight = np.asarray(bld_train_rw.instance_weights).reshape(-1)

    if len(sample_weight) != len(df_train):
        raise AssertionError(
            f"sample_weight length mismatch: len(sample_weight)={len(sample_weight)} vs len(df_train)={len(df_train)}"
        )
    if np.isnan(sample_weight).any():
        raise AssertionError("NaNs found in sample_weight.")

    # Extra alignment check vs X_train/y_train produced by preprocess_fit_transform + same train_idx
    X, y, sex_bin, _ = preprocess_fit_transform(df_raw, cfg)
    X_train, y_train, sex_train = X[train_idx], y[train_idx], sex_bin[train_idx]
    if len(y_train) != len(sample_weight):
        raise AssertionError(
            f"Alignment failure: len(y_train)={len(y_train)} vs len(sample_weight)={len(sample_weight)}"
        )

    # 8) Save weights
    np.save("results/metrics/reweighing_sample_weight.npy", sample_weight)

    # 9) Save summary grouped by (sex_bin, income_bin)
    summary = (
        df_train.assign(sample_weight=sample_weight)
        .groupby([cfg.sensitive_col_bin, cfg.label_col_bin])
        .agg(
            n=("sample_weight", "size"),
            weight_mean=("sample_weight", "mean"),
            weight_std=("sample_weight", "std"),
            weight_min=("sample_weight", "min"),
            weight_max=("sample_weight", "max"),
        )
        .reset_index()
        .sort_values([cfg.sensitive_col_bin, cfg.label_col_bin])
    )
    summary.to_csv("results/metrics/reweighing_weight_summary.csv", index=False)

    # 10) Plot histogram of weights
    plt.figure(figsize=(7, 4))
    plt.hist(sample_weight, bins=40)
    plt.title("AIF360 Reweighing: sample_weight distribution (train set)")
    plt.xlabel("sample_weight")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig("reports/figures/reweighing_weight_hist.png", dpi=200)
    plt.close()

    print("Phase 4 complete.")
    print("Saved: results/metrics/reweighing_sample_weight.npy")
    print("Saved: results/metrics/reweighing_weight_summary.csv")
    print("Saved: reports/figures/reweighing_weight_hist.png")
    print(f"Train rows: {len(df_train)} | sample_weight mean={sample_weight.mean():.6f} std={sample_weight.std():.6f}")


if __name__ == "__main__":
    main()
