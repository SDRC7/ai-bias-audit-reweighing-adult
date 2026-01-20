import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.config import get_config
from src.data_load import load_adult_openml
from src.preprocess import preprocess_fit_transform, add_binary_columns
from src.utils_repro import set_global_seed, ensure_dir, assert_binary_series


def main():
    cfg = get_config()
    set_global_seed(cfg.random_seed)

    # Ensure output dirs
    ensure_dir("results/eda")
    ensure_dir("results/splits")
    ensure_dir("reports/figures")

    # Load raw
    df_raw = load_adult_openml(data_home="data/raw/openml")

    # Create binary columns for EDA
    df_bins = add_binary_columns(df_raw, cfg)

    # EDA outputs: group counts
    group_counts = (
        df_bins[cfg.sensitive_col_bin]
        .value_counts()
        .rename_axis(cfg.sensitive_col_bin)
        .reset_index(name="count")
        .sort_values(cfg.sensitive_col_bin)
    )
    group_counts.to_csv("results/eda/group_counts.csv", index=False)

    # EDA outputs: label counts
    label_counts = (
        df_bins[cfg.label_col_bin]
        .value_counts()
        .rename_axis(cfg.label_col_bin)
        .reset_index(name="count")
        .sort_values(cfg.label_col_bin)
    )
    label_counts.to_csv("results/eda/label_counts.csv", index=False)

    # Plot: label distribution by group
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df_bins, x=cfg.sensitive_col_bin, hue=cfg.label_col_bin)
    plt.title("Adult: income_bin by sex_bin")
    plt.xlabel("sex_bin (0=Female, 1=Male)")
    plt.ylabel("count")
    plt.legend(title="income_bin (0<=50K, 1>50K)")
    plt.tight_layout()
    plt.savefig("reports/figures/eda_label_by_group.png", dpi=200)
    plt.close()

    # Preprocess to X/y/sex (and fitted preprocessor for later phases)
    X, y, sex_bin, _preprocessor = preprocess_fit_transform(df_raw, cfg)

    # Guards before split
    assert_binary_series(y, "income_bin(y)")
    assert_binary_series(sex_bin, "sex_bin")

    n = len(y)
    idx = np.arange(n)

    train_idx, test_idx = train_test_split(
        idx,
        test_size=cfg.test_size,
        random_state=cfg.random_seed,
        stratify=y,
    )

    # Save indices
    np.save("results/splits/train_idx.npy", train_idx)
    np.save("results/splits/test_idx.npy", test_idx)

    # Alignment checks
    if len(set(train_idx).intersection(set(test_idx))) != 0:
        raise AssertionError("train/test overlap found")
    if len(train_idx) + len(test_idx) != n:
        raise AssertionError("split sizes do not add up")

    # Verify split alignment across X/y/sex
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    sex_train, sex_test = sex_bin[train_idx], sex_bin[test_idx]

    if not (len(X_train) == len(y_train) == len(sex_train)):
        raise AssertionError("train split misalignment")
    if not (len(X_test) == len(y_test) == len(sex_test)):
        raise AssertionError("test split misalignment")

    # No-NaN guards post-split
    if np.isnan(X_train).any() or np.isnan(X_test).any():
        raise AssertionError("NaNs found in split X arrays.")

    print("Phase 1 complete.")
    print(f"Rows: n={n}, train={len(train_idx)}, test={len(test_idx)}")
    print("Wrote: results/eda/group_counts.csv, results/eda/label_counts.csv, reports/figures/eda_label_by_group.png")
    print("Wrote: results/splits/train_idx.npy, results/splits/test_idx.npy")


if __name__ == "__main__":
    main()

