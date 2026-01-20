import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
)

from src.utils_repro import ensure_dir


def main():
    # Ensure output dirs exist
    ensure_dir("results/metrics")
    ensure_dir("reports/figures")

    # Load predictions CSV from Phase 2
    path = "results/predictions/reweighed_test_preds.csv"
    df = pd.read_csv(path)

    required_cols = ["y_true", "y_pred", "y_prob", "sex_bin"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise AssertionError(f"Missing columns in {path}: {missing}. Found: {list(df.columns)}")

    # Basic guards
    for c in required_cols:
        if df[c].isna().any():
            raise AssertionError(f"Found NaNs in column '{c}'")

    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()
    sex_bin = df["sex_bin"].astype(int).to_numpy()

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise AssertionError(f"y_true not binary {{0,1}}. Got: {sorted(set(np.unique(y_true)))}")
    if not set(np.unique(y_pred)).issubset({0, 1}):
        raise AssertionError(f"y_pred not binary {{0,1}}. Got: {sorted(set(np.unique(y_pred)))}")
    if not set(np.unique(sex_bin)).issubset({0, 1}):
        raise AssertionError(f"sex_bin not binary {{0,1}}. Got: {sorted(set(np.unique(sex_bin)))}")

    # Group selection rates
    mf = MetricFrame(
        metrics=selection_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sex_bin,
    )
    selection_by_group = mf.by_group  # Series indexed by 0/1

    sel_df = selection_by_group.rename("selection_rate").reset_index()
    first_col = sel_df.columns[0]
    sel_df = sel_df.rename(columns={first_col: 'sex_bin'}).sort_values('sex_bin')
    sel_df.to_csv("results/metrics/reweighed_group_selection_rates.csv", index=False)

    # Plot: selection rate by group
    plt.figure(figsize=(6, 4))
    plt.bar(sel_df["sex_bin"].astype(str), sel_df["selection_rate"])
    plt.ylim(0, 1)
    plt.xlabel("sex_bin (0=Female, 1=Male)")
    plt.ylabel("Selection rate (P(y_pred=1))")
    plt.title("Baseline: Selection rate by group")
    plt.tight_layout()
    plt.savefig("reports/figures/reweighed_selection_rate_by_group.png", dpi=200)
    plt.close()

    # Scalar fairness metrics
    dp_diff = demographic_parity_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sex_bin)
    dp_ratio = demographic_parity_ratio(y_true=y_true, y_pred=y_pred, sensitive_features=sex_bin)
    eo_diff = equalized_odds_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sex_bin)
    eo_ratio = equalized_odds_ratio(y_true=y_true, y_pred=y_pred, sensitive_features=sex_bin)

    fairness = {
        "demographic_parity_difference": float(dp_diff),
        "demographic_parity_ratio": float(dp_ratio),
        "equalized_odds_difference": float(eo_diff),
        "equalized_odds_ratio": float(eo_ratio),
        "n_test": int(len(df)),
        "groups_present": sorted(list(set(sex_bin.tolist()))),
    }

    with open("results/metrics/reweighed_fairness_metrics.json", "w") as f:
        json.dump(fairness, f, indent=2)

    # Plot: fairness metrics
    metric_names = [
        "demographic_parity_difference",
        "demographic_parity_ratio",
        "equalized_odds_difference",
        "equalized_odds_ratio",
    ]
    metric_vals = [fairness[m] for m in metric_names]

    plt.figure(figsize=(10, 4))
    plt.bar(metric_names, metric_vals)
    plt.xticks(rotation=25, ha="right")
    plt.title("Baseline: Fairlearn fairness metrics")
    plt.tight_layout()
    plt.savefig("reports/figures/reweighed_fairness_metrics.png", dpi=200)
    plt.close()

    print("Phase 3 complete.")
    print(json.dumps(fairness, indent=2))


if __name__ == "__main__":
    main()

