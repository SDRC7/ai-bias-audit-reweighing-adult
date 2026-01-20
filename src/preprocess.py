from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def _strip_strings_inplace(df: pd.DataFrame) -> pd.DataFrame:
    # OpenML returns many categoricals as pandas "category"; convert safely
    for col in df.columns:
        if str(df[col].dtype) in ("object", "category"):
            df[col] = df[col].astype(str).str.strip()
            # Convert common missing markers to actual NaN
            df[col] = df[col].replace({"?": np.nan, "None": np.nan, "nan": np.nan})
    return df


def add_binary_columns(df_raw: pd.DataFrame, cfg) -> pd.DataFrame:
    """
    Adds:
      - cfg.label_col_bin: income_bin
      - cfg.sensitive_col_bin: sex_bin
    using cfg.income_mapping and cfg.sex_mapping.
    """
    df = df_raw.copy()
    df = _strip_strings_inplace(df)

    # Label mapping (handles >50K and >50K. variants)
    y_raw = df[cfg.raw_label_col].astype(str).str.strip()
    df[cfg.label_col_bin] = y_raw.map(cfg.income_mapping)

    # Sensitive mapping
    s_raw = df[cfg.raw_sensitive_col].astype(str).str.strip()
    df[cfg.sensitive_col_bin] = s_raw.map(cfg.sex_mapping)

    # Guards: all mapped
    if df[cfg.label_col_bin].isna().any():
        bad = sorted(df.loc[df[cfg.label_col_bin].isna(), cfg.raw_label_col].unique().tolist())
        raise AssertionError(f"Unmapped label values in {cfg.raw_label_col}: {bad}")

    if df[cfg.sensitive_col_bin].isna().any():
        bad = sorted(df.loc[df[cfg.sensitive_col_bin].isna(), cfg.raw_sensitive_col].unique().tolist())
        raise AssertionError(f"Unmapped sensitive values in {cfg.raw_sensitive_col}: {bad}")

    df[cfg.label_col_bin] = df[cfg.label_col_bin].astype(int)
    df[cfg.sensitive_col_bin] = df[cfg.sensitive_col_bin].astype(int)

    return df


def build_preprocessor(df_with_bins: pd.DataFrame, cfg) -> ColumnTransformer:
    drop_cols = [cfg.raw_label_col, cfg.label_col_bin, cfg.raw_sensitive_col, cfg.sensitive_col_bin]
    feature_cols = [c for c in df_with_bins.columns if c not in drop_cols]

    X_df = df_with_bins[feature_cols].copy()

    cat_cols = [c for c in X_df.columns if str(X_df[c].dtype) in ("object", "category")]
    num_cols = [c for c in X_df.columns if c not in cat_cols]

    # sklearn>=1.2 uses sparse_output; keep compatibility by trying
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", ohe),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def preprocess_fit_transform(df_raw: pd.DataFrame, cfg):
    """
    Returns:
      X (np.ndarray), y (np.ndarray), sex_bin (np.ndarray), preprocessor (fitted)
    """
    df = add_binary_columns(df_raw, cfg)

    y = df[cfg.label_col_bin].to_numpy(dtype=int)
    sex_bin = df[cfg.sensitive_col_bin].to_numpy(dtype=int)

    # Binary guards
    if not set(np.unique(y)).issubset({0, 1}):
        raise AssertionError(f"{cfg.label_col_bin} must be in {{0,1}}; got {sorted(set(np.unique(y)))}")
    if not set(np.unique(sex_bin)).issubset({0, 1}):
        raise AssertionError(f"{cfg.sensitive_col_bin} must be in {{0,1}}; got {sorted(set(np.unique(sex_bin)))}")

    preprocessor = build_preprocessor(df, cfg)

    drop_cols = [cfg.raw_label_col, cfg.label_col_bin, cfg.raw_sensitive_col, cfg.sensitive_col_bin]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X_df = df[feature_cols].copy()

    X = preprocessor.fit_transform(X_df)

    # No-NaN guard after preprocessing
    if np.isnan(X).any():
        raise AssertionError("Found NaNs in X after preprocessing (imputation/encoding failed).")

    # Shape alignment guard
    if not (len(X) == len(y) == len(sex_bin)):
        raise AssertionError(f"Length mismatch: len(X)={len(X)}, len(y)={len(y)}, len(sex)={len(sex_bin)}")

    return X, y, sex_bin, preprocessor

