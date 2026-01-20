import pandas as pd
from sklearn.datasets import fetch_openml

EXPECTED_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "class"
]

def load_adult_openml(data_home: str = "data/raw/openml") -> pd.DataFrame:
    """
    Deterministic load: OpenML Adult dataset name='adult', version=2, as_frame=True.
    Returns a raw DataFrame with OpenML raw columns (including 'sex' and 'class').
    """
    bunch = fetch_openml(
        name="adult",
        version=2,
        as_frame=True,
        cache=True,
        data_home=data_home,
    )
    df = bunch.frame.copy()

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise AssertionError(
            f"OpenML Adult columns mismatch. Missing: {missing}. Got: {list(df.columns)}"
        )

    # Keep only expected columns and keep ordering stable
    df = df[EXPECTED_COLUMNS].copy()

    # Deterministic row order for downstream index-based splits
    df = df.reset_index(drop=True)

    return df

