import os
import random
from pathlib import Path

import numpy as np


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def assert_binary_series(arr, name: str) -> None:
    vals = set(np.unique(arr))
    if not vals.issubset({0, 1}):
        raise AssertionError(f"{name} must be binary in {{0,1}}; got values={sorted(list(vals))}")

