from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ProjectConfig:
    # Repro
    random_seed: int = 0
    test_size: float = 0.20

    # Dataset identity
    dataset_name: str = "adult_census_income_openml_v2"

    # Raw column names (OpenML Adult v2)
    raw_label_col: str = "class"
    raw_sensitive_col: str = "sex"

    # Engineered binary columns
    label_col_bin: str = "income_bin"      # 1 => >50K, 0 => <=50K
    sensitive_col_bin: str = "sex_bin"     # 1 => Male, 0 => Female

    # Explicit encodings (locked)
    sex_mapping: Dict[str, int] = None
    income_mapping: Dict[str, int] = None

    # AIF360 label definitions
    favorable_label: int = 1
    unfavorable_label: int = 0

    # Group definitions (AIF360 expects list of dicts)
    privileged_groups: List[Dict[str, int]] = None
    unprivileged_groups: List[Dict[str, int]] = None

    # Metric names (for reporting)
    fairness_metrics: List[str] = None

    # Success criteria
    target_dp_diff_reduction: float = 0.30  # >=30% reduction
    max_accuracy_drop_pp: float = 3.0       # <=3 percentage-point drop allowed
    robustness_n_seeds: int = 5             # robustness option A


def get_config() -> ProjectConfig:
    # Mappings (Adult sometimes has trailing '.' variants)
    sex_mapping = {"Female": 0, "Male": 1}
    income_mapping = {"<=50K": 0, ">50K": 1, "<=50K.": 0, ">50K.": 1}

    privileged_groups = [{"sex_bin": 1}]     # Male
    unprivileged_groups = [{"sex_bin": 0}]   # Female

    fairness_metrics = [
        "selection_rate (by group)",
        "demographic_parity_difference",
        "demographic_parity_ratio",
        "equalized_odds_difference",
        "equalized_odds_ratio",
    ]

    return ProjectConfig(
        sex_mapping=sex_mapping,
        income_mapping=income_mapping,
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        fairness_metrics=fairness_metrics,
    )
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ProjectConfig:
    # Repro
    random_seed: int = 0
    test_size: float = 0.20

    # Dataset identity
    dataset_name: str = "adult_census_income_openml_v2"

    # Raw column names (OpenML Adult v2)
    raw_label_col: str = "class"
    raw_sensitive_col: str = "sex"

    # Engineered binary columns
    label_col_bin: str = "income_bin"      # 1 => >50K, 0 => <=50K
    sensitive_col_bin: str = "sex_bin"     # 1 => Male, 0 => Female

    # Explicit encodings (locked)
    sex_mapping: Dict[str, int] = None
    income_mapping: Dict[str, int] = None

    # AIF360 label definitions
    favorable_label: int = 1
    unfavorable_label: int = 0

    # Group definitions (AIF360 expects list of dicts)
    privileged_groups: List[Dict[str, int]] = None
    unprivileged_groups: List[Dict[str, int]] = None

    # Metric names (for reporting)
    fairness_metrics: List[str] = None

    # Success criteria
    target_dp_diff_reduction: float = 0.30  # >=30% reduction
    max_accuracy_drop_pp: float = 3.0       # <=3 percentage-point drop allowed
    robustness_n_seeds: int = 5             # robustness option A


def get_config() -> ProjectConfig:
    # Mappings (Adult sometimes has trailing '.' variants)
    sex_mapping = {"Female": 0, "Male": 1}
    income_mapping = {"<=50K": 0, ">50K": 1, "<=50K.": 0, ">50K.": 1}

    privileged_groups = [{"sex_bin": 1}]     # Male
    unprivileged_groups = [{"sex_bin": 0}]   # Female

    fairness_metrics = [
        "selection_rate (by group)",
        "demographic_parity_difference",
        "demographic_parity_ratio",
        "equalized_odds_difference",
        "equalized_odds_ratio",
    ]

    return ProjectConfig(
        sex_mapping=sex_mapping,
        income_mapping=income_mapping,
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        fairness_metrics=fairness_metrics,
    )

