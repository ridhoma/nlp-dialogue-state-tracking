"""
Econometric Linear/Logit (`LinearModel`) and Gradient Boosted Decision Tree
(`XGBoostModel`) estimator classes.
"""

from data_science.models import gbdt, linear
from data_science.models.gbdt import (
    DEFAULT_XGB_PARAMS,
    XGBoostModel,
    compute_native_shap_interactions,
    compute_native_treeshap,
)
from data_science.models.linear import (
    LinearModel,
    compute_vif,
)

__all__ = [
    "DEFAULT_XGB_PARAMS",
    "LinearModel",
    "XGBoostModel",
    "compute_native_shap_interactions",
    "compute_native_treeshap",
    "compute_vif",
    "gbdt",
    "linear",
]
