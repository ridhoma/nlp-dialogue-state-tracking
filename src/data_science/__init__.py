"""
Data science utilities for model evaluation, metrics, feature encoding, modeling, and visualisation.
"""

from data_science import evaluations, features, models, visualisations
from data_science.evaluations import (
    BinaryClassificationResult,
    MultiLabelClassificationResult,
    evaluate_binary_classification,
    evaluate_multilabel_classification,
)
from data_science.models import LinearModel, XGBoostModel
from data_science.visualisations import (
    plot_correlation_matrix,
    plot_distribution,
)

__all__ = [
    "BinaryClassificationResult",
    "LinearModel",
    "MultiLabelClassificationResult",
    "XGBoostModel",
    "evaluate_binary_classification",
    "evaluate_multilabel_classification",
    "evaluations",
    "features",
    "models",
    "plot_correlation_matrix",
    "plot_distribution",
    "visualisations",
]
