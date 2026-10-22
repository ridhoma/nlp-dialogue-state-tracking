"""
Data science utilities for model evaluation, metrics, and tabular analysis.
"""

from data_science import evaluations, features, visualisations
from data_science.evaluations import (
    MultiLabelEvaluationResult,
    evaluate_multilabel,
)
from data_science.visualisations import (
    plot_correlation_matrix,
    plot_distribution,
)

__all__ = [
    "MultiLabelEvaluationResult",
    "evaluate_multilabel",
    "evaluations",
    "features",
    "plot_correlation_matrix",
    "plot_distribution",
    "visualisations",
]
