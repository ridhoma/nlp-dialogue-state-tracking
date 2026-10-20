"""
Data science utilities for model evaluation, metrics, and tabular analysis.
"""

from data_science import features, visualisation
from data_science.evaluation import (
    MultiLabelEvaluationResult,
    evaluate_multilabel,
)
from data_science.visualisation import (
    plot_correlation_matrix,
    plot_distribution,
)

__all__ = [
    "MultiLabelEvaluationResult",
    "evaluate_multilabel",
    "features",
    "plot_correlation_matrix",
    "plot_distribution",
    "visualisation",
]
