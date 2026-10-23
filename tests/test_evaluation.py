"""
Unit tests for `data_science.evaluations` binary and multi-label classification utilities.
"""

import pandas as pd
import pytest

from data_science.evaluations import (
    BinaryClassificationResult,
    MultiLabelClassificationResult,
    evaluate_binary_classification,
    evaluate_multilabel_classification,
)


def test_evaluate_binary_classification_arrays_and_df():
    """Verify binary classification metrics and HTML dashboard generation."""
    df = pd.DataFrame(
        {
            "resolved": [1, 1, 0, 0],
            "prob": [0.9, 0.8, 0.2, 0.1],
        }
    )
    res = evaluate_binary_classification(df, true_col="resolved", prob_col="prob")
    assert isinstance(res, BinaryClassificationResult)
    metrics = res.to_dict()
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["specificity"] == 1.0
    assert len(res.per_label) == 2

    html_out = res._repr_html_()
    assert "Binary Classification Evaluation" in html_out
    assert "ROC-AUC" in html_out


def test_evaluate_multilabel_perfect():
    """Verify perfect match gives 1.0 on all metrics."""
    df = pd.DataFrame(
        {
            "true_slots": [["email", "phone"], ["order_id"], []],
            "inferred_slots": [["email", "phone"], ["order_id"], []],
        }
    )

    result = evaluate_multilabel_classification(
        df, true_col="true_slots", pred_col="inferred_slots"
    )
    assert isinstance(result, MultiLabelClassificationResult)
    metrics = result.to_dict()

    assert metrics["exact_match_ratio"] == 1.0
    assert metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["hamming_loss"] == 0.0

    # Check per-label dataframe
    per_lbl = result.per_label.set_index("label")
    assert per_lbl.loc["email", "f1"] == 1.0
    assert per_lbl.loc["email", "support"] == 1
    assert per_lbl.loc["order_id", "tp"] == 1
    assert per_lbl.loc["order_id", "fp"] == 0


def test_evaluate_multilabel_partial():
    """Verify known true/false positive counts."""
    y_true = [["A", "B"], ["C"], []]
    y_pred = [["A"], ["C", "D"], []]

    result = evaluate_multilabel_classification(y_true=y_true, y_pred=y_pred)
    assert result.num_samples == 3
    assert result.num_classes == 4  # A, B, C, D

    per_lbl = result.per_label.set_index("label")
    assert per_lbl.loc["A", "tp"] == 1
    assert per_lbl.loc["A", "fn"] == 0
    assert per_lbl.loc["A", "precision"] == 1.0

    assert per_lbl.loc["B", "fn"] == 1
    assert per_lbl.loc["B", "recall"] == 0.0

    assert per_lbl.loc["D", "fp"] == 1
    assert per_lbl.loc["D", "precision"] == 0.0

    assert pytest.approx(result.to_dict()["exact_match_ratio"]) == 1 / 3


def test_evaluate_multilabel_html_repr():
    """Verify interactive HTML dashboard generation."""
    df = pd.DataFrame(
        {
            "true": [["slot_a", "slot_b"], ["slot_a"]],
            "pred": [["slot_a"], ["slot_b"]],
        }
    )
    result = evaluate_multilabel_classification(df, true_col="true", pred_col="pred")
    html_out = result._repr_html_()

    assert "Multi-Label Classification Evaluation" in html_out
    assert "slot_a" in html_out
    assert "slot_b" in html_out
    assert "Exact Match" in html_out


def test_evaluate_multilabel_drop_zero_support():
    """Verify drop_zero_support=True excludes labels never present in ground truth."""
    y_true = [["A", "B"], ["C"], []]
    y_pred = [["A"], ["C", "D"], []]  # 'D' was never in ground truth

    res_keep = evaluate_multilabel_classification(
        y_true=y_true, y_pred=y_pred, drop_zero_support=False
    )
    assert res_keep.num_classes == 4
    assert "D" in res_keep.per_label["label"].values
    assert res_keep.per_label.set_index("label").loc["D", "support"] == 0

    res_drop = evaluate_multilabel_classification(
        y_true=y_true, y_pred=y_pred, drop_zero_support=True
    )
    assert res_drop.num_classes == 3
    assert "D" not in res_drop.per_label["label"].values
    assert (res_drop.per_label["support"] > 0).all()
