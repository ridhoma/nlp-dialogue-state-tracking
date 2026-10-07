"""
Unit tests for data_science.evaluation multi-label classification utilities.
"""

import pandas as pd
import pytest

from data_science.evaluation import (
    MultiLabelEvaluationResult,
    evaluate_multilabel,
)


def test_evaluate_multilabel_perfect():
    """Verify perfect match gives 1.0 on all metrics."""
    df = pd.DataFrame(
        {
            "true_slots": [["email", "phone"], ["order_id"], []],
            "inferred_slots": [["email", "phone"], ["order_id"], []],
        }
    )

    result = evaluate_multilabel(df, true_col="true_slots", pred_col="inferred_slots")
    assert isinstance(result, MultiLabelEvaluationResult)
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
    # Turn 0: true=[A, B], pred=[A]       -> TP(A)=1, FN(B)=1
    # Turn 1: true=[C],    pred=[C, D]    -> TP(C)=1, FP(D)=1
    # Turn 2: true=[],     pred=[]        -> True negative on all
    y_true = [["A", "B"], ["C"], []]
    y_pred = [["A"], ["C", "D"], []]

    result = evaluate_multilabel(y_true=y_true, y_pred=y_pred)
    assert result.num_samples == 3
    assert result.num_classes == 4  # A, B, C, D

    per_lbl = result.per_label.set_index("label")
    # A: TP=1, FP=0, FN=0 -> Precision=1, Recall=1
    assert per_lbl.loc["A", "tp"] == 1
    assert per_lbl.loc["A", "fn"] == 0
    assert per_lbl.loc["A", "precision"] == 1.0

    # B: TP=0, FP=0, FN=1 -> Recall=0
    assert per_lbl.loc["B", "fn"] == 1
    assert per_lbl.loc["B", "recall"] == 0.0

    # D: TP=0, FP=1, FN=0 -> Precision=0
    assert per_lbl.loc["D", "fp"] == 1
    assert per_lbl.loc["D", "precision"] == 0.0

    # Subset accuracy: Turn 2 matched exactly (empty), Turns 0 and 1 differed
    assert pytest.approx(result.to_dict()["exact_match_ratio"]) == 1 / 3


def test_evaluate_multilabel_html_repr():
    """Verify interactive HTML dashboard generation."""
    df = pd.DataFrame(
        {
            "true": [["slot_a", "slot_b"], ["slot_a"]],
            "pred": [["slot_a"], ["slot_b"]],
        }
    )
    result = evaluate_multilabel(df, true_col="true", pred_col="pred")
    html_out = result._repr_html_()

    assert "Multi-Label Classification Evaluation" in html_out
    assert "slot_a" in html_out
    assert "slot_b" in html_out
    assert "Exact Match" in html_out


def test_evaluate_multilabel_drop_zero_support():
    """Verify drop_zero_support=True excludes labels never present in ground truth."""
    y_true = [["A", "B"], ["C"], []]
    y_pred = [["A"], ["C", "D"], []]  # 'D' was never in ground truth

    # Default (drop_zero_support=False): 'D' is kept with support=0
    res_keep = evaluate_multilabel(
        y_true=y_true, y_pred=y_pred, drop_zero_support=False
    )
    assert res_keep.num_classes == 4
    assert "D" in res_keep.per_label["label"].values
    assert res_keep.per_label.set_index("label").loc["D", "support"] == 0

    # With drop_zero_support=True: 'D' is dropped, only {A, B, C} evaluated
    res_drop = evaluate_multilabel(y_true=y_true, y_pred=y_pred, drop_zero_support=True)
    assert res_drop.num_classes == 3
    assert "D" not in res_drop.per_label["label"].values
    assert (res_drop.per_label["support"] > 0).all()


def test_evaluation_multilabel_alias():
    """Verify evaluation_multilabel alias works identically."""
    from data_science.evaluation import evaluation_multilabel

    y_true = [["A"], ["B"]]
    y_pred = [["A"], ["B"]]
    res = evaluation_multilabel(y_true=y_true, y_pred=y_pred)
    assert res.to_dict()["exact_match_ratio"] == 1.0
