"""
Binary and Multi-Label Classification Evaluation Utilities.

Provides comprehensive evaluation metrics for:
- Binary classification & probability calibration (`evaluate_binary_classification`)
- Multi-label classification (`evaluate_multilabel_classification`)
with structured result containers (`BinaryClassificationResult`,
`MultiLabelClassificationResult`) and interactive HTML dashboards for Jupyter Notebooks.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    hamming_loss,
    jaccard_score,
    log_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import MultiLabelBinarizer


class BinaryClassificationResult:
    """
    Encapsulates binary classification and probability calibration evaluation results.
    Provides structured DataFrames (`.summary`, `.per_label`) and rich HTML rendering
    for Jupyter notebooks, mirroring `MultiLabelClassificationResult`.
    """

    def __init__(
        self,
        summary_metrics: pd.DataFrame,
        per_label_metrics: pd.DataFrame,
        num_samples: int,
        threshold: float = 0.5,
    ):
        self.summary = summary_metrics
        self.per_label = per_label_metrics
        self.num_samples = num_samples
        self.threshold = threshold

    def to_dict(self) -> dict[str, float]:
        """Convert all summary metrics into a flat dictionary."""
        return {
            str(k): float(v)
            for k, v in self.summary.set_index("metric")["value"].to_dict().items()
        }

    def print_summary(self) -> None:
        """Print formatted text summary to standard output."""
        print(
            f"=== Binary Classification Evaluation Summary (N={self.num_samples}, threshold={self.threshold:.2f}) ==="
        )
        print(self.summary.to_string(index=False))
        print("\n=== Per-Class Breakdown ===")
        print(self.per_label.to_string(index=False))

    def __repr__(self) -> str:
        m = self.to_dict()
        roc_auc = m.get("roc_auc", float("nan"))
        pr_auc = m.get("pr_auc", float("nan"))
        f1 = m.get("f1", 0.0)
        return (
            f"<BinaryClassificationResult: N={self.num_samples}, "
            f"ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}, F1={f1:.3f}>"
        )

    def _repr_html_(self) -> str:
        """Interactive HTML dashboard for Jupyter Notebooks."""
        from utils.renderings import render_binary_evaluation_html

        return render_binary_evaluation_html(
            summary=self.summary,
            per_label=self.per_label,
            num_samples=self.num_samples,
            threshold=self.threshold,
        )


def evaluate_binary_classification(
    data: pd.DataFrame | None = None,
    true_col: str | None = None,
    prob_col: str | None = None,
    pred_col: str | None = None,
    y_true: Sequence[int | float] | np.ndarray | pd.Series | None = None,
    y_prob: Sequence[float] | np.ndarray | pd.Series | None = None,
    y_pred: Sequence[int | float] | np.ndarray | pd.Series | None = None,
    threshold: float = 0.5,
    class_names: tuple[str, str] = ("0 (negative)", "1 (positive)"),
) -> BinaryClassificationResult:
    """
    Evaluate binary classification and probability calibration performance for target Y in {0, 1}.

    Can be invoked either by passing a DataFrame with column names:
        evaluate_binary_classification(df, true_col="resolved", prob_col="pred_prob")
    Or by directly passing arrays/sequences:
        evaluate_binary_classification(y_true=y, y_prob=probs)

    Parameters
    ----------
    data : pd.DataFrame, optional
        DataFrame containing ground-truth and predicted probability/label columns.
    true_col : str, optional
        Name of the binary ground-truth column (`0` or `1`) in `data`.
    prob_col : str, optional
        Name of the predicted probability column `P(Y = 1 | X)` in `data`.
    pred_col : str, optional
        Name of the hard binary prediction column in `data` (if `prob_col` is not used).
    y_true : array-like, optional
        Ground-truth binary labels in `{0, 1}`.
    y_prob : array-like, optional
        Predicted probabilities `P(Y = 1 | X)`.
    y_pred : array-like, optional
        Hard binary predictions in `{0, 1}` (if `y_prob` is not provided).
    threshold : float, default=0.5
        Probability cutoff used to convert `y_prob` into hard predictions `y_pred`.
    class_names : tuple[str, str], default=("0 (negative)", "1 (positive)")
        Display labels for class `0` and class `1` in `.per_label`.

    Returns
    -------
    BinaryClassificationResult
        Evaluation result containing `.summary` (ROC-AUC, PR-AUC, Brier, Log-Loss,
        Accuracy, Precision, Recall, F1, Specificity) and `.per_label` breakdown.
    """
    if data is not None:
        if true_col is None:
            raise ValueError("When passing `data`, `true_col` must be provided.")
        y_t = np.asarray(data[true_col], dtype=int)
        y_p = np.asarray(data[prob_col], dtype=float) if prob_col is not None else None
        y_hat = np.asarray(data[pred_col], dtype=int) if pred_col is not None else None
    elif y_true is not None:
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.asarray(y_prob, dtype=float) if y_prob is not None else None
        y_hat = np.asarray(y_pred, dtype=int) if y_pred is not None else None
    else:
        raise ValueError(
            "Must provide either (`data`, `true_col`, `prob_col`/`pred_col`) "
            "or (`y_true`, `y_prob`/`y_pred`)."
        )

    if y_p is None and y_hat is None:
        raise ValueError("Must provide either predicted probabilities (`prob_col`/`y_prob`) or labels (`pred_col`/`y_pred`).")

    num_samples = len(y_t)
    if num_samples == 0:
        raise ValueError("Cannot evaluate empty predictions dataset.")

    if y_p is not None and len(y_p) != num_samples:
        raise ValueError(f"Length mismatch: len(y_true)={num_samples} vs len(y_prob)={len(y_p)}")
    if y_hat is not None and len(y_hat) != num_samples:
        raise ValueError(f"Length mismatch: len(y_true)={num_samples} vs len(y_pred)={len(y_hat)}")

    if y_hat is None and y_p is not None:
        y_hat = (y_p >= threshold).astype(int)

    assert y_hat is not None

    # Probability calibration & ranking metrics
    if y_p is not None:
        y_p_clipped = np.clip(y_p, 1e-7, 1.0 - 1e-7)
        has_both_classes = len(np.unique(y_t)) > 1
        roc_auc = float(roc_auc_score(y_t, y_p)) if has_both_classes else float("nan")
        pr_auc = float(average_precision_score(y_t, y_p)) if has_both_classes else float("nan")
        brier = float(brier_score_loss(y_t, y_p_clipped))
        lloss = float(log_loss(y_t, y_p_clipped, labels=[0, 1]))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")
        brier = float("nan")
        lloss = float("nan")

    # Thresholded classification metrics
    acc = float(accuracy_score(y_t, y_hat))
    prec = float(precision_score(y_t, y_hat, zero_division=0))
    rec = float(recall_score(y_t, y_hat, zero_division=0))
    f1 = float(f1_score(y_t, y_hat, zero_division=0))

    tn = int(np.sum((y_t == 0) & (y_hat == 0)))
    fp = int(np.sum((y_t == 0) & (y_hat == 1)))
    fn = int(np.sum((y_t == 1) & (y_hat == 0)))
    tp = int(np.sum((y_t == 1) & (y_hat == 1)))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    summary_records = [
        {"metric": "roc_auc", "value": roc_auc},
        {"metric": "pr_auc", "value": pr_auc},
        {"metric": "brier_score", "value": brier},
        {"metric": "log_loss", "value": lloss},
        {"metric": "accuracy", "value": acc},
        {"metric": "precision", "value": prec},
        {"metric": "recall", "value": rec},
        {"metric": "f1", "value": f1},
        {"metric": "specificity", "value": specificity},
    ]
    summary_df = pd.DataFrame(summary_records)

    # Per-class breakdown (Class 0 and Class 1)
    prec_per, rec_per, f1_per, sup_per = precision_recall_fscore_support(
        y_t, y_hat, labels=[0, 1], average=None, zero_division=0
    )
    per_label_rows = [
        {
            "label": class_names[0],
            "support": int(sup_per[0]),
            "predicted": int(np.sum(y_hat == 0)),
            "tp": tn,
            "fp": fn,
            "fn": fp,
            "precision": float(prec_per[0]),
            "recall": float(rec_per[0]),
            "f1": float(f1_per[0]),
        },
        {
            "label": class_names[1],
            "support": int(sup_per[1]),
            "predicted": int(np.sum(y_hat == 1)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(prec_per[1]),
            "recall": float(rec_per[1]),
            "f1": float(f1_per[1]),
        },
    ]
    per_label_df = pd.DataFrame(per_label_rows)

    return BinaryClassificationResult(
        summary_metrics=summary_df,
        per_label_metrics=per_label_df,
        num_samples=num_samples,
        threshold=threshold,
    )


def _normalize_label_collection(raw_val: Any) -> list[str]:
    """Normalizes arbitrary label representations (lists, arrays, strings, NaN) into clean list of strings."""
    if raw_val is None:
        return []
    if isinstance(raw_val, float) and pd.isna(raw_val):
        return []
    if isinstance(raw_val, str):
        val_str = raw_val.strip()
        if not val_str:
            return []
        if "," in val_str:
            return [s.strip() for s in val_str.split(",") if s.strip()]
        return [val_str]
    if isinstance(raw_val, (list, tuple, set, np.ndarray)) or hasattr(
        raw_val, "__iter__"
    ):
        clean = []
        for item in raw_val:
            if item is not None and not (isinstance(item, float) and pd.isna(item)):
                s = str(item).strip()
                if s:
                    clean.append(s)
        return clean
    return [str(raw_val).strip()]


class MultiLabelClassificationResult:
    """
    Encapsulates multi-label classification evaluation results.
    Provides structured DataFrames and rich HTML rendering for Jupyter notebooks.
    """

    def __init__(
        self,
        summary_metrics: pd.DataFrame,
        per_label_metrics: pd.DataFrame,
        num_samples: int,
        num_classes: int,
    ):
        self.summary = summary_metrics
        self.per_label = per_label_metrics
        self.num_samples = num_samples
        self.num_classes = num_classes

    def to_dict(self) -> dict[str, Any]:
        """Convert all summary metrics into a flat dictionary."""
        return self.summary.set_index("metric")["value"].to_dict()

    def print_summary(self) -> None:
        """Print formatted text summary to standard output."""
        print(
            f"=== Multi-Label Evaluation Summary (N={self.num_samples}, Classes={self.num_classes}) ==="
        )
        print(self.summary.to_string(index=False))
        print("\n=== Per-Label Metrics (Top 15 by Support) ===")
        print(self.per_label.head(15).to_string(index=False))

    def __repr__(self) -> str:
        micro_f1 = self.to_dict().get("micro_f1", 0.0)
        macro_f1 = self.to_dict().get("macro_f1", 0.0)
        return f"<MultiLabelClassificationResult: N={self.num_samples}, Micro-F1={micro_f1:.3f}, Macro-F1={macro_f1:.3f}>"

    def _repr_html_(self) -> str:
        """Interactive HTML dashboard for Jupyter Notebooks."""
        from utils.renderings import render_evaluation_html

        return render_evaluation_html(
            summary=self.summary,
            per_label=self.per_label,
            num_samples=self.num_samples,
            num_classes=self.num_classes,
        )


def evaluate_multilabel_classification(
    data: pd.DataFrame | None = None,
    true_col: str | None = None,
    pred_col: str | None = None,
    y_true: Sequence[Iterable[str]] | None = None,
    y_pred: Sequence[Iterable[str]] | None = None,
    classes: Sequence[str] | None = None,
    drop_zero_support: bool = False,
) -> MultiLabelClassificationResult:
    """
    Evaluates multi-label classification performance comparing true vs predicted label sets.

    Can be invoked either by passing a DataFrame with column names:
        evaluate_multilabel_classification(df_evals, true_col="true_slots", pred_col="inferred_slots")
    Or by directly passing sequences of labels:
        evaluate_multilabel_classification(y_true=true_labels, y_pred=pred_labels)

    Parameters
    ----------
    data : pd.DataFrame, optional
        DataFrame containing true and predicted label columns.
    true_col : str, optional
        Name of the column containing ground-truth label collections.
    pred_col : str, optional
        Name of the column containing predicted/inferred label collections.
    y_true : Sequence[Iterable[str]], optional
        Sequence of ground truth label iterables.
    y_pred : Sequence[Iterable[str]], optional
        Sequence of predicted label iterables.
    classes : Sequence[str], optional
        Explicit class list to evaluate. If omitted, the union of all observed
        classes in `y_true` and `y_pred` is used.
    drop_zero_support : bool, default=False
        If True, labels with 0 support in ground truth (`y_true`) are excluded
        from evaluation (both per-label table and aggregate metrics). Useful
        when models predict against a broad candidate ontology but the test
        dataset only annotates a subset of labels.

    Returns
    -------
    MultiLabelClassificationResult
        Evaluation result containing `.summary` (global metrics) and
        `.per_label` (detailed precision/recall breakdown), with interactive HTML display.
    """
    # 1. Resolve inputs
    if data is not None:
        if true_col is None or pred_col is None:
            raise ValueError(
                "When passing `data` DataFrame, both `true_col` and `pred_col` must be provided."
            )
        raw_true = data[true_col].tolist()
        raw_pred = data[pred_col].tolist()
    elif y_true is not None and y_pred is not None:
        raw_true = list(y_true)
        raw_pred = list(y_pred)
    else:
        raise ValueError(
            "Must provide either (`data`, `true_col`, `pred_col`) or (`y_true`, `y_pred`)."
        )

    if len(raw_true) != len(raw_pred):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(raw_true)} vs len(y_pred)={len(raw_pred)}"
        )

    # 2. Normalize label representations
    norm_true = [_normalize_label_collection(item) for item in raw_true]
    norm_pred = [_normalize_label_collection(item) for item in raw_pred]

    num_samples = len(norm_true)
    if num_samples == 0:
        raise ValueError("Cannot evaluate empty predictions dataset.")

    # 3. Determine target evaluation classes
    true_labels_present = set()
    for s in norm_true:
        true_labels_present.update(s)

    if drop_zero_support:
        if classes is not None:
            class_list = [c for c in classes if c in true_labels_present]
        else:
            class_list = sorted(true_labels_present)
        # Filter predicted labels to target classes to prevent scikit-learn warnings
        target_set = set(class_list)
        norm_pred = [[lbl for lbl in s if lbl in target_set] for s in norm_pred]
    else:
        if classes is not None:
            class_list = list(classes)
        else:
            all_labels = set(true_labels_present)
            for s in norm_pred:
                all_labels.update(s)
            class_list = sorted(all_labels)

    mlb = MultiLabelBinarizer(classes=class_list)
    Y_true = mlb.fit_transform(norm_true)
    Y_pred = mlb.transform(norm_pred)
    target_classes = list(mlb.classes_)
    num_classes = len(target_classes)

    if num_classes == 0:
        # Trivial edge case: zero classes observed
        summary_df = pd.DataFrame([{"metric": "exact_match_ratio", "value": 1.0}])
        per_label_df = pd.DataFrame(
            columns=[
                "label",
                "support",
                "predicted",
                "tp",
                "fp",
                "fn",
                "precision",
                "recall",
                "f1",
            ]
        )
        return MultiLabelClassificationResult(summary_df, per_label_df, num_samples, 0)

    # 4. Compute per-label metrics
    prec_per, rec_per, f1_per, sup_per = precision_recall_fscore_support(
        Y_true, Y_pred, average=None, zero_division=0
    )

    per_label_rows = []
    for idx, c_name in enumerate(target_classes):
        yt = Y_true[:, idx]
        yp = Y_pred[:, idx]
        tp = int(np.sum((yt == 1) & (yp == 1)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        pred_count = int(np.sum(yp == 1))
        support_count = int(sup_per[idx])

        per_label_rows.append(
            {
                "label": c_name,
                "support": support_count,
                "predicted": pred_count,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": float(prec_per[idx]),
                "recall": float(rec_per[idx]),
                "f1": float(f1_per[idx]),
            }
        )

    per_label_df = (
        pd.DataFrame(per_label_rows)
        .sort_values(by=["support", "f1"], ascending=[False, False])
        .reset_index(drop=True)
    )

    # 5. Compute global summary metrics
    exact_match = float(accuracy_score(Y_true, Y_pred))
    h_loss = float(hamming_loss(Y_true, Y_pred))
    jaccard_samples = float(
        jaccard_score(Y_true, Y_pred, average="samples", zero_division=0)
    )

    # Micro, Macro, Weighted, Samples averages
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average="weighted", zero_division=0
    )
    samples_p, samples_r, samples_f1, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average="samples", zero_division=0
    )

    summary_records = [
        {"metric": "exact_match_ratio", "value": exact_match},
        {"metric": "hamming_loss", "value": h_loss},
        {"metric": "jaccard_similarity_samples", "value": jaccard_samples},
        {"metric": "micro_precision", "value": float(micro_p)},
        {"metric": "micro_recall", "value": float(micro_r)},
        {"metric": "micro_f1", "value": float(micro_f1)},
        {"metric": "macro_precision", "value": float(macro_p)},
        {"metric": "macro_recall", "value": float(macro_r)},
        {"metric": "macro_f1", "value": float(macro_f1)},
        {"metric": "weighted_precision", "value": float(weighted_p)},
        {"metric": "weighted_recall", "value": float(weighted_r)},
        {"metric": "weighted_f1", "value": float(weighted_f1)},
        {"metric": "samples_precision", "value": float(samples_p)},
        {"metric": "samples_recall", "value": float(samples_r)},
        {"metric": "samples_f1", "value": float(samples_f1)},
    ]

    summary_df = pd.DataFrame(summary_records)

    return MultiLabelClassificationResult(
        summary_metrics=summary_df,
        per_label_metrics=per_label_df,
        num_samples=num_samples,
        num_classes=num_classes,
    )
