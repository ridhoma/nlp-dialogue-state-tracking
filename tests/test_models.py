"""
Unit tests for `data_science.models.LinearModel` and `data_science.models.XGBoostModel`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from data_science.evaluations import BinaryClassificationResult
from data_science.models import LinearModel, XGBoostModel, compute_vif


def _make_synthetic_dataset(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, size=n)
    x2 = rng.normal(0.0, 1.0, size=n)
    x3 = 0.5 * x1 + rng.normal(0.0, 0.8, size=n)
    logits = -0.3 + 1.2 * x1 - 0.9 * x2 + 0.6 * (x1 * x2)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, probs, size=n)
    return pd.DataFrame(
        {
            "convo_id": np.arange(n),
            "split": "train",
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "resolved": y,
        }
    )


def test_compute_vif() -> None:
    df = _make_synthetic_dataset(200)
    vif_df = compute_vif(df[["x1", "x2", "x3"]])
    assert list(vif_df.columns) == ["feature", "vif", "multi_r2"]
    assert len(vif_df) == 3
    assert (vif_df["vif"] >= 1.0).all()


def test_linear_model_specify_cv_and_fit() -> None:
    df = _make_synthetic_dataset(300)

    # 1. Specify model
    logit = LinearModel(model_type="logit", cov_type="HC3", target_col="resolved")
    assert not logit.is_fitted

    # 2. Cross-validate
    cv_out = logit.cross_validate(df, n_splits=3)
    assert not logit.is_fitted  # CV does not mutate estimator state
    assert cv_out["oof_preds"].shape == (300,)
    assert isinstance(cv_out["oof_evaluation"], BinaryClassificationResult)
    assert len(cv_out["fold_metrics"]) == 3
    assert cv_out["oof_metrics"]["roc_auc"] > 0.65

    # 3. Fit
    logit.fit(df)
    assert logit.is_fitted
    assert logit.num_obs == 300
    assert "odds_ratio" in logit.coef_table.columns
    assert "marginal_effect_ame" in logit.coef_table.columns
    assert logit.metrics["roc_auc"] > 0.70
    assert isinstance(logit.evaluation, BinaryClassificationResult)

    probs = logit.predict_proba(df)
    assert probs.shape == (300,)
    assert np.all((probs > 0.0) & (probs < 1.0))

    # OLS (Linear Probability Model)
    ols = LinearModel(model_type="ols", cov_type="HC3").fit(df)
    assert ols.is_fitted
    assert "r2" in ols.metrics
    assert "t_stat" in ols.coef_table.columns


def test_xgboost_model_specify_cv_and_fit() -> None:
    df = _make_synthetic_dataset(300)
    train_df = df.iloc[:220].reset_index(drop=True)
    val_df = df.iloc[220:].reset_index(drop=True)

    # 1. Specify model
    xgb_model = XGBoostModel(
        max_depth=3,
        n_estimators=60,
        learning_rate=0.08,
        early_stopping_rounds=15,
        target_col="resolved",
    )
    assert not xgb_model.is_fitted

    # 2. Cross-validate
    cv_xgb = xgb_model.cross_validate(df, n_splits=3)
    assert not xgb_model.is_fitted
    assert cv_xgb["oof_preds"].shape == (300,)
    assert isinstance(cv_xgb["oof_evaluation"], BinaryClassificationResult)
    assert cv_xgb["oof_shap_values"].shape == (300, 3)
    assert len(cv_xgb["cv_importance_table"]) == 3
    assert cv_xgb["oof_metrics"]["roc_auc"] > 0.65

    # 3. Fit with validation early stopping
    xgb_model.fit(train_df, eval_df=val_df)
    assert xgb_model.is_fitted
    assert xgb_model.num_obs == 220
    assert list(xgb_model.shap_values.columns) == ["x1", "x2", "x3"]
    assert xgb_model.shap_values.shape == (220, 3)
    assert "val_roc_auc" in xgb_model.metrics
    assert isinstance(xgb_model.evaluation, BinaryClassificationResult)
    assert isinstance(xgb_model.val_evaluation, BinaryClassificationResult)
    assert len(xgb_model.importance_table) == 3

    inter_df = xgb_model.compute_shap_interactions(val_df, top_k=3)
    assert len(inter_df) == 3
    assert list(inter_df.columns) == ["feature_a", "feature_b", "mean_abs_interaction"]
