"""
Econometric Linear Probability Model (OLS) and Logistic Regression (Logit)
estimator class with HC3 heteroskedasticity-robust inference, Variance Inflation
Factor (VIF) diagnostics, Average Marginal Effects (AME), and cross-validation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from data_science.evaluations import (
    BinaryClassificationResult,
    evaluate_binary_classification,
)


def _significance_stars(p_val: float) -> str:
    """Return standard academic significance stars for a p-value."""
    if np.isnan(p_val):
        return ""
    if p_val < 0.001:
        return "***"
    if p_val < 0.01:
        return "**"
    if p_val < 0.05:
        return "*"
    if p_val < 0.10:
        return "."
    return ""


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Variance Inflation Factor (VIF) for each feature in `X`.

    Automatically prepends an intercept column if not already present so VIFs
    reflect centered multi-variable R^2_j, and returns a DataFrame sorted by
    descending VIF.
    """
    X_clean = X.astype(float).copy()
    has_const = "const" in X_clean.columns
    X_with_const = X_clean if has_const else sm.add_constant(X_clean, has_constant="add")
    mat = X_with_const.to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    for idx, col in enumerate(X_with_const.columns):
        if col == "const":
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            vif_val = float(variance_inflation_factor(mat, idx))
        r2_multi = (
            float(1.0 - (1.0 / vif_val))
            if vif_val >= 1.0 and np.isfinite(vif_val)
            else 1.0
        )
        rows.append(
            {
                "feature": col,
                "vif": vif_val,
                "multi_r2": round(r2_multi, 4),
            }
        )

    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


class LinearModel:
    """
    Econometric OLS (Linear Probability Model) and MLE Logistic Regression (`logit`)
    estimator supporting the workflow:
        specify model -> `.cross_validate(df)` -> `.fit(df)`

    Parameters
    ----------
    model_type : {"ols", "logit"}, default="logit"
        - `"ols"`: Ordinary Least Squares Linear Probability Model (LPM).
        - `"logit"`: Maximum Likelihood Logistic Regression.
    cov_type : str, default="HC3"
        Covariance estimator for standard errors (`"HC3"`, `"HC1"`, or `"nonrobust"`).
    maxiter : int, default=200
        Maximum Newton-Raphson iterations when fitting `"logit"`.
    target_col : str, default="resolved"
        Default target column name when passing a `df` to `.fit()`, `.cross_validate()`,
        or `.evaluate()`.
    feature_cols : Sequence[str] | None, optional
        Explicit feature column list. If `None`, automatically selects all numeric
        columns in `df` except `target_col` and `drop_cols`.
    drop_cols : Sequence[str], default=("convo_id", "split", "flow", "intent")
        Identifier/metadata columns automatically excluded when `feature_cols` is `None`.
    threshold : float, default=0.5
        Probability threshold for binary classification metrics.
    """

    def __init__(
        self,
        *,
        model_type: Literal["ols", "logit"] = "logit",
        cov_type: str = "HC3",
        maxiter: int = 200,
        target_col: str = "resolved",
        feature_cols: Sequence[str] | None = None,
        drop_cols: Sequence[str] = ("convo_id", "split", "flow", "intent"),
        threshold: float = 0.5,
    ) -> None:
        m_type = model_type.strip().lower()
        if m_type not in {"ols", "logit"}:
            raise ValueError(f"Unsupported `model_type='{model_type}'`. Choose 'ols' or 'logit'.")
        self.model_type: Literal["ols", "logit"] = m_type  # type: ignore[assignment]
        self.cov_type = cov_type
        self.maxiter = maxiter
        self.target_col = target_col
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.drop_cols = tuple(drop_cols)
        self.threshold = threshold

        # Learned attributes populated by `.fit()`
        self.fitted_model: Any = None
        self.coef_table: pd.DataFrame = pd.DataFrame()
        self.vif_table: pd.DataFrame = pd.DataFrame()
        self.evaluation: BinaryClassificationResult | None = None
        self.metrics: dict[str, float] = {}
        self.feature_names: list[str] = []
        self.num_obs: int = 0

    @property
    def is_fitted(self) -> bool:
        """Whether `.fit()` has been called on this estimator."""
        return self.fitted_model is not None

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted yet. Call `.fit(df)` first."
            )

    def _resolve_inputs(
        self,
        df: pd.DataFrame | None,
        X: pd.DataFrame | None,
        y: pd.Series | np.ndarray | None,
    ) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
        """Extract numeric feature matrix `X_df`, target array `y_arr`, and feature list."""
        if df is not None:
            if self.target_col not in df.columns:
                raise ValueError(f"Target column '{self.target_col}' not found in `df`.")
            if self.feature_cols is None:
                excluded = {self.target_col, *self.drop_cols}
                feat_list = [
                    c
                    for c in df.columns
                    if c not in excluded and pd.api.types.is_numeric_dtype(df[c])
                ]
            else:
                feat_list = list(self.feature_cols)
            X_df = df[feat_list].astype(float).copy()
            y_arr = df[self.target_col].to_numpy(dtype=float)
        elif X is not None and y is not None:
            feat_list = (
                list(self.feature_cols)
                if self.feature_cols is not None
                else list(X.columns)
            )
            X_df = X[feat_list].astype(float).copy()
            y_arr = np.asarray(y, dtype=float)
        else:
            raise ValueError("Provide either `df` or both `X` and `y`.")

        if X_df.isna().any().any():
            null_cols = X_df.columns[X_df.isna().any()].tolist()
            raise ValueError(f"Feature matrix `X` contains NaN values in columns: {null_cols}")

        return X_df, y_arr, feat_list

    def fit(
        self,
        df: pd.DataFrame | None = None,
        *,
        X: pd.DataFrame | None = None,
        y: pd.Series | np.ndarray | None = None,
    ) -> LinearModel:
        """
        Fit the configured OLS or Logistic Regression model on `df` or `(X, y)`.

        Populates `.coef_table`, `.vif_table`, `.evaluation`, `.metrics`, and
        `.fitted_model` on `self`, and returns `self`.
        """
        X_df, y_arr, feat_list = self._resolve_inputs(df, X, y)

        vif_df = compute_vif(X_df)
        vif_map = dict(zip(vif_df["feature"], vif_df["vif"]))

        X_const = sm.add_constant(X_df, has_constant="add")

        if self.model_type == "ols":
            spec = sm.OLS(y_arr, X_const)
            res = (
                spec.fit(cov_type=self.cov_type)
                if self.cov_type != "nonrobust"
                else spec.fit()
            )
            stat_col_name = "t_stat"
            stat_vals = res.tvalues
            fit_stats = {
                "r2": float(res.rsquared),
                "adj_r2": float(res.rsquared_adj),
                "aic": float(res.aic),
                "bic": float(res.bic),
            }
            ame_map: dict[str, float] = {c: float(res.params[c]) for c in feat_list}
        else:
            spec = sm.Logit(y_arr, X_const)
            res = (
                spec.fit(cov_type=self.cov_type, maxiter=self.maxiter, disp=False)
                if self.cov_type != "nonrobust"
                else spec.fit(maxiter=self.maxiter, disp=False)
            )
            stat_col_name = "z_stat"
            stat_vals = res.tvalues
            fit_stats = {
                "pseudo_r2_mcfadden": float(res.prsquared),
                "log_likelihood": float(res.llf),
                "aic": float(res.aic),
                "bic": float(res.bic),
            }
            p_hat = np.asarray(res.predict(X_const), dtype=float)
            mean_dp_deta = float(np.mean(p_hat * (1.0 - p_hat)))
            ame_map = {c: float(res.params[c]) * mean_dp_deta for c in feat_list}

        ci = res.conf_int()
        coef_rows: list[dict[str, Any]] = []
        for term in X_const.columns:
            beta = float(res.params[term])
            se = float(res.bse[term])
            stat_v = float(stat_vals[term])
            p_val = float(res.pvalues[term])
            ci_low = float(ci.loc[term, 0])
            ci_high = float(ci.loc[term, 1])

            row_dict: dict[str, Any] = {
                "term": term,
                "coef": beta,
                "std_err": se,
                stat_col_name: stat_v,
                "p_value": p_val,
                "sig": _significance_stars(p_val),
                "ci_2.5%": ci_low,
                "ci_97.5%": ci_high,
            }
            if self.model_type == "logit":
                row_dict["odds_ratio"] = (
                    float(np.exp(np.clip(beta, -50, 50))) if term != "const" else np.nan
                )
                row_dict["marginal_effect_ame"] = ame_map.get(term, np.nan)
            row_dict["vif"] = vif_map.get(term, np.nan)
            coef_rows.append(row_dict)

        in_sample_probs = np.asarray(res.predict(X_const), dtype=float)
        eval_res = evaluate_binary_classification(
            y_true=y_arr, y_prob=in_sample_probs, threshold=self.threshold
        )

        self.fitted_model = res
        self.coef_table = pd.DataFrame(coef_rows)
        self.vif_table = vif_df
        self.evaluation = eval_res
        self.metrics = {**fit_stats, **eval_res.to_dict()}
        self.feature_names = feat_list
        self.num_obs = int(len(y_arr))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict resolution probabilities `P(Y = 1 | X)` on `X`.
        For OLS (LPM), returns raw linear predictions.
        """
        self._check_is_fitted()
        X_sub = X[self.feature_names].astype(float).copy()
        X_const = sm.add_constant(X_sub, has_constant="add")
        return np.asarray(self.fitted_model.predict(X_const), dtype=float)

    def predict(
        self,
        X: pd.DataFrame,
        *,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Predict binary labels `{0, 1}` on `X`."""
        thresh = self.threshold if threshold is None else threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def evaluate(
        self,
        df: pd.DataFrame | None = None,
        *,
        X: pd.DataFrame | None = None,
        y: pd.Series | np.ndarray | None = None,
        threshold: float | None = None,
    ) -> BinaryClassificationResult:
        """Evaluate the fitted model on `df` or `(X, y)` and return a `BinaryClassificationResult`."""
        self._check_is_fitted()
        if df is not None:
            X_eval = df
            y_eval = df[self.target_col].to_numpy(dtype=int)
        elif X is not None and y is not None:
            X_eval = X
            y_eval = np.asarray(y, dtype=int)
        else:
            raise ValueError("Provide either `df` or both `X` and `y`.")

        thresh = self.threshold if threshold is None else threshold
        probs = self.predict_proba(X_eval)
        return evaluate_binary_classification(
            y_true=y_eval, y_prob=probs, threshold=thresh
        )

    def cross_validate(
        self,
        df: pd.DataFrame,
        *,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """
        Run Stratified K-Fold Cross-Validation using this model's configured specification.

        Does not mutate the fitted state of `self`.

        Returns
        -------
        dict[str, Any]
            - `"oof_preds"`: Out-of-fold predicted probabilities `(N,)`.
            - `"oof_evaluation"`: `BinaryClassificationResult` on out-of-fold predictions.
            - `"oof_metrics"`: Flat dictionary of out-of-fold metrics.
            - `"fold_metrics"`: Per-fold evaluation metrics DataFrame.
        """
        X_all, y_all_float, feat_list = self._resolve_inputs(df, None, None)
        X_all = X_all.reset_index(drop=True)
        y_all = y_all_float.astype(int)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        oof_preds = np.zeros(len(X_all), dtype=float)
        fold_rows: list[dict[str, float]] = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
            fold_model = LinearModel(
                model_type=self.model_type,
                cov_type=self.cov_type,
                maxiter=self.maxiter,
                target_col=self.target_col,
                feature_cols=feat_list,
                drop_cols=self.drop_cols,
                threshold=self.threshold,
            )
            fold_model.fit(X=X_all.iloc[train_idx], y=y_all[train_idx])
            val_probs = fold_model.predict_proba(X_all.iloc[val_idx])
            oof_preds[val_idx] = val_probs

            fold_eval = evaluate_binary_classification(
                y_true=y_all[val_idx], y_prob=val_probs, threshold=self.threshold
            )
            m_fold = fold_eval.to_dict()
            m_fold["fold"] = fold_idx
            fold_rows.append(m_fold)

        oof_eval = evaluate_binary_classification(
            y_true=y_all, y_prob=oof_preds, threshold=self.threshold
        )

        return {
            "oof_preds": oof_preds,
            "oof_evaluation": oof_eval,
            "oof_metrics": oof_eval.to_dict(),
            "fold_metrics": pd.DataFrame(fold_rows),
        }

    def summary_text(self) -> str:
        """Return the underlying statsmodels regression summary text."""
        self._check_is_fitted()
        return str(self.fitted_model.summary())

    def __repr__(self) -> str:
        if not self.is_fitted:
            return (
                f"<LinearModel(model_type='{self.model_type}', cov_type='{self.cov_type}', "
                f"fitted=False)>"
            )
        roc_auc = self.metrics.get("roc_auc", float("nan"))
        return (
            f"<LinearModel(model_type='{self.model_type}', cov_type='{self.cov_type}', "
            f"fitted=True, N={self.num_obs}, features={len(self.feature_names)}, "
            f"ROC-AUC={roc_auc:.3f})>"
        )
