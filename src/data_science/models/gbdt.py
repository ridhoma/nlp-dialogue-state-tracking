"""
Gradient Boosted Decision Tree (XGBoost) estimator class with Stratified K-Fold
cross-validation, early stopping, and native C++ TreeSHAP attributions and
pairwise feature interaction discovery.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from data_science.evaluations import (
    BinaryClassificationResult,
    evaluate_binary_classification,
)


def _ensure_macos_openmp() -> None:
    """
    Preload `libomp.dylib` on macOS if bundled inside `.venv` (e.g. via PyTorch)
    or Homebrew so `xgboost` loads cleanly without requiring global `DYLD_LIBRARY_PATH`.
    """
    if sys.platform != "darwin":
        return
    candidates = [
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "torch"
        / "lib"
        / "libomp.dylib",
        Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
        Path("/usr/local/opt/libomp/lib/libomp.dylib"),
    ]
    for path in candidates:
        if path.exists():
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                break
            except OSError:
                continue


_ensure_macos_openmp()
import xgboost as xgb  # noqa: E402


DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "tree_method": "hist",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 400,
    "min_child_weight": 10.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}


def compute_native_treeshap(
    booster: xgb.Booster,
    X: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute exact TreeSHAP feature attributions (in log-odds margin units) using
    XGBoost's native C++ engine (`pred_contribs=True`).

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        `(shap_df, base_values)` where `shap_df` has shape `(N, P)` matching `X`
        and `base_values` has shape `(N,)` representing the expected log-odds intercept.
    """
    dmat = xgb.DMatrix(X, feature_names=list(X.columns))
    contribs = booster.predict(dmat, pred_contribs=True)
    shap_df = pd.DataFrame(contribs[:, :-1], columns=X.columns, index=X.index)
    base_values = np.asarray(contribs[:, -1], dtype=float)
    return shap_df, base_values


def compute_native_shap_interactions(
    booster: xgb.Booster,
    X: pd.DataFrame,
    *,
    max_samples: int = 2000,
    top_k: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute exact 2-way TreeSHAP interaction values using XGBoost's native C++
    engine (`pred_interactions=True`) and rank the strongest off-diagonal feature
    pairs `(feature_a, feature_b)` by mean absolute interaction strength.
    """
    if len(X) > max_samples:
        X_sub = X.sample(n=max_samples, random_state=random_state)
    else:
        X_sub = X

    feat_names = list(X_sub.columns)
    dmat = xgb.DMatrix(X_sub, feature_names=feat_names)
    # Shape: (N, P + 1, P + 1), where the last row/col is the bias term
    interactions = booster.predict(dmat, pred_interactions=True)[:, :-1, :-1]
    mean_abs_inter = np.mean(np.abs(interactions), axis=0)

    pairs: list[dict[str, Any]] = []
    num_features = len(feat_names)
    for i in range(num_features):
        for j in range(i + 1, num_features):
            # Off-diagonal interaction between feature i and j is split symmetrically: 2 * |Phi_{i,j}|
            pair_strength = float(2.0 * mean_abs_inter[i, j])
            pairs.append(
                {
                    "feature_a": feat_names[i],
                    "feature_b": feat_names[j],
                    "mean_abs_interaction": pair_strength,
                }
            )

    df_pairs = (
        pd.DataFrame(pairs)
        .sort_values("mean_abs_interaction", ascending=False)
        .reset_index(drop=True)
    )
    return df_pairs.head(top_k)


class XGBoostModel:
    """
    Gradient Boosted Decision Tree (XGBoost) binary classifier with native C++
    TreeSHAP attributions and interaction discovery, supporting the workflow:
        specify model -> `.cross_validate(df)` -> `.fit(df)`

    Parameters
    ----------
    params : Mapping[str, Any] | None, optional
        Dictionary of XGBoost hyperparameter overrides merged on top of
        `DEFAULT_XGB_PARAMS`.
    early_stopping_rounds : int | None, default=30
        Early stopping patience when validation data is supplied to `.fit()` or
        during per-fold `.cross_validate()`.
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
    **xgb_kwargs : Any
        Additional XGBoost hyperparameters passed directly as keyword arguments
        (e.g. `max_depth=4`, `learning_rate=0.05`, `n_estimators=400`).
    """

    def __init__(
        self,
        *,
        params: Mapping[str, Any] | None = None,
        early_stopping_rounds: int | None = 30,
        target_col: str = "resolved",
        feature_cols: Sequence[str] | None = None,
        drop_cols: Sequence[str] = ("convo_id", "split", "flow", "intent"),
        threshold: float = 0.5,
        **xgb_kwargs: Any,
    ) -> None:
        merged = {**DEFAULT_XGB_PARAMS}
        if params is not None:
            merged.update(dict(params))
        if xgb_kwargs:
            merged.update(xgb_kwargs)

        self.params: dict[str, Any] = merged
        self.early_stopping_rounds = early_stopping_rounds
        self.target_col = target_col
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.drop_cols = tuple(drop_cols)
        self.threshold = threshold

        # Learned attributes populated by `.fit()`
        self.fitted_model: xgb.XGBClassifier | None = None
        self.importance_table: pd.DataFrame = pd.DataFrame()
        self.shap_values: pd.DataFrame = pd.DataFrame()
        self.shap_base_value: float = float("nan")
        self.evaluation: BinaryClassificationResult | None = None
        self.val_evaluation: BinaryClassificationResult | None = None
        self.metrics: dict[str, float] = {}
        self.feature_names: list[str] = []
        self.best_iteration: int = 0
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
        """Extract numeric feature matrix `X_df`, binary target `y_arr`, and feature list."""
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
            y_arr = df[self.target_col].to_numpy(dtype=int)
        elif X is not None and y is not None:
            feat_list = (
                list(self.feature_cols)
                if self.feature_cols is not None
                else list(X.columns)
            )
            X_df = X[feat_list].astype(float).copy()
            y_arr = np.asarray(y, dtype=int)
        else:
            raise ValueError("Provide either `df` or both `X` and `y`.")

        return X_df, y_arr, feat_list

    def fit(
        self,
        df: pd.DataFrame | None = None,
        *,
        X: pd.DataFrame | None = None,
        y: pd.Series | np.ndarray | None = None,
        eval_df: pd.DataFrame | None = None,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> XGBoostModel:
        """
        Fit the configured XGBoost classifier on `df` or `(X, y)` with optional
        validation early stopping (`eval_df` or `(X_val, y_val)`).

        Populates `.importance_table`, `.shap_values`, `.evaluation`, `.metrics`,
        and `.fitted_model` on `self`, and returns `self`.
        """
        X_train, y_train, feat_list = self._resolve_inputs(df, X, y)

        run_params = dict(self.params)
        has_val = eval_df is not None or (X_val is not None and y_val is not None)

        if has_val and self.early_stopping_rounds is not None:
            run_params["early_stopping_rounds"] = self.early_stopping_rounds
        else:
            run_params.pop("early_stopping_rounds", None)

        clf = xgb.XGBClassifier(**run_params)

        fit_kwargs: dict[str, Any] = {"verbose": False}
        if eval_df is not None:
            X_v = eval_df[feat_list].astype(float)
            y_v = eval_df[self.target_col].to_numpy(dtype=int)
            fit_kwargs["eval_set"] = [(X_train, y_train), (X_v, y_v)]
        elif X_val is not None and y_val is not None:
            X_v = X_val[feat_list].astype(float)
            y_v = np.asarray(y_val, dtype=int)
            fit_kwargs["eval_set"] = [(X_train, y_train), (X_v, y_v)]

        clf.fit(X_train, y_train, **fit_kwargs)

        booster = clf.get_booster()
        best_iter = int(getattr(clf, "best_iteration", clf.n_estimators - 1))

        gain_scores = booster.get_score(importance_type="gain")
        weight_scores = booster.get_score(importance_type="weight")
        cover_scores = booster.get_score(importance_type="cover")

        shap_df, base_vals = compute_native_treeshap(booster, X_train)
        mean_abs_shap = shap_df.abs().mean(axis=0)
        mean_shap = shap_df.mean(axis=0)

        imp_rows: list[dict[str, Any]] = []
        total_gain = sum(gain_scores.values()) or 1.0
        total_abs_shap = float(mean_abs_shap.sum()) or 1.0

        for col in feat_list:
            g_val = float(gain_scores.get(col, 0.0))
            w_val = float(weight_scores.get(col, 0.0))
            c_val = float(cover_scores.get(col, 0.0))
            mas_val = float(mean_abs_shap[col])

            x_std = float(X_train[col].std())
            s_std = float(shap_df[col].std())
            if x_std > 1e-12 and s_std > 1e-12:
                corr_dir = float(
                    np.corrcoef(X_train[col].to_numpy(), shap_df[col].to_numpy())[0, 1]
                )
            else:
                corr_dir = 0.0

            imp_rows.append(
                {
                    "feature": col,
                    "mean_abs_shap": mas_val,
                    "shap_share_pct": round(100.0 * mas_val / total_abs_shap, 2),
                    "direction_corr": round(corr_dir, 4),
                    "gain": g_val,
                    "gain_share_pct": round(100.0 * g_val / total_gain, 2),
                    "split_count": int(w_val),
                    "cover": c_val,
                    "mean_shap": float(mean_shap[col]),
                }
            )

        imp_table = (
            pd.DataFrame(imp_rows)
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        train_probs = np.asarray(clf.predict_proba(X_train)[:, 1], dtype=float)
        train_eval = evaluate_binary_classification(
            y_true=y_train, y_prob=train_probs, threshold=self.threshold
        )
        metrics = train_eval.to_dict()

        val_eval: BinaryClassificationResult | None = None
        if has_val:
            val_probs = np.asarray(clf.predict_proba(X_v)[:, 1], dtype=float)
            val_eval = evaluate_binary_classification(
                y_true=y_v, y_prob=val_probs, threshold=self.threshold
            )
            for k, v in val_eval.to_dict().items():
                metrics[f"val_{k}"] = v

        self.fitted_model = clf
        self.importance_table = imp_table
        self.shap_values = shap_df
        self.shap_base_value = float(np.mean(base_vals))
        self.evaluation = train_eval
        self.val_evaluation = val_eval
        self.metrics = metrics
        self.feature_names = feat_list
        self.best_iteration = best_iter
        self.num_obs = int(len(y_train))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict resolution probabilities `P(Y = 1 | X)` on `X`."""
        self._check_is_fitted()
        assert self.fitted_model is not None
        X_sub = X[self.feature_names].astype(float)
        return np.asarray(self.fitted_model.predict_proba(X_sub)[:, 1], dtype=float)

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

    def compute_shap(self, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Compute native TreeSHAP attributions for any DataFrame `X`."""
        self._check_is_fitted()
        assert self.fitted_model is not None
        X_sub = X[self.feature_names].astype(float)
        return compute_native_treeshap(self.fitted_model.get_booster(), X_sub)

    def compute_shap_interactions(
        self,
        X: pd.DataFrame,
        *,
        max_samples: int = 2000,
        top_k: int = 20,
    ) -> pd.DataFrame:
        """Compute top-k pairwise TreeSHAP interactions on `X`."""
        self._check_is_fitted()
        assert self.fitted_model is not None
        X_sub = X[self.feature_names].astype(float)
        return compute_native_shap_interactions(
            self.fitted_model.get_booster(),
            X_sub,
            max_samples=max_samples,
            top_k=top_k,
        )

    def cross_validate(
        self,
        df: pd.DataFrame,
        *,
        n_splits: int = 5,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """
        Run Stratified K-Fold Cross-Validation using this model's configured
        hyperparameters and early stopping, without mutating `self`.

        Returns
        -------
        dict[str, Any]
            - `"oof_preds"`: Out-of-fold predicted probabilities `(N,)`.
            - `"oof_evaluation"`: `BinaryClassificationResult` on out-of-fold predictions.
            - `"oof_metrics"`: Flat dictionary of out-of-fold metrics.
            - `"fold_metrics"`: Per-fold evaluation metrics DataFrame.
            - `"oof_shap_values"`: Out-of-fold TreeSHAP attributions DataFrame `(N, P)`.
            - `"cv_importance_table"`: Out-of-fold aggregated feature importance DataFrame.
        """
        X_all, y_all, feat_list = self._resolve_inputs(df, None, None)
        X_all = X_all.reset_index(drop=True)

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        oof_preds = np.zeros(len(X_all), dtype=float)
        oof_shap = np.zeros((len(X_all), len(feat_list)), dtype=float)
        fold_rows: list[dict[str, Any]] = []
        gain_accum: dict[str, float] = {c: 0.0 for c in feat_list}
        split_accum: dict[str, float] = {c: 0.0 for c in feat_list}

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
            X_tr, y_tr = X_all.iloc[train_idx], y_all[train_idx]
            X_va, y_va = X_all.iloc[val_idx], y_all[val_idx]

            fold_model = XGBoostModel(
                params=self.params,
                early_stopping_rounds=self.early_stopping_rounds,
                target_col=self.target_col,
                feature_cols=feat_list,
                drop_cols=self.drop_cols,
                threshold=self.threshold,
            )
            fold_model.fit(X=X_tr, y=y_tr, X_val=X_va, y_val=y_va)

            val_probs = fold_model.predict_proba(X_va)
            oof_preds[val_idx] = val_probs

            val_shap_df, _ = fold_model.compute_shap(X_va)
            oof_shap[val_idx, :] = val_shap_df.to_numpy(dtype=float)

            for _, row in fold_model.importance_table.iterrows():
                col_name = str(row["feature"])
                gain_accum[col_name] += float(row["gain"]) / n_splits
                split_accum[col_name] += float(row["split_count"]) / n_splits

            fold_eval = evaluate_binary_classification(
                y_true=y_va, y_prob=val_probs, threshold=self.threshold
            )
            m_fold = fold_eval.to_dict()
            m_fold["fold"] = fold_idx
            m_fold["best_iteration"] = fold_model.best_iteration
            fold_rows.append(m_fold)

        oof_shap_df = pd.DataFrame(oof_shap, columns=feat_list, index=df.index)
        mean_abs_oof_shap = oof_shap_df.abs().mean(axis=0)
        total_abs_shap = float(mean_abs_oof_shap.sum()) or 1.0
        total_gain = sum(gain_accum.values()) or 1.0

        cv_imp_rows: list[dict[str, Any]] = []
        for col in feat_list:
            mas_val = float(mean_abs_oof_shap[col])
            g_val = float(gain_accum[col])
            x_std = float(X_all[col].std())
            s_std = float(oof_shap_df[col].std())
            if x_std > 1e-12 and s_std > 1e-12:
                corr_dir = float(
                    np.corrcoef(X_all[col].to_numpy(), oof_shap_df[col].to_numpy())[0, 1]
                )
            else:
                corr_dir = 0.0

            cv_imp_rows.append(
                {
                    "feature": col,
                    "oof_mean_abs_shap": mas_val,
                    "shap_share_pct": round(100.0 * mas_val / total_abs_shap, 2),
                    "direction_corr": round(corr_dir, 4),
                    "mean_gain": g_val,
                    "gain_share_pct": round(100.0 * g_val / total_gain, 2),
                    "mean_split_count": round(split_accum[col], 1),
                }
            )

        cv_imp_table = (
            pd.DataFrame(cv_imp_rows)
            .sort_values("oof_mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

        oof_eval = evaluate_binary_classification(
            y_true=y_all, y_prob=oof_preds, threshold=self.threshold
        )

        return {
            "oof_preds": oof_preds,
            "oof_evaluation": oof_eval,
            "oof_metrics": oof_eval.to_dict(),
            "fold_metrics": pd.DataFrame(fold_rows),
            "oof_shap_values": oof_shap_df,
            "cv_importance_table": cv_imp_table,
        }

    def __repr__(self) -> str:
        max_d = self.params.get("max_depth")
        lr = self.params.get("learning_rate")
        if not self.is_fitted:
            return f"<XGBoostModel(max_depth={max_d}, learning_rate={lr}, fitted=False)>"
        roc_auc = self.metrics.get(
            "val_roc_auc", self.metrics.get("roc_auc", float("nan"))
        )
        return (
            f"<XGBoostModel(max_depth={max_d}, learning_rate={lr}, fitted=True, "
            f"N={self.num_obs}, features={len(self.feature_names)}, "
            f"best_iter={self.best_iteration}, ROC-AUC={roc_auc:.3f})>"
        )
