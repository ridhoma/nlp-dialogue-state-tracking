"""
Categorical feature encoding utilities for econometric (OLS / Logit) and
tree-based (GBDT) modeling.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


def one_hot_encode(
    df: pd.DataFrame,
    columns: str | Sequence[str],
    *,
    drop_first: bool = False,
    reference_categories: Mapping[str, Any] | None = None,
    drop_original_columns: bool = False,
    only_encoded: bool = False,
    prefix_sep: str = "_",
    dummy_na: bool = False,
    dtype: Any = int,
) -> pd.DataFrame:
    """
    One-hot encode one or more categorical columns into numeric (0/1) indicator columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the categorical column(s) to encode.
    columns : str | Sequence[str]
        Single column name or sequence of column names to one-hot encode
        (e.g., `["flow", "member_level"]`).
    drop_first : bool, default False
        If `True`, drops one reference category per encoded column to avoid the
        dummy variable trap (perfect multicollinearity with the intercept in OLS/Logit).
        If a column has an explicit baseline in `reference_categories`, that category
        is dropped instead of the alphabetically first category.
    reference_categories : Mapping[str, Any] | None, optional
        Optional mapping from column name to the specific category level to drop as
        the baseline reference (e.g., `{"member_level": "guest"}`). Specifying a column
        in `reference_categories` drops that reference level regardless of `drop_first`.
    drop_original_columns : bool, default False
        If `True`, removes the original categorical `columns` from the returned DataFrame.
        Ignored when `only_encoded=True`.
    only_encoded : bool, default False
        If `True`, returns a DataFrame containing *only* the newly created one-hot
        indicator columns (preserving `df.index`).
    prefix_sep : str, default "_"
        Delimiter between the column name and category value (e.g., `"flow_order_issue"`).
    dummy_na : bool, default False
        If `True`, adds an indicator column for `NaN` values.
    dtype : data-type, default int
        Data type of the output one-hot columns. Defaults to `int` (`0`/`1`) rather
        than `bool` for direct compatibility with `statsmodels` and `scikit-learn`.

    Returns
    -------
    pd.DataFrame
        DataFrame with one-hot encoded indicator columns (or only the encoded columns
        when `only_encoded=True`).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` must be a pandas DataFrame; got {type(df).__name__}.")

    cols = [columns] if isinstance(columns, str) else list(columns)
    if not cols:
        raise ValueError("`columns` must specify at least one column name.")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")

    ref_map = dict(reference_categories) if reference_categories is not None else {}
    invalid_refs = [c for c in ref_map if c not in cols]
    if invalid_refs:
        raise ValueError(
            f"`reference_categories` contains columns not in `columns`: {invalid_refs}"
        )

    encoded_blocks: list[pd.DataFrame] = []

    for col in cols:
        dummies = pd.get_dummies(
            df[col],
            prefix=col,
            prefix_sep=prefix_sep,
            dummy_na=dummy_na,
            drop_first=False,
            dtype=dtype,
        )

        if col in ref_map:
            ref_val = ref_map[col]
            ref_col_name = f"{col}{prefix_sep}{ref_val}"
            if ref_col_name not in dummies.columns:
                available = [
                    c[len(f"{col}{prefix_sep}") :] for c in dummies.columns
                ]
                raise ValueError(
                    f"Reference category '{ref_val}' for column '{col}' not found. "
                    f"Available categories: {available}"
                )
            dummies = dummies.drop(columns=[ref_col_name])
        elif drop_first and dummies.shape[1] > 1:
            dummies = dummies.iloc[:, 1:]

        encoded_blocks.append(dummies)

    df_dummies = (
        pd.concat(encoded_blocks, axis=1)
        if encoded_blocks
        else pd.DataFrame(index=df.index)
    )

    if only_encoded:
        return df_dummies

    base_df = df.drop(columns=cols) if drop_original_columns else df.copy()
    # Avoid duplicate column names if re-encoding
    overlap = [c for c in df_dummies.columns if c in base_df.columns]
    if overlap:
        base_df = base_df.drop(columns=overlap)

    return pd.concat([base_df, df_dummies], axis=1)
