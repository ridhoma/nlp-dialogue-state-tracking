#!/usr/bin/env python
"""
Build versioned conversation-level tabular feature matrices for post-mortem
resolution modeling (OLS / Logistic Regression and GBDT / XGBoost).

Follows the feature selection and encoding pipeline curated in
`notebooks/02_feature_engineering.ipynb` and writes versioned Parquet files
plus JSON provenance manifests to `data/processed/`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Literal

import pandas as pd

import abcd_helper as abcd
from abcd_helper.config import PROCESSED_DATA_DIR
from data_science.features.ast import extract_ast_features_batch
from data_science.features.dst import extract_dst_features_batch
from data_science.features.encodings import one_hot_encode

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Curated feature subsets from notebooks/02_feature_engineering.ipynb
DST_SELECTED_FEATURES: list[str] = [
    "mandatory_slot_coverage",
    "all_mandatory_slots_filled",
    "unused_slot_ratio",
    "slot_filling_latency",
    "upfront_slot_density",
    "slot_repetition_count",
    "total_slots_disclosed",
    "mandatory_slots_count",
]

AST_SELECTED_FEATURES: list[str] = [
    "num_canonical_actions",
    "action_count_delta",
    "sop_edit_distance_norm",
    "action_thrashing_count",
    "premature_action_attempts",
    "action_execution_latency_mean",
]

OLS_REFERENCE_CATEGORIES: dict[str, str] = {
    "flow": "account_access",
    "member_level": "guest",
}


def build_context_features(
    df_convos: pd.DataFrame,
    encoding: Literal["ols", "full"] = "ols",
) -> pd.DataFrame:
    """
    Construct conversation context and structural features, including
    `share_customer_turns`, `num_order_products`, and one-hot encoded
    `flow` and `member_level`.

    Parameters
    ----------
    df_convos : pd.DataFrame
        Raw conversations table loaded via `abcd.load_conversations()`.
    encoding : {"ols", "full"}, default "ols"
        - `"ols"`: Drops reference categories (`flow="account_access"`,
          `member_level="guest"`) to avoid the dummy variable trap in linear models.
        - `"full"`: Retains all K indicator columns for tree-based models (GBDT).
    """
    df_context = df_convos[
        ["convo_id", "resolved", "flow", "member_level", "num_turns"]
    ].copy()

    df_context["share_customer_turns"] = (
        df_convos["num_customer_turns"] / df_convos["num_turns"].clip(lower=1)
    ).astype(float)

    df_context["num_order_products"] = (
        df_convos["order_products"]
        .str.split(",")
        .str.len()
        .fillna(0)
        .astype(int)
    )

    is_ols = encoding == "ols"
    df_context = one_hot_encode(
        df_context,
        columns=["flow", "member_level"],
        drop_original_columns=True,
        drop_first=is_ols,
        reference_categories=OLS_REFERENCE_CATEGORIES if is_ols else None,
    )
    return df_context


def assemble_feature_dataset(
    df_convos: pd.DataFrame,
    df_dst: pd.DataFrame,
    df_ast: pd.DataFrame,
    encoding: Literal["ols", "full"] = "ols",
    dst_features: list[str] = DST_SELECTED_FEATURES,
    ast_features: list[str] = AST_SELECTED_FEATURES,
) -> pd.DataFrame:
    """
    Merge context features with selected DST and AST feature subsets on `convo_id`.
    """
    df_context = build_context_features(df_convos, encoding=encoding)
    df_dst_sub = df_dst[["convo_id", *dst_features]]
    df_ast_sub = df_ast[["convo_id", *ast_features]]

    return (
        df_context.merge(df_dst_sub, how="inner", on="convo_id")
        .merge(df_ast_sub, how="inner", on="convo_id")
    )


def save_versioned_dataset(
    df: pd.DataFrame,
    output_path: Path,
    *,
    version: str,
    encoding: str,
    slots_path: Path,
) -> Path:
    """
    Save the feature DataFrame to Parquet alongside a `.meta.json` provenance file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    meta_path = output_path.with_suffix(".meta.json")
    metadata = {
        "version": version,
        "encoding": encoding,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_slots_file": str(slots_path),
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "null_count": int(df.isna().sum().sum()),
        "reference_categories_dropped": (
            OLS_REFERENCE_CATEGORIES if encoding == "ols" else {}
        ),
        "dst_selected_features": DST_SELECTED_FEATURES,
        "ast_selected_features": AST_SELECTED_FEATURES,
        "columns": list(df.columns),
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    logger.info(
        "Saved %s (%d rows, %d cols) and manifest %s",
        output_path,
        df.shape[0],
        df.shape[1],
        meta_path.name,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build versioned tabular feature datasets (OLS & GBDT) on ABCD."
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v01",
        help="Dataset version tag (e.g. 'v01', 'v02'). Default: v01",
    )
    parser.add_argument(
        "--encoding",
        choices=["ols", "full", "both"],
        default="ols",
        help=(
            "One-hot encoding mode: 'ols' (drops reference categories), "
            "'full' (keeps all categories for GBDT), or 'both'. Default: ols"
        ),
    )
    parser.add_argument(
        "--slots-path",
        type=Path,
        default=PROCESSED_DATA_DIR / "customer_slots_extracted_run_02.parquet",
        help="Path to extracted customer slots Parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Target directory for versioned Parquet outputs (default: data/processed).",
    )
    args = parser.parse_args()

    t0 = time.time()
    logger.info("Starting feature engineering pipeline (version=%s, encoding=%s)...", args.version, args.encoding)

    df_convos = abcd.load_conversations()
    logger.info("Loaded conversations table: %d rows", len(df_convos))

    df_turns = abcd.load_turns()
    logger.info("Loaded turns table: %d rows", len(df_turns))

    logger.info("Loading NLP-extracted slots from %s...", args.slots_path)
    df_slots = pd.read_parquet(args.slots_path)
    logger.info("Loaded NLP-extracted slots table: %d rows", len(df_slots))

    df_schemas = abcd.load_intent_slots(as_df=True)
    logger.info("Loaded intent slots schema: %d rows", len(df_schemas))

    logger.info("Extracting DST features for %d conversations...", len(df_convos))
    df_dst = extract_dst_features_batch(
        df_convos=df_convos,
        df_slots=df_slots,
        df_schemas=df_schemas,
    )

    logger.info("Extracting AST features for %d conversations...", len(df_convos))
    df_ast = extract_ast_features_batch(
        df_convos=df_convos,
        df_turns=df_turns,
        df_slots=df_slots,
    )

    encodings_to_build: list[Literal["ols", "full"]] = (
        ["ols", "full"] if args.encoding == "both" else [args.encoding]
    )

    for enc in encodings_to_build:
        df_out = assemble_feature_dataset(
            df_convos=df_convos,
            df_dst=df_dst,
            df_ast=df_ast,
            encoding=enc,
        )
        out_file = args.output_dir / f"features_{enc}_{args.version}.parquet"
        save_versioned_dataset(
            df_out,
            out_file,
            version=args.version,
            encoding=enc,
            slots_path=args.slots_path,
        )

    elapsed = time.time() - t0
    logger.info("Pipeline completed in %.2fs!", elapsed)


if __name__ == "__main__":
    main()
