"""
Feature Matrix Builder & Orchestrator.

Combines conversation metadata, target labels, DST features, and AST features into a single,
clean tabular dataset ready for gradient-boosted modeling (XGBoost/LightGBM) and SHAP discovery.
"""

import logging
from pathlib import Path

import pandas as pd

from abcd_helper import loader
from data_science.features.ast import extract_ast_features_batch
from data_science.features.dst import extract_dst_features_batch

logger = logging.getLogger(__name__)


def build_feature_matrix(
    df_convos: pd.DataFrame | None = None,
    df_turns: pd.DataFrame | None = None,
    df_slots: pd.DataFrame | None = None,
    df_schemas: pd.DataFrame | None = None,
    slots_path: str | Path = "data/processed/customer_slots_extracted_run_01.parquet",
) -> pd.DataFrame:
    """
    Build the complete tabular feature matrix for post-mortem conversation analysis.

    Merges:
      1. Structural conversation features (turns, speaker turn ratios)
      2. Dialogue State Tracking (DST / CST) features
      3. Action State Tracking (AST) features
      4. Metadata & target label (split, flow, intent, resolved)

    Args:
        df_convos: Conversations DataFrame. If None, loaded via abcd_helper.loader.
        df_turns: Turns DataFrame. If None, loaded via abcd_helper.loader.
        df_slots: Extracted customer slots DataFrame. If None, loaded from slots_path.
        df_schemas: Normative intent schema DataFrame. If None, loaded via abcd_helper.loader.
        slots_path: Path to extracted customer slots parquet file.

    Returns:
        Complete tabular DataFrame with 1 row per conversation.
    """
    if df_convos is None:
        logger.info("Loading conversations table...")
        df_convos = loader.load_conversations()

    if df_turns is None:
        logger.info("Loading turns table...")
        df_turns = loader.load_turns()

    if df_slots is None:
        logger.info(f"Loading extracted slots from {slots_path}...")
        df_slots = pd.read_parquet(slots_path)

    if df_schemas is None:
        logger.info("Loading intent schema table...")
        df_schemas = loader.load_intent_slots(as_df=True)

    logger.info(f"Extracting DST features for {len(df_convos)} conversations...")
    df_dst = extract_dst_features_batch(df_convos, df_slots, df_schemas)

    logger.info(f"Extracting AST features for {len(df_convos)} conversations...")
    df_ast = extract_ast_features_batch(df_convos, df_turns, df_slots)

    logger.info("Assembling base conversation structural features...")
    # Base structural & volume features
    base_cols = [
        "convo_id",
        "split",
        "flow",
        "intent",
        "num_turns",
        "num_customer_turns",
        "num_agent_turns",
        "resolved",
    ]
    df_base = df_convos[[c for c in base_cols if c in df_convos.columns]].copy()

    # Feature: customer to total turns ratio
    if "num_turns" in df_base.columns and "num_customer_turns" in df_base.columns:
        df_base["customer_turn_ratio"] = (
            df_base["num_customer_turns"] / df_base["num_turns"].clip(lower=1)
        ).astype(float)

    # Merge on convo_id
    logger.info("Joining feature tables...")
    merged = df_base.merge(df_dst, on="convo_id", how="left")
    merged = merged.merge(df_ast, on="convo_id", how="left")

    logger.info(f"Successfully assembled feature matrix: {merged.shape[0]} rows, {merged.shape[1]} columns.")
    return merged


def save_feature_matrix(
    df_matrix: pd.DataFrame,
    output_path: str | Path = "data/processed/features_tabular.parquet",
) -> None:
    """
    Save the tabular feature matrix to a Parquet file.

    Args:
        df_matrix: Assembled feature DataFrame.
        output_path: Target parquet file path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_matrix.to_parquet(path, index=False)
    logger.info(f"Saved tabular features ({len(df_matrix)} rows) to {path}")
