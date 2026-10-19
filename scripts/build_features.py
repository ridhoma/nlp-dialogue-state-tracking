#!/usr/bin/env python
"""
Build conversation-level tabular feature matrix for post-mortem resolution modeling.

Merges:
1. Conversation baseline metadata and structural features
2. Dialogue State Tracking (DST / CST) features
3. Action State Tracking (AST) features
Saves result to data/processed/features_tabular.parquet.
"""

import argparse
import logging
import time

from data_science.features.builder import build_feature_matrix, save_feature_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Build conversation-level tabular features (DST + AST) on ABCD"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/features_tabular.parquet",
        help="Target parquet output path (default: data/processed/features_tabular.parquet)",
    )
    parser.add_argument(
        "--slots_path",
        type=str,
        default="data/processed/customer_slots_extracted_run_02.parquet",
        help="Path to extracted customer slots parquet (default: data/processed/customer_slots_extracted_run_02.parquet)",
    )
    args = parser.parse_args()

    t0 = time.time()
    logger.info("Starting feature matrix construction pipeline...")

    df_features = build_feature_matrix(slots_path=args.slots_path)
    save_feature_matrix(df_features, output_path=args.output)

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.2f}s! Generated {len(df_features)} conversation rows.")


if __name__ == "__main__":
    main()
