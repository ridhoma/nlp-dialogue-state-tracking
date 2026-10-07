#!/usr/bin/env python
"""
Run Dialogue State Tracking (Slot Extraction) across all turns using GPU batched inference.
Generates turn-level inferred slot states for downstream feature engineering.
"""

import argparse
import logging
from itertools import batched

import pandas as pd
from tqdm import tqdm

from abcd_helper import loader
from nlp.context import extract_customer_contexts
from nlp.slots import SemanticMatcher

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run slot extraction on ABCD dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="valhalla/distilbart-mnli-12-3",
        help="HuggingFace model for semantic NLI fallback",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/turn_slot_states.parquet",
        help="Path to save the output dataframe",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of conversations to process (for testing)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for GPU inference",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for calibrated NLI entailment",
    )
    args = parser.parse_args()

    logger.info("Loading conversations and intent schema...")
    df_turns = loader.load_turns(use_cache=True)

    if args.sample:
        convo_ids = df_turns["convo_id"].unique()[: args.sample]
        df_turns = df_turns[df_turns["convo_id"].isin(convo_ids)]

    logger.info("Extracting contextualized customer documents...")
    # Flat list of customer documents is ideal for batching across conversations
    all_documents = extract_customer_contexts(df_turns, group_by_convo=False)
    logger.info(f"Generated {len(all_documents)} customer documents.")

    logger.info("Gathering global universe of slots for unused_slot_ratio tracking...")
    df_schema = loader.load_intent_slots(as_df=True)
    global_slots = set()
    df_schema["all_slots"].apply(lambda x: global_slots.update(x))
    available_slots = sorted(global_slots)
    logger.info(f"Loaded {len(available_slots)} global slots.")

    logger.info(
        f"Initializing SemanticMatcher with model {args.model} and threshold {args.threshold}"
    )
    matcher = SemanticMatcher(
        model_name=args.model,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )

    records = []

    logger.info("Starting batched slot extraction...")

    for batch_docs in tqdm(
        list(batched(all_documents, args.batch_size)), desc="Batches"
    ):
        batch_texts = [
            doc.get_full_document(strip_prior_customer_utterances=True)
            for doc in batch_docs
        ]
        batch_inferred = matcher.extract_batch(batch_texts, available_slots)

        for doc, inferred_slots in zip(batch_docs, batch_inferred):
            records.append(
                {
                    "convo_id": doc.convo_id,
                    "turn_id": doc.turn_id,
                    "inferred_slots": inferred_slots,
                }
            )

    df_results = pd.DataFrame(records)
    logger.info(f"Saving {len(df_results)} turn states to {args.output}")
    df_results.to_parquet(args.output, index=False)
    logger.info("Done!")


if __name__ == "__main__":
    main()
