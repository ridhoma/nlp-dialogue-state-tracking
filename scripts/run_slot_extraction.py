#!/usr/bin/env python
"""
Run Dialogue State Tracking (Slot Extraction) across all turns using GPU batched inference.
Generates turn-level inferred slot states for downstream feature engineering.
Uses the benchmark-verified Run 05 architecture:
- Stage 1: Generalized Informative Utterance Gate (tau=0.02)
- Stage 2: Option 1 Calibrated Contextual Extraction (dynamic thresholds, default=0.90)
- Stage 3: Customer-Support Verification Disambiguation Gate (tau=0.05)
"""

import argparse
import logging
from itertools import batched

import pandas as pd
from tqdm import tqdm

from abcd_helper import loader
from nlp.context import extract_customer_contexts
from nlp.slots import (
    DEFAULT_GATE_HYPOTHESIS,
    DEFAULT_SLOT_HYPOTHESES,
    DEFAULT_SLOT_THRESHOLDS,
    SemanticMatcher,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run Dialogue State Tracking (Slot Extraction) on ABCD dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/bart-large-mnli",
        help="HuggingFace model for semantic NLI slot extraction (default: facebook/bart-large-mnli)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/customer_slots_extracted_run_01.parquet",
        help="Path to save the output dataframe (default: data/processed/customer_slots_extracted_run_01.parquet)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of conversations to process (for quick smoke testing)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for GPU forward passes (default: 128)",
    )
    parser.add_argument(
        "--default_threshold",
        type=float,
        default=0.90,
        help="Default probability threshold for slots not in dynamic mapping (default: 0.90)",
    )
    parser.add_argument(
        "--disable_gate",
        action="store_true",
        help="Disable Stage 1 Informative Utterance Gate",
    )
    parser.add_argument(
        "--gate_threshold",
        type=float,
        default=0.02,
        help="Entailment threshold for Stage 1 informative gate (default: 0.02)",
    )
    parser.add_argument(
        "--disable_disambiguation",
        action="store_true",
        help="Disable Stage 3 Contextual Disambiguation Gate",
    )
    parser.add_argument(
        "--verification_threshold",
        type=float,
        default=0.05,
        help="Entailment threshold for Stage 3 customer utterance verification (default: 0.05)",
    )
    parser.add_argument(
        "--all_ontology_slots",
        action="store_true",
        help="Evaluate all 25 ontology action slots instead of the 10 core customer entity slots",
    )
    args = parser.parse_args()

    logger.info("Loading conversations and intent schema...")
    df_turns = loader.load_turns(use_cache=True)

    if args.sample:
        convo_ids = df_turns["convo_id"].unique()[: args.sample]
        df_turns = df_turns[df_turns["convo_id"].isin(convo_ids)]
        logger.info(f"Sampled {args.sample} conversations ({len(df_turns)} turns).")

    logger.info("Extracting contextualized customer documents...")
    all_documents = extract_customer_contexts(
        df_turns, min_prior_agent_utterances=1, group_by_convo=False
    )
    logger.info(f"Generated {len(all_documents)} customer documents.")

    if args.all_ontology_slots:
        logger.info("Gathering all 25 ontology slots from schema...")
        df_schema = loader.load_intent_slots(as_df=True)
        global_slots = set()
        df_schema["all_slots"].apply(lambda x: global_slots.update(x))
        available_slots = sorted(global_slots)
    else:
        logger.info(f"Using {len(DEFAULT_SLOT_HYPOTHESES)} candidate customer entity slots...")
        available_slots = sorted(DEFAULT_SLOT_HYPOTHESES.keys())

    logger.info(f"Loaded {len(available_slots)} available slots: {available_slots}")

    enable_gate = not args.disable_gate
    enable_disambiguation = not args.disable_disambiguation

    logger.info(
        f"Initializing SemanticMatcher with model {args.model}, "
        f"gate={enable_gate} (tau={args.gate_threshold}), "
        f"disambiguation={enable_disambiguation} (tau={args.verification_threshold})"
    )
    matcher = SemanticMatcher(
        model_name=args.model,
        hypothesis_template=DEFAULT_SLOT_HYPOTHESES,
        threshold=DEFAULT_SLOT_THRESHOLDS,
        default_threshold=args.default_threshold,
        enable_informative_gate=enable_gate,
        gate_hypothesis=DEFAULT_GATE_HYPOTHESIS,
        gate_threshold=args.gate_threshold,
        enable_contextual_disambiguation=enable_disambiguation,
        verification_threshold=args.verification_threshold,
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
        batch_inferred = matcher.extract_batch(
            batch_texts,
            available_slots,
            batch_size=args.batch_size,
            return_scores=True,
        )

        for doc, extracted in zip(batch_docs, batch_inferred):
            records.append(
                {
                    "convo_id": doc.convo_id,
                    "turn_id": doc.turn_id,
                    "inferred_slots": list(extracted.keys()),
                    "confidence_scores": list(extracted.values()),
                }
            )

    df_results = pd.DataFrame(records)
    logger.info(f"Saving {len(df_results)} turn states to {args.output}")
    df_results.to_parquet(args.output, index=False)
    logger.info("Slot extraction pipeline completed successfully!")


if __name__ == "__main__":
    main()
