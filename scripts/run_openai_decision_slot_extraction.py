#!/usr/bin/env python
"""
Run Dialogue State Tracking (Slot Extraction) via OpenAI Decisions API (`client.decisions.create`).

Uses `gpt-6-luna` to evaluate multiple slot predicates simultaneously in a single
decision request per customer turn with zero output token fees ($0.10/1M input).

Supports `--dry-run` to validate context formatting, inspect question predicates,
and calculate exact token/cost estimates without needing an active OpenAI API key.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from abcd_helper import loader
from nlp.context import extract_customer_contexts
from nlp.prompts import DEFAULT_SLOT_PREDICATES
from nlp.slots import DEFAULT_SLOT_THRESHOLDS, OpenAIDecisionSlotExtractor
from utils.openai_helper import (
    DEFAULT_OPENAI_DECISION_MODEL,
    build_decision_predicate_questions,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_EVAL_SET_PATH = "data/samples/slot_extraction_evaluation_set.parquet"
DEFAULT_OUTPUT_PARQUET = (
    "data/processed/customer_slots_extracted_openai_decision.parquet"
)


def load_target_turns(
    dataset: str,
    data_path: str | None = None,
    sample: int | None = None,
) -> tuple[pd.DataFrame, list[int | str]]:
    """
    Load dialogue turns and resolve target conversation IDs based on dataset selection.
    """
    df_all_turns = loader.load_turns(use_cache=True)

    if data_path:
        logger.info("Loading custom dataset from '%s'...", data_path)
        custom_df = pd.read_parquet(data_path)
        target_cids = list(custom_df["convo_id"].unique())
    elif dataset == "benchmark":
        bench_p = Path(BENCHMARK_EVAL_SET_PATH)
        if not bench_p.exists():
            raise FileNotFoundError(
                f"Benchmark evaluation set not found at '{bench_p}'"
            )
        logger.info(
            "Loading benchmark evaluation set (1,000 convos) from '%s'...", bench_p
        )
        df_bench = pd.read_parquet(bench_p)
        target_cids = list(df_bench["convo_id"].unique())
    elif dataset == "full":
        logger.info("Using full ABCD corpus (10,042 conversations)...")
        target_cids = list(df_all_turns["convo_id"].unique())
    else:
        raise ValueError(f"Invalid dataset '{dataset}'. Must be 'benchmark' or 'full'.")

    if sample and sample > 0:
        target_cids = target_cids[:sample]
        logger.info("Sampled %d conversations.", len(target_cids))

    df_filtered_turns = df_all_turns[df_all_turns["convo_id"].isin(target_cids)]
    logger.info(
        "Resolved %d conversations (%d turns, %d customer turns).",
        len(target_cids),
        len(df_filtered_turns),
        len(df_filtered_turns[df_filtered_turns["speaker"] == "customer"]),
    )
    return df_filtered_turns, target_cids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Slot Extraction via OpenAI Decisions API (gpt-6-luna / predicate batching)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["benchmark", "full"],
        default="benchmark",
        help="Target dataset: 'benchmark' (1,000 eval convos) or 'full' (all 10,042 convos). Default: benchmark.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Optional explicit path to parquet dataset (overrides --dataset).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional integer to sample the first N conversations (for smoke testing).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_OPENAI_DECISION_MODEL,
        help=f"OpenAI Decisions API model identifier (default: {DEFAULT_OPENAI_DECISION_MODEL}).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Default probability threshold cutoff for predicate answers (default: 0.50).",
    )
    parser.add_argument(
        "--dynamic-thresholds",
        action="store_true",
        help="Use slot-specific calibrated thresholds from DEFAULT_SLOT_THRESHOLDS.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Thread pool concurrency for parallel turn evaluations (default: 8).",
    )
    parser.add_argument(
        "--min-prior-agent-utterances",
        type=int,
        default=1,
        help="Minimum preceding agent utterances to include in context window (default: 1).",
    )
    parser.add_argument(
        "--all-ontology-slots",
        action="store_true",
        help="Evaluate all 25 ontology action slots instead of the default 11 customer entity slots.",
    )
    parser.add_argument(
        "--return-scores",
        action="store_true",
        help="Include raw probability scores in the output parquet alongside slot names.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PARQUET,
        help=f"Output Parquet destination path (default: {DEFAULT_OUTPUT_PARQUET}).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (falls back to OPENAI_API_KEY environment variable).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run offline validation, inspect context documents & predicates, and compute cost without calling API.",
    )

    args = parser.parse_args()

    # 1. Load target conversation turns
    df_turns, target_cids = load_target_turns(
        dataset=args.dataset, data_path=args.data_path, sample=args.sample
    )

    # 2. Extract customer contextual documents
    logger.info("Extracting contextualized customer dialogue documents...")
    all_documents = extract_customer_contexts(
        df_turns,
        min_prior_agent_utterances=args.min_prior_agent_utterances,
        group_by_convo=False,
    )
    logger.info("Generated %d customer contextual documents.", len(all_documents))

    # 3. Resolve target slot list
    if args.all_ontology_slots:
        logger.info("Resolving all 25 ontology action slots from schema...")
        df_schema = loader.load_intent_slots(as_df=True)
        global_slots = set()
        df_schema["all_slots"].apply(lambda x: global_slots.update(x))
        available_slots = sorted(global_slots)
    else:
        available_slots = sorted(DEFAULT_SLOT_PREDICATES.keys())

    logger.info(
        "Candidate slot universe (%d slots): %s", len(available_slots), available_slots
    )

    # 4. Handle --dry-run
    if args.dry_run:
        questions = build_decision_predicate_questions(available_slots)
        print("\n" + "=" * 70)
        print("OPENAI DECISIONS API: DRY RUN INSPECTION & COST ESTIMATE")
        print("=" * 70)
        print(f"Model:                 {args.model}")
        print(f"Conversations:         {len(target_cids):,}")
        print(f"Customer Turn Docs:    {len(all_documents):,}")
        print(f"Candidate Slots:       {len(available_slots)} predicates per turn")
        print(f"Predicate Questions:   {len(questions)} per request")
        print(f"Workers Concurrency:   {args.max_workers}")

        # Token & cost estimates
        avg_doc_tokens = 110  # ~110 tokens per contextual dialogue window
        avg_q_tokens = len(questions) * 15  # ~15 tokens per predicate question
        tokens_per_turn = avg_doc_tokens + avg_q_tokens
        total_tokens = tokens_per_turn * len(all_documents)
        estimated_cost = (total_tokens / 1_000_000) * 0.10  # $0.10 / 1M input tokens

        print("-" * 70)
        print(f"Tokens per Request:    ~{tokens_per_turn:,} tokens")
        print(f"Total Input Tokens:    ~{total_tokens:,} tokens")
        print(
            f"Estimated Total Cost:  ~${estimated_cost:.4f} USD (Zero output token charges)"
        )
        print("-" * 70)
        print("\nSample Predicate Questions (First 5):")
        for q in questions[:5]:
            print(f"  * [{q['name']}]: {q['instructions']}")

        if all_documents:
            print("\nSample Contextual Document (First Turn):")
            sample_doc = all_documents[0].get_full_document(
                strip_prior_customer_utterances=True
            )
            for line in sample_doc.splitlines():
                print(f"    {line}")
        print("=" * 70 + "\n")
        logger.info("Dry run completed successfully. No API requests were dispatched.")
        sys.exit(0)

    # 5. Initialize Extractor & Run Inference
    threshold_setting = (
        DEFAULT_SLOT_THRESHOLDS if args.dynamic_thresholds else args.threshold
    )
    extractor = OpenAIDecisionSlotExtractor(
        model_name=args.model,
        threshold=threshold_setting,
        default_threshold=args.threshold,
        api_key=args.api_key,
        max_workers=args.max_workers,
    )

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Executing Decisions API extraction across %d turns...", len(all_documents)
    )
    df_results = extractor.extract_from_conversations(
        df_turns=df_turns,
        available_slots=available_slots,
        min_prior_agent_utterances=args.min_prior_agent_utterances,
        return_scores=args.return_scores,
        max_workers=args.max_workers,
    )

    logger.info("Saving %d extracted turn states to '%s'...", len(df_results), out_p)
    df_results.to_parquet(out_p, index=False)
    logger.info("Decisions API slot extraction completed successfully!")


if __name__ == "__main__":
    main()
