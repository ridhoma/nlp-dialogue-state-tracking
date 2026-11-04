#!/usr/bin/env python
"""
Run Dialogue State Tracking (Slot Extraction) via the OpenAI Batch API.

Uses whole-conversation structured prompts (`SLOTS_EXTRACTION_PROMPT`) with
`gpt-6-luna` (or `gpt-4o-mini`) via `/v1/responses` or `/v1/chat/completions` at a
50% discount rate.

Supports staged offline execution:
1. `prepare`: Generates the `.jsonl` input file without making API calls (no API key needed).
2. `submit`: Uploads the input file and creates an asynchronous batch job.
3. `poll`: Checks status of an ongoing batch job.
4. `download`: Downloads completed batch results and generates the final Parquet table.
5. `full`: Executes all steps end-to-end.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from abcd_helper import loader
from nlp.slots import OpenAIBatchSlotExtractor
from utils.openai_helper import (
    DEFAULT_OPENAI_BATCH_MODEL,
    parse_batch_output_jsonl,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BENCHMARK_EVAL_SET_PATH = "data/samples/slot_extraction_evaluation_set.parquet"
DEFAULT_OUTPUT_PARQUET = "data/processed/customer_slots_extracted_openai_batch.parquet"
DEFAULT_INPUT_JSONL = "data/processed/batch_input_openai.jsonl"
DEFAULT_OUTPUT_JSONL = "data/processed/batch_output_openai.jsonl"


def load_target_turns(
    dataset: str,
    data_path: str | None = None,
    sample: int | None = None,
) -> tuple[pd.DataFrame, list[int | str]]:
    """
    Load dialogue turns and resolve target conversation IDs based on dataset selection.

    Parameters
    ----------
    dataset : {'benchmark', 'full'}
        Dataset mode.
    data_path : str, optional
        Explicit path to a dataset file.
    sample : int, optional
        Number of conversations to sample.

    Returns
    -------
    tuple[pd.DataFrame, list[int | str]]
        The loaded turns DataFrame and the list of target conversation IDs.
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
        logger.info("Sampled %d conversations for batch execution.", len(target_cids))

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
        description="Run Slot Extraction via OpenAI Batch API (gpt-6-luna / responses endpoint)"
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
        default=DEFAULT_OPENAI_BATCH_MODEL,
        help=f"OpenAI model identifier (default: {DEFAULT_OPENAI_BATCH_MODEL}).",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        choices=["/v1/responses", "/v1/chat/completions"],
        default="/v1/responses",
        help="Target OpenAI API endpoint for batch requests (default: /v1/responses).",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["prepare", "submit", "poll", "download", "full"],
        default="full",
        help=(
            "Action to perform: 'prepare' (offline JSONL generation only), "
            "'submit' (upload & launch batch), 'poll' (wait for completion), "
            "'download' (download results & save parquet), or 'full' (all steps). Default: full."
        ),
    )
    parser.add_argument(
        "--batch-input-jsonl",
        type=str,
        default=DEFAULT_INPUT_JSONL,
        help=f"File path for the prepared batch input JSONL (default: {DEFAULT_INPUT_JSONL}).",
    )
    parser.add_argument(
        "--batch-output-jsonl",
        type=str,
        default=DEFAULT_OUTPUT_JSONL,
        help=f"File path for saving downloaded batch output JSONL (default: {DEFAULT_OUTPUT_JSONL}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PARQUET,
        help=f"Output Parquet destination path (default: {DEFAULT_OUTPUT_PARQUET}).",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Existing OpenAI Batch ID (required for 'poll' or 'download' actions if not running 'full').",
    )
    parser.add_argument(
        "--convos-per-request",
        type=int,
        default=1,
        help="Number of conversations bundled per API request line (default: 1).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (falls back to OPENAI_API_KEY environment variable).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between batch status checks during polling (default: 30).",
    )

    args = parser.parse_args()

    extractor = OpenAIBatchSlotExtractor(
        model_name=args.model,
        endpoint=args.endpoint,
        api_key=args.api_key,
        convos_per_request=args.convos_per_request,
    )

    # -------------------------------------------------------------------------
    # Action: prepare (Offline JSONL Generation)
    # -------------------------------------------------------------------------
    if args.action == "prepare":
        df_turns, target_cids = load_target_turns(
            dataset=args.dataset, data_path=args.data_path, sample=args.sample
        )
        out_p = Path(args.batch_input_jsonl)
        lines = extractor.prepare_batch_file(
            df_turns=df_turns,
            convo_ids=target_cids,
            output_path=out_p,
        )
        logger.info(
            "Successfully generated batch input JSONL with %d requests at '%s'. "
            "Ready for submission with `--action submit`.",
            len(lines),
            out_p,
        )
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Action: submit (Upload & Launch Batch)
    # -------------------------------------------------------------------------
    if args.action == "submit":
        in_p = Path(args.batch_input_jsonl)
        if not in_p.exists():
            logger.info(
                "Batch input file '%s' does not exist. Generating it first...", in_p
            )
            df_turns, target_cids = load_target_turns(
                dataset=args.dataset, data_path=args.data_path, sample=args.sample
            )
            extractor.prepare_batch_file(
                df_turns, convo_ids=target_cids, output_path=in_p
            )

        batch_id = extractor.submit_batch(in_p)
        print(f"\nBATCH SUBMISSION SUCCESSFUL!\nBatch ID: {batch_id}\n")
        print(
            f"To poll status, run:\n  python {sys.argv[0]} --action poll --batch-id {batch_id}"
        )
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Action: poll (Wait for Completion)
    # -------------------------------------------------------------------------
    if args.action == "poll":
        if not args.batch_id:
            logger.error("--batch-id is required for action 'poll'.")
            sys.exit(1)
        extractor.poll_batch(args.batch_id, poll_interval_seconds=args.poll_interval)
        print(f"\nBatch {args.batch_id} has completed! To download results, run:")
        print(
            f"  python {sys.argv[0]} --action download --batch-id {args.batch_id} --output {args.output}"
        )
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Action: download (Download & Save Parquet)
    # -------------------------------------------------------------------------
    if args.action == "download":
        if not args.batch_id and not Path(args.batch_output_jsonl).exists():
            logger.error(
                "Either --batch-id or an existing --batch-output-jsonl file is required for 'download'."
            )
            sys.exit(1)

        if args.batch_id:
            df_results = extractor.download_and_parse_results(
                batch_id=args.batch_id,
                download_path=args.batch_output_jsonl,
            )
        else:
            logger.info(
                "Parsing existing local batch output JSONL '%s'...",
                args.batch_output_jsonl,
            )
            df_results = parse_batch_output_jsonl(args.batch_output_jsonl)

        out_parquet = Path(args.output)
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_parquet(out_parquet, index=False)
        logger.info(
            "Saved %d extracted turn states to '%s'.", len(df_results), out_parquet
        )
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Action: full (End-to-End Execution)
    # -------------------------------------------------------------------------
    if args.action == "full":
        df_turns, target_cids = load_target_turns(
            dataset=args.dataset, data_path=args.data_path, sample=args.sample
        )
        in_p = Path(args.batch_input_jsonl)
        out_p = Path(args.output)

        logger.info("1/4: Preparing batch input JSONL at '%s'...", in_p)
        extractor.prepare_batch_file(df_turns, convo_ids=target_cids, output_path=in_p)

        logger.info(
            "2/4: Submitting batch job to OpenAI (%s on %s)...",
            args.model,
            args.endpoint,
        )
        batch_id = extractor.submit_batch(in_p)

        logger.info("3/4: Polling batch status (Batch ID: %s)...", batch_id)
        extractor.poll_batch(batch_id, poll_interval_seconds=args.poll_interval)

        logger.info("4/4: Downloading results and saving Parquet to '%s'...", out_p)
        df_results = extractor.download_and_parse_results(
            batch_id=batch_id,
            download_path=args.batch_output_jsonl,
        )
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_parquet(out_p, index=False)

        logger.info(
            "OpenAI Batch slot extraction complete! Processed %d turn states -> '%s'.",
            len(df_results),
            out_p,
        )


if __name__ == "__main__":
    main()
