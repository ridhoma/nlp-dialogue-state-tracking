"""
OpenAI API Utilities and Helpers for Dialogue State Tracking (DST).

Provides Pydantic structured output schemas, dialogue formatting for whole-conversation
prompts, batch JSONL generation for the OpenAI Batch API, and predicate question
builders for the OpenAI Decisions API.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

# Default model identifiers
DEFAULT_OPENAI_BATCH_MODEL = "gpt-6-luna"
DEFAULT_OPENAI_DECISION_MODEL = "gpt-6-luna"

logger = logging.getLogger(__name__)


# =============================================================================
# 1. Pydantic Schemas for Structured JSON Output
# =============================================================================


class TurnSlotAnnotation(BaseModel):
    """Slot annotations for a single customer turn."""

    conversation_id: int = Field(description="Unique conversation identifier")
    turn_id: int = Field(description="Turn ID of the customer utterance")
    slots: list[str] = Field(
        default_factory=list,
        description="List of active slot names disclosed by the customer in this turn",
    )


class ConversationSlotExtractionBatch(BaseModel):
    """Collection of turn slot annotations for one or more conversations."""

    annotations: list[TurnSlotAnnotation] = Field(
        default_factory=list,
        description="List of customer turn slot annotations",
    )


# =============================================================================
# 2. Client Authentication & Factory
# =============================================================================


def get_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 60.0,
    max_retries: int = 3,
    **kwargs: Any,
) -> Any:
    """
    Initialize and return an authenticated OpenAI client.

    Parameters
    ----------
    api_key : str, optional
        OpenAI API key. If omitted, read from the `OPENAI_API_KEY` environment variable.
    base_url : str, optional
        Custom API base URL (e.g. for enterprise or proxy endpoints).
    timeout : float, default 60.0
        Request timeout in seconds.
    max_retries : int, default 3
        Number of automatic retry attempts for transient errors.
    **kwargs : Any
        Additional keyword arguments forwarded to `openai.OpenAI()`.

    Returns
    -------
    openai.OpenAI
        Configured OpenAI client instance.

    Raises
    ------
    ValueError
        If no API key is provided and `OPENAI_API_KEY` is unset.
    """
    from openai import OpenAI

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not resolved_key:
        raise ValueError(
            "OpenAI API key is missing. Please provide `api_key` or set the "
            "`OPENAI_API_KEY` environment variable."
        )

    return OpenAI(
        api_key=resolved_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        **kwargs,
    )


# =============================================================================
# 3. Conversation Formatting & Dialogue Serialization
# =============================================================================


def format_conversation_for_llm(
    turns: list[dict[str, Any]] | pd.DataFrame,
    convo_id: int | str | None = None,
    include_actions: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """
    Format a single conversation into the exact JSON input structure
    prescribed by `SLOTS_EXTRACTION_PROMPT`.

    Parameters
    ----------
    turns : list[dict] | pd.DataFrame
        Turn sequence for the conversation.
    convo_id : int | str, optional
        Conversation identifier. Inferred from records if omitted.
    include_actions : bool, default False
        Whether to retain action logs (`speaker == 'action'`).

    Returns
    -------
    dict[str, list[dict]]
        Dictionary with `<convo_id>` as key and turn exchanges as value:
        `{"2101": [{"turn_id": 1, "speaker": "AGENT", "text": "..."}, ...]}`
    """
    records = (
        turns.to_dict(orient="records") if isinstance(turns, pd.DataFrame) else turns
    )
    if not records:
        cid = str(convo_id) if convo_id is not None else "0"
        return {cid: []}

    resolved_cid = str(
        convo_id if convo_id is not None else records[0].get("convo_id", "0")
    )

    formatted_turns: list[dict[str, Any]] = []
    for r in records:
        spk = str(r.get("speaker", "")).lower()
        if spk == "action" and not include_actions:
            continue
        formatted_turns.append(
            {
                "turn_id": int(r.get("turn_id", 0)),
                "speaker": str(r.get("speaker", "")).upper(),
                "text": str(r.get("text", "")),
            }
        )

    # Sort turns chronologically
    formatted_turns.sort(key=lambda t: t["turn_id"])
    return {resolved_cid: formatted_turns}


def format_conversations_batch_for_llm(
    df_turns: pd.DataFrame,
    convo_ids: Iterable[int | str],
    include_actions: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """
    Format a batch of conversations into a single dictionary mapping
    `<convo_id> -> list[turns]`.
    """
    batch_dict: dict[str, list[dict[str, Any]]] = {}
    subset = df_turns[df_turns["convo_id"].isin(convo_ids)]
    for cid, c_turns in subset.groupby("convo_id"):
        batch_dict.update(
            format_conversation_for_llm(
                c_turns, convo_id=cid, include_actions=include_actions
            )
        )
    return batch_dict


# =============================================================================
# 4. OpenAI Batch API JSONL Generation
# =============================================================================


def build_batch_request_line(
    custom_id: str,
    conversations_payload: dict[str, list[dict[str, Any]]],
    model: str = DEFAULT_OPENAI_BATCH_MODEL,
    endpoint: Literal["/v1/responses", "/v1/chat/completions"] = "/v1/responses",
    prompt: str | None = None,
) -> dict[str, Any]:
    """
    Build a single JSONL request line for the OpenAI Batch API.

    Parameters
    ----------
    custom_id : str
        Unique identifier for mapping the request back to the input data.
    conversations_payload : dict
        Formatted conversation(s) dictionary.
    model : str, default 'gpt-6-luna'
        Target OpenAI model identifier.
    endpoint : {'/v1/responses', '/v1/chat/completions'}, default '/v1/responses'
        Target API endpoint path.
    prompt : str, optional
        Extraction instructions prompt (defaults to SLOTS_EXTRACTION_PROMPT).

    Returns
    -------
    dict[str, Any]
        Dictionary matching the OpenAI Batch API schema.
    """
    if prompt is None:
        from nlp.prompts import SLOTS_EXTRACTION_PROMPT

        resolved_prompt = SLOTS_EXTRACTION_PROMPT
    else:
        resolved_prompt = prompt

    json_schema = ConversationSlotExtractionBatch.model_json_schema()
    serialized_input = json.dumps(conversations_payload, indent=2)

    if endpoint == "/v1/responses":
        body: dict[str, Any] = {
            "model": model,
            "instructions": resolved_prompt,
            "input": serialized_input,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "slot_extraction_batch",
                    "strict": True,
                    "schema": json_schema,
                }
            },
        }
    elif endpoint == "/v1/chat/completions":
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": resolved_prompt},
                {"role": "user", "content": serialized_input},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "slot_extraction_batch",
                    "strict": True,
                    "schema": json_schema,
                },
            },
        }
    else:
        raise ValueError(
            f"Unsupported endpoint '{endpoint}'. Supported: '/v1/responses', '/v1/chat/completions'"
        )

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": endpoint,
        "body": body,
    }


def prepare_batch_input_jsonl(
    df_turns: pd.DataFrame,
    convo_ids: list[int | str] | None = None,
    output_path: str | Path | None = None,
    model: str = DEFAULT_OPENAI_BATCH_MODEL,
    endpoint: Literal["/v1/responses", "/v1/chat/completions"] = "/v1/responses",
    convos_per_request: int = 1,
    prompt: str | None = None,
) -> list[dict[str, Any]]:
    """
    Generate the complete list of JSONL lines (and optionally write to file)
    for submitting a batch job to OpenAI.

    Parameters
    ----------
    df_turns : pd.DataFrame
        Complete turns DataFrame.
    convo_ids : list[int | str], optional
        Subset of conversation IDs to include. If None, uses all unique IDs in `df_turns`.
    output_path : str | Path, optional
        Destination `.jsonl` file path.
    model : str, default 'gpt-6-luna'
        OpenAI model identifier.
    endpoint : {'/v1/responses', '/v1/chat/completions'}, default '/v1/responses'
        OpenAI endpoint path.
    convos_per_request : int, default 1
        Number of conversations bundled per API request.
    prompt : str, optional
        Extraction instructions prompt (defaults to SLOTS_EXTRACTION_PROMPT).

    Returns
    -------
    list[dict[str, Any]]
        List of batch request line dictionaries.
    """
    from itertools import batched

    if prompt is None:
        from nlp.prompts import SLOTS_EXTRACTION_PROMPT

        resolved_prompt = SLOTS_EXTRACTION_PROMPT
    else:
        resolved_prompt = prompt

    if convo_ids is None:
        target_ids = list(df_turns["convo_id"].unique())
    else:
        target_ids = list(convo_ids)

    logger.info(
        "Preparing batch JSONL for %d conversations (convos_per_request=%d, endpoint=%s, model=%s)",
        len(target_ids),
        convos_per_request,
        endpoint,
        model,
    )

    batch_lines: list[dict[str, Any]] = []
    chunk_list = list(batched(target_ids, convos_per_request))

    for chunk_idx, chunk_cids in enumerate(chunk_list):
        batch_payload = format_conversations_batch_for_llm(df_turns, chunk_cids)
        custom_id = (
            f"convo_{chunk_cids[0]}"
            if len(chunk_cids) == 1
            else f"batch_{chunk_idx}_convos_{chunk_cids[0]}_{chunk_cids[-1]}"
        )
        line = build_batch_request_line(
            custom_id=custom_id,
            conversations_payload=batch_payload,
            model=model,
            endpoint=endpoint,
            prompt=resolved_prompt,
        )
        batch_lines.append(line)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(item) + "\n" for item in batch_lines)
        logger.info(
            "Successfully wrote %d batch request lines to '%s'", len(batch_lines), out_p
        )

    return batch_lines


# =============================================================================
# 5. Parsing Batch API Results
# =============================================================================


def extract_content_text_from_batch_response(body: dict[str, Any]) -> str:
    """
    Extract the raw response text from either a `/v1/responses` or
    `/v1/chat/completions` response body dictionary.
    """
    # 1. Check for /v1/chat/completions structure
    choices = body.get("choices")
    if choices and isinstance(choices, list) and len(choices) > 0:
        first_choice = choices[0]
        msg = first_choice.get("message", {})
        content = msg.get("content")
        if content:
            return str(content)

    # 2. Check for /v1/responses structure
    output_items = body.get("output")
    if output_items and isinstance(output_items, list):
        for item in output_items:
            content_list = item.get("content")
            if isinstance(content_list, list):
                for part in content_list:
                    if isinstance(part, dict) and "text" in part:
                        return str(part["text"])

    # 3. Direct text fallback
    if "text" in body and isinstance(body["text"], str):
        return body["text"]

    raise ValueError(
        f"Could not locate generated text content in response body: {list(body.keys())}"
    )


def parse_llm_slot_annotations(raw_text: str) -> list[TurnSlotAnnotation]:
    """
    Parse LLM output text into a list of structured `TurnSlotAnnotation` objects.
    Resilient to markdown code blocks, raw JSON arrays, and root objects.
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    data = json.loads(cleaned)
    raw_list: list[dict[str, Any]] = []

    if isinstance(data, list):
        raw_list = data
    elif isinstance(data, dict):
        if "annotations" in data and isinstance(data["annotations"], list):
            raw_list = data["annotations"]
        else:
            # Fallback: find first list value
            for val in data.values():
                if isinstance(val, list):
                    raw_list = val
                    break

    results: list[TurnSlotAnnotation] = []
    for item in raw_list:
        if isinstance(item, dict):
            cid = int(item.get("conversation_id", item.get("convo_id", 0)))
            tid = int(item.get("turn_id", 0))
            raw_slots = item.get("slots", [])
            slots_list = [str(s).strip() for s in raw_slots if s and str(s).strip()]
            results.append(
                TurnSlotAnnotation(
                    conversation_id=cid,
                    turn_id=tid,
                    slots=sorted(set(slots_list)),
                )
            )

    return results


def parse_batch_output_jsonl(
    jsonl_content_or_path: str | Path,
) -> pd.DataFrame:
    """
    Parse downloaded OpenAI Batch API output JSONL content or file into
    a standardized DataFrame with columns `['convo_id', 'turn_id', 'inferred_slots']`.

    Parameters
    ----------
    jsonl_content_or_path : str | Path
        Raw string content or filesystem Path to the `.jsonl` file.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns `['convo_id', 'turn_id', 'inferred_slots']`.
    """
    lines: list[str] = []
    is_file = False
    if isinstance(jsonl_content_or_path, Path):
        is_file = jsonl_content_or_path.is_file()
    elif isinstance(jsonl_content_or_path, str) and "\n" not in jsonl_content_or_path:
        try:
            is_file = Path(jsonl_content_or_path).is_file()
        except OSError:
            is_file = False

    if is_file:
        with open(jsonl_content_or_path, encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = str(jsonl_content_or_path).strip().splitlines()

    records: list[dict[str, Any]] = []
    error_count = 0

    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as err:
            logger.warning("Line %d failed JSON decode: %s", line_idx, err)
            error_count += 1
            continue

        err_obj = item.get("error")
        if err_obj:
            logger.warning("Line %d returned API error: %s", line_idx, err_obj)
            error_count += 1
            continue

        resp = item.get("response", {})
        status_code = resp.get("status_code", 200)
        if status_code != 200:
            logger.warning("Line %d returned non-200 status %s", line_idx, status_code)
            error_count += 1
            continue

        body = resp.get("body", {})
        try:
            content_text = extract_content_text_from_batch_response(body)
            annotations = parse_llm_slot_annotations(content_text)
            for ann in annotations:
                records.append(
                    {
                        "convo_id": ann.conversation_id,
                        "turn_id": ann.turn_id,
                        "inferred_slots": ann.slots,
                    }
                )
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as ex:
            logger.warning("Failed parsing line %d content: %s", line_idx, ex)
            error_count += 1

    logger.info(
        "Parsed %d turn slot annotations from batch output (%d errors).",
        len(records),
        error_count,
    )

    df_out = pd.DataFrame(records)
    if df_out.empty:
        return pd.DataFrame(columns=["convo_id", "turn_id", "inferred_slots"])

    df_out["convo_id"] = df_out["convo_id"].astype("int64")
    df_out["turn_id"] = df_out["turn_id"].astype("int64")
    df_out = df_out.sort_values(["convo_id", "turn_id"]).reset_index(drop=True)
    return df_out


# =============================================================================
# 6. OpenAI Decisions API Helpers
# =============================================================================


def build_decision_predicate_questions(
    available_slots: Iterable[str],
    custom_instructions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Build an array of `predicate` questions for the OpenAI Decisions API
    (`client.decisions.create`), asking whether each slot is disclosed.

    Parameters
    ----------
    available_slots : Iterable[str]
        List of candidate slot names to evaluate.
    custom_instructions : dict[str, str], optional
        Custom instructions mapping slot -> instruction string.

    Returns
    -------
    list[dict[str, Any]]
        Question dictionaries formatted for `client.decisions.create`.
    """
    if custom_instructions is not None:
        instructions_map = custom_instructions
    else:
        from nlp.prompts import DEFAULT_SLOT_PREDICATES

        instructions_map = DEFAULT_SLOT_PREDICATES
    questions: list[dict[str, Any]] = []

    for slot in available_slots:
        instruction = instructions_map.get(
            slot,
            f"Does the customer specify, enter, or confirm their {slot.replace('_', ' ')}?",
        )
        questions.append(
            {
                "name": slot,
                "type": "predicate",
                "instructions": instruction,
            }
        )

    return questions


def parse_decision_answers(
    answers: list[Any],
    thresholds: dict[str, float] | float = 0.50,
    default_threshold: float = 0.50,
) -> dict[str, float]:
    """
    Parse answers from a `Decision` response, returning a mapping of
    active slot names to their predicted probabilities.

    Parameters
    ----------
    answers : list[Any]
        The `.answers` attribute from `client.decisions.create`.
    thresholds : dict[str, float] | float, default 0.50
        Confidence threshold cutoff(s).
    default_threshold : float, default 0.50
        Fallback threshold if slot not in dictionary.

    Returns
    -------
    dict[str, float]
        Mapping of active slot name to probability score.
    """
    scores: dict[str, float] = {}

    for ans in answers:
        # Support both Pydantic model attributes and dictionary access
        name = getattr(ans, "name", None) or (
            ans.get("name") if isinstance(ans, dict) else None
        )
        prob = getattr(ans, "probability", None) or (
            ans.get("probability") if isinstance(ans, dict) else None
        )

        if name is None or prob is None:
            continue

        prob_val = float(prob)
        if isinstance(thresholds, dict):
            cutoff = thresholds.get(name, thresholds.get("default", default_threshold))
        else:
            cutoff = float(thresholds)

        if prob_val >= cutoff:
            scores[str(name)] = prob_val

    return scores
