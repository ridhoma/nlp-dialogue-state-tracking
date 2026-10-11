"""
Data Loading API for ABCD Conversation Analytics.
Provides fast, queryable access to conversations, turns, subflows, and guidelines.
"""

import json

import pandas as pd

from abcd_helper.config import (
    ACTIONS_PARQUET,
    CONVERSATIONS_PARQUET,
    INTENT_SCHEMA_PARQUET,
    RAW_GUIDELINES,
    RAW_KB,
    RAW_ONTOLOGY,
    SUBFLOWS_PARQUET,
    TURNS_PARQUET,
)
from abcd_helper.ingestion import run_ingestion
from abcd_helper.viewer import Conversation

# Global caches for instant in-memory repeated access
_CACHED_CONVERSATIONS: pd.DataFrame | None = None
_CACHED_TURNS: pd.DataFrame | None = None
_CACHED_SUBFLOWS: pd.DataFrame | None = None
_CACHED_ACTIONS: pd.DataFrame | None = None
_CACHED_INTENT_SCHEMA: pd.DataFrame | None = None


def _ensure_processed_files() -> None:
    """Ensure Parquet data files exist; run ingestion if missing."""
    if (
        not CONVERSATIONS_PARQUET.exists()
        or not TURNS_PARQUET.exists()
        or not INTENT_SCHEMA_PARQUET.exists()
    ):
        run_ingestion()


def load_conversations(
    split: str | list[str] | None = None,
    intent: str | list[str] | None = None,
    resolved: int | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load conversation-level dataset (1 row per conversation).

    Parameters
    ----------
    split : str or list of str, optional
        Filter by 'train', 'dev', or 'test'.
    intent : str or list of str, optional
        Filter by canonical intent (e.g. 'refund_initiate', 'return_size').
    resolved : int, optional
        Filter by resolution status (1 for resolved, 0 for unresolved).
    use_cache : bool, default True
        Whether to cache the DataFrame in memory for fast subsequent calls.

    Returns
    -------
    pd.DataFrame
        Conversations table with metadata, action sequences, and target variable Y.
    """
    global _CACHED_CONVERSATIONS
    _ensure_processed_files()

    if _CACHED_CONVERSATIONS is None or not use_cache:
        df = pd.read_parquet(CONVERSATIONS_PARQUET)
        if use_cache:
            _CACHED_CONVERSATIONS = df
    else:
        df = _CACHED_CONVERSATIONS.copy()

    # Apply filters
    if split is not None:
        if isinstance(split, str):
            df = df[df["split"] == split]
        else:
            df = df[df["split"].isin(split)]

    if intent is not None:
        if isinstance(intent, str):
            df = df[df["intent"] == intent]
        else:
            df = df[df["intent"].isin(intent)]

    if resolved is not None:
        df = df[df["resolved"] == resolved]

    return df.reset_index(drop=True)


def load_turns(
    convo_id: int | list[int] | None = None,
    split: str | None = None,
    speaker: str | list[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Load turn-level dataset (~220k rows across all conversations).

    Parameters
    ----------
    convo_id : int or list of int, optional
        Filter to one or more conversation IDs.
    split : str, optional
        Filter by 'train', 'dev', or 'test'.
    speaker : str or list of str, optional
        Filter by 'customer', 'agent', or 'action'.
    use_cache : bool, default True
        Whether to cache the DataFrame in memory.

    Returns
    -------
    pd.DataFrame
        Turns table with utterances, speaker tags, actions, and delexed slots.
    """
    global _CACHED_TURNS
    _ensure_processed_files()

    # Fast path for single convo_id without loading full cache if parquet filter supported
    if convo_id is not None and isinstance(convo_id, int) and _CACHED_TURNS is None:
        df = pd.read_parquet(TURNS_PARQUET, filters=[("convo_id", "==", convo_id)])
        if speaker is not None:
            if isinstance(speaker, str):
                df = df[df["speaker"] == speaker]
            else:
                df = df[df["speaker"].isin(speaker)]
        return df.reset_index(drop=True)

    if _CACHED_TURNS is None or not use_cache:
        df = pd.read_parquet(TURNS_PARQUET)
        if use_cache:
            _CACHED_TURNS = df
    else:
        df = _CACHED_TURNS.copy()

    if convo_id is not None:
        if isinstance(convo_id, int):
            df = df[df["convo_id"] == convo_id]
        else:
            df = df[df["convo_id"].isin(convo_id)]

    if split is not None:
        df = df[df["split"] == split]

    if speaker is not None:
        if isinstance(speaker, str):
            df = df[df["speaker"] == speaker]
        else:
            df = df[df["speaker"].isin(speaker)]

    return df.reset_index(drop=True)


def load_subflows() -> pd.DataFrame:
    """Load canonical subflows and expected action SOP paths."""
    global _CACHED_SUBFLOWS
    _ensure_processed_files()
    if _CACHED_SUBFLOWS is None:
        _CACHED_SUBFLOWS = pd.read_parquet(SUBFLOWS_PARQUET)
    return _CACHED_SUBFLOWS.copy()


def load_actions() -> pd.DataFrame:
    """Load the 30 discrete system actions, categories, and required slots."""
    global _CACHED_ACTIONS
    _ensure_processed_files()
    if _CACHED_ACTIONS is None:
        _CACHED_ACTIONS = pd.read_parquet(ACTIONS_PARQUET)
    return _CACHED_ACTIONS.copy()


def load_ontology() -> dict:
    """Load raw ontology.json mapping intents, vocabulary, and actions."""
    with open(RAW_ONTOLOGY, "r", encoding="utf-8") as f:
        return json.load(f)


def load_guidelines() -> dict:
    """Load raw guidelines.json SOP procedures."""
    with open(RAW_GUIDELINES, "r", encoding="utf-8") as f:
        return json.load(f)


def load_kb() -> dict:
    """Load raw kb.json subflow-to-action sequences."""
    with open(RAW_KB, "r", encoding="utf-8") as f:
        return json.load(f)


def load_intent_slots(
    intent: str | None = None,
    flow: str | None = None,
    as_df: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame | dict:
    """
    Load the normative intent schema defining mandatory customer slots and action parameters.

    Ground-truth requirements are derived directly from the ABCD standard operating procedures
    (guidelines.json, ontology.json, and kb.json) across all 55 intents.

    Mandatory customer slots are structured in Conjunctive Normal Form (CNF):
    `list[list[str]]` where:
    - The outer list represents logical AND (each slot group must be satisfied).
    - The inner list represents logical OR (alternative acceptable slot representations,
      such as `['customer_name', 'account_id']` or `['street_address', 'full_address']`).

    Satisfaction for an intent's mandatory slots can be evaluated as:
    ```python
    satisfied = [any(s in extracted_slots for s in group) for group in mandatory_slots]
    is_complete = all(satisfied)
    density = sum(satisfied) / len(mandatory_slots) if mandatory_slots else 1.0
    ```

    Parameters
    ----------
    intent : str, optional
        Canonical intent name (e.g. 'refund_initiate', 'recover_password').
    flow : str, optional
        Conversation flow category (e.g. 'product_defect', 'account_access').
        Recommended when querying generic intent names shared across flows (e.g. 'membership').
    as_df : bool, default False
        If True, always returns a pandas DataFrame even for single-row matches.
        If False (default), returns a dictionary when exactly one matching subflow is resolved.
    use_cache : bool, default True
        Whether to cache the DataFrame in memory for subsequent calls.

    Returns
    -------
    pd.DataFrame or dict
        If both intent and flow are None, returns the complete 55-row schema DataFrame.
        If a specific subflow is matched and as_df is False, returns a dictionary containing:
        - 'flow': str
        - 'intent': str
        - 'canonical_actions': list[str]
        - 'mandatory_slots': list[list[str]] (CNF: outer AND, inner OR)
        - 'num_mandatory_slots': int
        - 'agent_action_slots': list[str]
        - 'all_slots': list[str]
        - 'sop_summary': str
    """
    global _CACHED_INTENT_SCHEMA
    _ensure_processed_files()

    if _CACHED_INTENT_SCHEMA is None or not use_cache:
        df = pd.read_parquet(INTENT_SCHEMA_PARQUET)
        from abcd_helper.intent_schema import ensure_native_lists

        df = ensure_native_lists(df)
        if use_cache:
            _CACHED_INTENT_SCHEMA = df
    else:
        df = _CACHED_INTENT_SCHEMA.copy()

    if flow is not None:
        df = df[df["flow"] == flow]

    if intent is not None:
        df = df[df["intent"] == intent]

    if df.empty:
        raise ValueError(
            f"No matching intent schema found for intent='{intent}', flow='{flow}'."
        )

    if not as_df and len(df) == 1:
        return df.iloc[0].to_dict()

    return df.reset_index(drop=True)


def get_conversation(convo_id: int) -> Conversation:
    """
    Retrieve a Conversation instance for interactive exploration.

    Parameters
    ----------
    convo_id : int
        Unique conversation ID.

    Returns
    -------
    Conversation
        Rich conversation object that renders visually in Jupyter notebooks
        or can be converted to DataFrame / printed.
    """
    df_convos = load_conversations()
    matching_convo = df_convos[df_convos["convo_id"] == convo_id]
    if matching_convo.empty:
        raise ValueError(f"Conversation ID {convo_id} not found.")

    convo_meta = matching_convo.iloc[0].to_dict()
    turns_df = load_turns(convo_id=convo_id)
    return Conversation(convo_meta=convo_meta, turns_df=turns_df)
