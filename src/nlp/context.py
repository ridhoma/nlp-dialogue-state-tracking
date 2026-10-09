"""
Dialogue Context Window Extraction for Conversational NLP.

Provides utilities to construct rolling conversational context windows for customer
utterances. Specifically rolls the context window backward until a required number
of prior agent utterances (default: 2) are captured, bridging sub-dialogues,
clarification loops, and burst customer responses.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DialogueTurnContext:
    """
    Encapsulates a customer utterance alongside its rolling conversational history.

    Attributes
    ----------
    turn_id : int
        The sequence identifier of the target customer turn.
    customer_text : str
        The raw natural language utterance spoken by the customer.
    customer_text_delexed : str
        The delexicalized customer utterance containing slot tokens (if available).
    convo_id : int | None
        Conversation identifier, if known.
    prior_turns : list[dict[str, Any]]
        All prior dialogue turns included in the rolling window (in chronological order).
    prior_agent_turns : list[dict[str, Any]]
        The subset of prior turns spoken by the agent (in chronological order).
    num_prior_agent_utterances : int
        Total count of prior agent utterances captured in this window.
    """

    turn_id: int
    customer_text: str
    customer_text_delexed: str = ""
    convo_id: int | None = None
    prior_turns: list[dict[str, Any]] = field(default_factory=list)
    prior_agent_turns: list[dict[str, Any]] = field(default_factory=list)
    num_prior_agent_utterances: int = 0

    def get_formatted_context(
        self, strip_prior_customer_utterances: bool = False
    ) -> str:
        """
        Formats prior turns into a clean, human-readable dialogue context string.

        Parameters
        ----------
        strip_prior_customer_utterances : bool, default=False
            If True, only includes prior agent utterances in the context window.
        """
        turns_to_include = (
            self.prior_agent_turns
            if strip_prior_customer_utterances
            else self.prior_turns
        )
        if not turns_to_include:
            return ""
        lines = []
        for t in turns_to_include:
            spk = str(t.get("speaker", "unknown")).upper()
            text = str(t.get("text", "")).strip()
            lines.append(f"[{spk}]: {text}")
        return "\n".join(lines)

    @property
    def formatted_context(self) -> str:
        """Property wrapper for backwards compatibility."""
        return self.get_formatted_context()

    def get_full_document(self, strip_prior_customer_utterances: bool = False) -> str:
        """
        Formats the prior context combined with the target customer message.

        Parameters
        ----------
        strip_prior_customer_utterances : bool, default=False
            If True, only includes prior agent utterances in the context window.
        """
        context_str = self.get_formatted_context(strip_prior_customer_utterances)
        current_str = f"[CUSTOMER]: {self.customer_text.strip()}"
        if context_str:
            return f"{context_str}\n{current_str}"
        return current_str

    @property
    def full_document(self) -> str:
        """Property wrapper for backwards compatibility."""
        return self.get_full_document()


def _normalize_turns(
    turns: list[dict[str, Any]] | pd.DataFrame,
) -> list[dict[str, Any]]:
    """Convert input turns into a uniform list of dictionaries."""
    if isinstance(turns, pd.DataFrame):
        return turns.to_dict(orient="records")
    if isinstance(turns, list):
        return turns
    raise TypeError(
        f"Expected turns to be a list[dict] or pd.DataFrame, got {type(turns).__name__}"
    )


def get_customer_turn_context(
    turns: list[dict[str, Any]] | pd.DataFrame,
    current_turn_idx: int,
    convo_id: int | None = None,
    min_prior_agent_utterances: int = 2,
    include_actions: bool = False,
) -> DialogueTurnContext | None:
    """
    Rolls the context window backwards from a customer turn until at least
    `min_prior_agent_utterances` agent utterances are captured.

    Parameters
    ----------
    turns : list[dict] | pd.DataFrame
        Turn sequence for a conversation (or multi-conversation DataFrame).
    current_turn_idx : int
        The index or turn_id of the customer turn to build context for.
    convo_id : int, optional
        Filter turns to this specific conversation ID before building context.
    min_prior_agent_utterances : int, default=2
        Minimum number of prior agent utterances to collect. The window stops
        expanding backwards once this threshold is satisfied or the start of
        the conversation is reached.
    include_actions : bool, default=False
        Whether to include system action logs (speaker == 'action') in the
        collected prior turns. Defaults to False (only natural language turns).

    Returns
    -------
    DialogueTurnContext | None
        Structured object containing the target utterance, chronological prior turns,
        prior agent utterances, and pre-formatted context text, or None if the turn
        was not spoken by a customer.
    """
    records = _normalize_turns(turns)
    if not records:
        raise ValueError("Cannot extract context from empty turns sequence.")

    if convo_id is not None:
        records = [r for r in records if r.get("convo_id") == convo_id]
        if not records:
            raise ValueError(f"No turns found for conversation ID {convo_id}.")

    # Locate target turn position
    target_pos = None
    for idx, r in enumerate(records):
        if r.get("turn_id") == current_turn_idx:
            target_pos = idx
            break

    if target_pos is None:
        # Fallback to positional index if valid
        if 0 <= current_turn_idx < len(records):
            target_pos = current_turn_idx
        else:
            raise IndexError(
                f"Turn identifier or index {current_turn_idx} not found in conversation turns "
                f"(length: {len(records)})."
            )

    target_turn = records[target_pos]

    # Validate speaker is customer
    speaker = target_turn.get("speaker")
    if speaker != "customer":
        cid_str = f" in conversation {convo_id}" if convo_id is not None else ""
        msg = f"Turn {current_turn_idx}{cid_str} has speaker '{speaker}', expected 'customer'."
        warnings.warn(msg, UserWarning, stacklevel=2)
        logger.warning("%s", msg)
        return None

    # Roll window backwards collecting prior turns within this conversation
    collected_prior: list[dict[str, Any]] = []
    collected_agent: list[dict[str, Any]] = []
    agent_count = 0

    for i in range(target_pos - 1, -1, -1):
        prev = records[i]
        spk = prev.get("speaker", "")

        # Check if action turns should be skipped in context
        if spk == "action" and not include_actions:
            continue

        collected_prior.append(prev)

        if spk == "agent":
            collected_agent.append(prev)
            agent_count += 1
            if agent_count >= min_prior_agent_utterances:
                break

    # Restore chronological order (earliest to latest)
    collected_prior.reverse()
    collected_agent.reverse()

    return DialogueTurnContext(
        turn_id=target_turn.get("turn_id", target_pos),
        customer_text=str(target_turn.get("text", "")),
        customer_text_delexed=str(target_turn.get("text_delexed", "")),
        convo_id=target_turn.get("convo_id"),
        prior_turns=collected_prior,
        prior_agent_turns=collected_agent,
        num_prior_agent_utterances=len(collected_agent),
    )


def extract_customer_contexts(
    turns: list[dict[str, Any]] | pd.DataFrame,
    min_prior_agent_utterances: int = 2,
    include_actions: bool = False,
    group_by_convo: bool = False,
) -> list[DialogueTurnContext] | dict[Any, list[DialogueTurnContext]]:
    """
    Batch extracts context windows for all customer turns across one or multiple conversations.

    When multiple conversations are present, turns are grouped by `convo_id`,
    ensuring that the rolling context window strictly respects conversation
    boundaries and never bleeds across conversations.

    Parameters
    ----------
    turns : list[dict] | pd.DataFrame
        Turn sequence for a single conversation or an entire multi-conversation corpus.
    min_prior_agent_utterances : int, default=2
        Minimum prior agent utterances to collect per customer turn.
    include_actions : bool, default=False
        Whether to retain action logs in prior turns.
    group_by_convo : bool, default=False
        If True, returns a dictionary mapping `convo_id -> list[DialogueTurnContext]`.
        If False, returns a flat list of `DialogueTurnContext` for all customer turns.

    Returns
    -------
    list[DialogueTurnContext] | dict[Any, list[DialogueTurnContext]]
        Extracted context objects, either as a flat list or grouped by conversation ID.
    """
    from collections import defaultdict

    records = _normalize_turns(turns)
    if not records:
        return {} if group_by_convo else []

    # Group by convo_id to ensure conversation boundary isolation
    convo_groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        convo_groups[r.get("convo_id")].append(r)

    results_grouped: dict[Any, list[DialogueTurnContext]] = {}
    results_flat: list[DialogueTurnContext] = []

    for cid, c_records in convo_groups.items():
        c_contexts: list[DialogueTurnContext] = []
        for idx, r in enumerate(c_records):
            if r.get("speaker") == "customer":
                # Roll window backward inside this specific conversation
                collected_prior = []
                collected_agent = []
                agent_count = 0
                for j in range(idx - 1, -1, -1):
                    prev = c_records[j]
                    spk = prev.get("speaker", "")
                    if spk == "action" and not include_actions:
                        continue
                    collected_prior.append(prev)
                    if spk == "agent":
                        collected_agent.append(prev)
                        agent_count += 1
                        if agent_count >= min_prior_agent_utterances:
                            break

                collected_prior.reverse()
                collected_agent.reverse()

                ctx = DialogueTurnContext(
                    turn_id=r.get("turn_id", idx),
                    customer_text=str(r.get("text", "")),
                    customer_text_delexed=str(r.get("text_delexed", "")),
                    convo_id=cid,
                    prior_turns=collected_prior,
                    prior_agent_turns=collected_agent,
                    num_prior_agent_utterances=len(collected_agent),
                )
                c_contexts.append(ctx)
                results_flat.append(ctx)

        results_grouped[cid] = c_contexts

    if group_by_convo:
        return results_grouped
    return results_flat


def extract_customer_utterance(text: str) -> str:
    """
    Extracts the customer utterance from a contextualized dialogue document.

    If the document contains `[CUSTOMER]: <text>`, extracts the trailing
    customer message. Otherwise, returns the stripped input text.

    Parameters
    ----------
    text : str
        The input dialogue string (either a contextual document or raw text).

    Returns
    -------
    str
        The stripped customer utterance.
    """
    if not text:
        return ""
    if "[CUSTOMER]:" in text:
        return text.rsplit("[CUSTOMER]:", 1)[-1].strip()
    return text.strip()

