"""
Generic Interactive Conversation Viewer for Jupyter Notebooks and CLI.

Provides dataset-agnostic conversation display objects and orchestration,
delegating HTML markup rendering to `utils.rendering`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from utils.rendering import render_conversation_html


class ConversationView:
    """
    Renderable conversation container.
    Implements `_repr_html_()` for native rendering in Jupyter notebooks.
    """

    def __init__(
        self,
        html_content: str,
        plain_text: str,
        turns: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ):
        self._html = html_content
        self._text = plain_text
        self.turns = turns
        self.metadata = metadata or {}

    def _repr_html_(self) -> str:
        return self._html

    def __repr__(self) -> str:
        return f"<ConversationView: {len(self.turns)} turns>"

    def __str__(self) -> str:
        return self._text

    def print_transcript(self) -> None:
        """Print plain text representation to standard output."""
        print(self._text)


def render_conversation(
    turns: pd.DataFrame | list[dict[str, Any]],
    title: str | None = None,
    metadata: dict[str, Any] | None = None,
    labels: str | pd.Series | dict[Any, Any] | list[Any] | None = None,
    status: str | None = None,
    status_color: str | None = None,
    speaker_col: str = "speaker",
    text_col: str = "text",
    turn_id_col: str = "turn_id",
    action_name_col: str = "action_name",
    action_values_col: str = "action_values",
    max_height: str = "520px",
) -> ConversationView:
    """
    Renders any conversation sequence into an interactive visualization.

    Parameters
    ----------
    turns : pd.DataFrame | list[dict]
        Turns sequence containing speaker and utterance text.
    title : str, optional
        Title header for the conversation viewer (e.g. 'Conversation #3592').
    metadata : dict, optional
        Key-value dictionary displayed in the top subtitle banner.
    labels : str | pd.Series | dict | list, optional
        Labels to attach to turns (e.g. inferred slots, intents, emotion):
        - str: column name inside `turns` (e.g. 'inferred_slots').
        - dict: mapping of turn_id/index -> label(s).
        - Series/list: aligned sequence of labels matching turns length.
    status : str, optional
        Status badge text (e.g. 'Resolved', 'Failed', 'Escalated').
    status_color : str, optional
        CSS background color or preset ('success', 'danger', 'warning').
    speaker_col : str, default='speaker'
        Column name identifying the speaker.
    text_col : str, default='text'
        Column name containing the dialogue text.
    turn_id_col : str, default='turn_id'
        Column name for turn sequence numbers.
    action_name_col : str, default='action_name'
        Column name for system action names (if present).
    action_values_col : str, default='action_values'
        Column name for system action arguments (if present).
    max_height : str, default='520px'
        CSS maximum height for the scrollable dialogue container.

    Returns
    -------
    ConversationView
        Renderable container with `_repr_html_()` and plain text methods.
    """
    # Normalize turns into list of dicts
    if isinstance(turns, pd.DataFrame):
        records = turns.to_dict(orient="records")
    elif isinstance(turns, list):
        records = [dict(t) for t in turns]
    else:
        raise TypeError(f"Expected DataFrame or list[dict], got {type(turns).__name__}")

    # Resolve labels per turn
    turn_labels: list[Any] = []
    if labels is None:
        turn_labels = [None] * len(records)
    elif isinstance(labels, str):
        turn_labels = [r.get(labels) for r in records]
    elif isinstance(labels, dict):
        for idx, r in enumerate(records):
            t_id = r.get(turn_id_col, idx)
            # Try turn_id key first, fallback to positional index
            val = labels.get(t_id, labels.get(idx))
            turn_labels.append(val)
    elif isinstance(labels, (list, tuple, pd.Series)):
        turn_labels = list(labels)
        if len(turn_labels) < len(records):
            turn_labels.extend([None] * (len(records) - len(turn_labels)))
    else:
        turn_labels = [None] * len(records)

    final_html, full_plain_text = render_conversation_html(
        records=records,
        turn_labels=turn_labels,
        title=title,
        metadata=metadata,
        status=status,
        status_color=status_color,
        speaker_col=speaker_col,
        text_col=text_col,
        turn_id_col=turn_id_col,
        action_name_col=action_name_col,
        action_values_col=action_values_col,
        max_height=max_height,
    )

    return ConversationView(
        html_content=final_html,
        plain_text=full_plain_text,
        turns=records,
        metadata=metadata,
    )
