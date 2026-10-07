"""
Generic Interactive Conversation Viewer for Jupyter Notebooks and CLI.

Provides dataset-agnostic HTML chat visualization with speaker bubbles,
action badges, custom metadata headers, and turn-level annotation chips
(e.g., active dialogue state tracking slots, intents, or sentiment labels).
"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd


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


def _is_empty_label(val: Any) -> bool:
    """Check if label value is None, empty collection, or NaN."""
    if val is None:
        return True
    if isinstance(val, (str, bytes)):
        return len(val.strip()) == 0
    if hasattr(val, "__len__"):
        return len(val) == 0
    if hasattr(val, "size"):
        return val.size == 0
    return bool(isinstance(val, float) and pd.isna(val))


def _format_label_badges(label_val: Any) -> str:
    """Formats turn-level labels into styled HTML chips."""
    if _is_empty_label(label_val):
        return ""

    chips: list[str] = []

    # Case 1: Dictionary mapping (e.g., {'email': 'cminh@email.com', 'username': True})
    if isinstance(label_val, dict):
        for k, v in label_val.items():
            k_esc = html.escape(str(k))
            if v is True:
                chips.append(
                    f'<span style="background-color: var(--convo-chip-bg, rgba(99, 102, 241, 0.15)); '
                    f"color: var(--convo-chip-text, #4f46e5); padding: 2px 7px; "
                    f"border-radius: 9999px; font-size: 11px; font-weight: 600; margin-right: 4px; "
                    f'display: inline-block;">🏷️ {k_esc}</span>'
                )
            else:
                v_esc = html.escape(str(v))
                chips.append(
                    f'<span style="background-color: var(--convo-chip-kv-bg, rgba(59, 130, 246, 0.15)); '
                    f"color: var(--convo-chip-kv-text, #2563eb); padding: 2px 7px; "
                    f"border-radius: 9999px; font-size: 11px; font-weight: 600; margin-right: 4px; "
                    f'display: inline-block;">🏷️ {k_esc}: <span style="font-family: monospace;">{v_esc}</span></span>'
                )

    # Case 2: Iterable of labels (list, tuple, set, numpy array)
    elif (
        isinstance(label_val, (list, tuple, set))
        or hasattr(label_val, "__iter__")
        and not isinstance(label_val, (str, bytes))
    ):
        for item in label_val:
            item_esc = html.escape(str(item))
            chips.append(
                f'<span style="background-color: var(--convo-chip-bg, rgba(99, 102, 241, 0.15)); '
                f"color: var(--convo-chip-text, #4f46e5); padding: 2px 7px; "
                f"border-radius: 9999px; font-size: 11px; font-weight: 600; margin-right: 4px; "
                f'display: inline-block;">🏷️ {item_esc}</span>'
            )

    # Case 3: Single scalar label / string
    else:
        text_esc = html.escape(str(label_val).strip())
        if text_esc:
            chips.append(
                f'<span style="background-color: var(--convo-chip-bg, rgba(99, 102, 241, 0.15)); '
                f"color: var(--convo-chip-text, #4f46e5); padding: 2px 7px; "
                f"border-radius: 9999px; font-size: 11px; font-weight: 600; margin-right: 4px; "
                f'display: inline-block;">🏷️ {text_esc}</span>'
            )

    if not chips:
        return ""

    return (
        '<div style="margin-top: 6px; padding-top: 4px; border-top: 1px dashed var(--convo-divider, rgba(128,128,128,0.2)); '
        'display: flex; flex-wrap: wrap; gap: 4px;">' + "".join(chips) + "</div>"
    )


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
    Renders any conversation sequence into an interactive HTML visualization.

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

    # Status badge styling
    status_badge_html = ""
    if status:
        bg = "rgba(148, 163, 184, 0.2)"
        fg = "var(--convo-meta-text, #334155)"
        st_lower = status.lower()
        if status_color:
            if status_color == "success":
                bg, fg = "rgba(16, 185, 129, 0.2)", "#10b981"
            elif status_color in ("danger", "error"):
                bg, fg = "rgba(239, 68, 68, 0.2)", "#ef4444"
            elif status_color == "warning":
                bg, fg = "rgba(245, 158, 11, 0.2)", "#f59e0b"
            else:
                bg = status_color
                fg = "#ffffff"
        elif "resolve" in st_lower or "success" in st_lower:
            bg, fg = "rgba(16, 185, 129, 0.2)", "#10b981"
        elif "fail" in st_lower or "unresolved" in st_lower or "error" in st_lower:
            bg, fg = "rgba(239, 68, 68, 0.2)", "#ef4444"
        elif "escalat" in st_lower:
            bg, fg = "rgba(249, 115, 22, 0.2)", "#f97316"

        status_badge_html = (
            f'<span style="background-color: {bg}; color: {fg}; padding: 3px 10px; '
            f'border-radius: 9999px; font-weight: 700; font-size: 12px; margin-left: 8px;">'
            f"{html.escape(str(status))}</span>"
        )

    # Subtitle / Metadata banner
    meta_html = ""
    if metadata:
        meta_items = [
            f"<b>{html.escape(str(k))}:</b> {html.escape(str(v))}"
            for k, v in metadata.items()
            if v is not None
        ]
        if meta_items:
            meta_html = (
                '<div style="background: var(--convo-meta-bg, #f8fafc); padding: 8px 16px; border-bottom: 1px solid var(--convo-border, #e2e8f0); '
                'font-size: 12px; color: var(--convo-meta-text, #475569); display: flex; flex-wrap: wrap; gap: 12px;">'
                + " | ".join(meta_items)
                + "</div>"
            )

    # Render turn bubbles
    bubbles_html: list[str] = []
    text_lines: list[str] = []

    if title:
        text_lines.append(f"=== {title} ===")
    if metadata:
        meta_str = " | ".join(f"{k}: {v}" for k, v in metadata.items())
        text_lines.append(meta_str)
    text_lines.append("-" * 60)

    for idx, (row, lbl) in enumerate(zip(records, turn_labels)):
        spk = str(row.get(speaker_col, "unknown")).lower()
        raw_text = str(row.get(text_col, ""))
        text_esc = html.escape(raw_text)
        t_id = row.get(turn_id_col, idx)
        label_badges_html = _format_label_badges(lbl)

        # Plain text log representation
        lbl_str = f" [TAGS: {lbl}]" if not _is_empty_label(lbl) else ""

        # Speaker categorization
        if spk in ("action", "system"):
            act_name = html.escape(str(row.get(action_name_col) or ""))
            act_vals = html.escape(str(row.get(action_values_col) or ""))
            args_div = (
                f'<div style="font-family: monospace; font-size: 11px; color: var(--convo-act-subtext, #92400e); margin-top: 2px;">args: {act_vals}</div>'
                if act_vals and act_vals != "None"
                else ""
            )
            act_label = (
                f"⚙️ Action [{t_id}]: {act_name}" if act_name else f"⚙️ System [{t_id}]"
            )

            bubbles_html.append(f"""
            <div style="text-align: center; margin: 10px 0;">
                <div style="display: inline-block; background: var(--convo-act-bg, #fef3c7); border: 1px solid var(--convo-act-border, #fde68a); border-radius: 12px; padding: 6px 14px; font-size: 12px; color: var(--convo-act-text, #92400e); max-width: 85%;">
                    <div style="font-weight: 700; text-transform: uppercase; color: var(--convo-act-header, #b45309);">{act_label}</div>
                    {args_div}
                    <div style="font-style: italic; color: var(--convo-act-text, #451a03); margin-top: 2px;">"{text_esc}"</div>
                    {label_badges_html}
                </div>
            </div>
            """)
            text_lines.append(
                f"[{t_id:02d}] ⚙️ ACTION: {act_name} -> {raw_text}{lbl_str}"
            )

        elif spk in ("customer", "user"):
            bubbles_html.append(f"""
            <div style="display: flex; justify-content: flex-start; margin: 6px 0;">
                <div style="max-width: 78%; background: var(--convo-cust-bg, #e0f2fe); border: 1px solid var(--convo-cust-border, #bae6fd); border-radius: 12px 12px 12px 2px; padding: 8px 12px; color: var(--convo-cust-text, #0369a1);">
                    <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; color: var(--convo-cust-header, #0284c7); margin-bottom: 2px;">
                        👤 Customer (Turn {t_id})
                    </div>
                    <div style="font-size: 13px; line-height: 1.4; color: var(--convo-cust-text, #0c4a6e); word-break: break-word;">
                        {text_esc}
                    </div>
                    {label_badges_html}
                </div>
            </div>
            """)
            text_lines.append(f"[{t_id:02d}] 👤 CUSTOMER: {raw_text}{lbl_str}")

        elif spk in ("agent", "assistant", "bot"):
            bubbles_html.append(f"""
            <div style="display: flex; justify-content: flex-end; margin: 6px 0;">
                <div style="max-width: 78%; background: var(--convo-agent-bg, #f1f5f9); border: 1px solid var(--convo-agent-border, #e2e8f0); border-radius: 12px 12px 2px 12px; padding: 8px 12px; color: var(--convo-agent-text, #334155); text-align: left;">
                    <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; color: var(--convo-agent-header, #64748b); margin-bottom: 2px; text-align: right;">
                        🧑‍💼 Agent (Turn {t_id})
                    </div>
                    <div style="font-size: 13px; line-height: 1.4; color: var(--convo-agent-text, #1e293b); word-break: break-word;">
                        {text_esc}
                    </div>
                    {label_badges_html}
                </div>
            </div>
            """)
            text_lines.append(f"[{t_id:02d}] 🧑‍💼 AGENT: {raw_text}{lbl_str}")

        else:
            # Generic neutral bubble
            bubbles_html.append(f"""
            <div style="display: flex; justify-content: flex-start; margin: 6px 0;">
                <div style="max-width: 78%; background: var(--convo-neutral-bg, #f3f4f6); border: 1px solid var(--convo-neutral-border, #e5e7eb); border-radius: 12px; padding: 8px 12px; color: var(--convo-neutral-text, #374151);">
                    <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; color: var(--convo-neutral-header, #6b7280); margin-bottom: 2px;">
                        {html.escape(spk.upper())} (Turn {t_id})
                    </div>
                    <div style="font-size: 13px; line-height: 1.4; color: var(--convo-neutral-text, #1f2937); word-break: break-word;">
                        {text_esc}
                    </div>
                    {label_badges_html}
                </div>
            </div>
            """)
            text_lines.append(f"[{t_id:02d}] {spk.upper()}: {raw_text}{lbl_str}")

    stream_html = "\n".join(bubbles_html)
    text_lines.append("-" * 60)
    full_plain_text = "\n".join(text_lines)

    title_display = html.escape(title or "Conversation")
    header_html = f"""
    <div style="background: var(--convo-header-bg, #1e293b); color: var(--convo-header-text, #f8fafc); padding: 12px 16px; display: flex; justify-content: space-between; align-items: center;">
        <div>
            <span style="font-weight: 700; font-size: 15px;">{title_display}</span>
            <span style="background: var(--convo-header-badge-bg, #334155); color: var(--convo-header-badge-text, #94a3b8); font-size: 11px; padding: 2px 6px; border-radius: 4px; margin-left: 6px;">{len(records)} turns</span>
        </div>
        <div>
            {status_badge_html}
        </div>
    </div>
    """

    final_html = f"""
    <div class="convo-view-container" style="
        --convo-bg: var(--jp-layout-color1, var(--vscode-editor-background, #ffffff));
        --convo-header-bg: #1e293b;
        --convo-header-text: #f8fafc;
        --convo-header-badge-bg: #334155;
        --convo-header-badge-text: #94a3b8;
        --convo-meta-bg: var(--jp-layout-color2, rgba(148, 163, 184, 0.08));
        --convo-meta-text: var(--jp-content-font-color2, #475569);
        --convo-border: var(--jp-border-color1, rgba(148, 163, 184, 0.25));
        --convo-scroll-bg: var(--jp-layout-color1, var(--vscode-editor-background, #ffffff));
        --convo-cust-bg: rgba(2, 132, 199, 0.08);
        --convo-cust-border: rgba(2, 132, 199, 0.25);
        --convo-cust-header: #0284c7;
        --convo-cust-text: var(--jp-content-font-color1, #0c4a6e);
        --convo-agent-bg: rgba(148, 163, 184, 0.12);
        --convo-agent-border: rgba(148, 163, 184, 0.25);
        --convo-agent-header: #64748b;
        --convo-agent-text: var(--jp-content-font-color1, #1e293b);
        --convo-act-bg: rgba(245, 158, 11, 0.12);
        --convo-act-border: rgba(245, 158, 11, 0.3);
        --convo-act-header: #d97706;
        --convo-act-text: var(--jp-content-font-color1, #78350f);
        --convo-act-subtext: #92400e;
        --convo-neutral-bg: rgba(148, 163, 184, 0.1);
        --convo-neutral-border: rgba(148, 163, 184, 0.22);
        --convo-neutral-header: #64748b;
        --convo-neutral-text: var(--jp-content-font-color1, #334155);
        --convo-chip-bg: rgba(99, 102, 241, 0.15);
        --convo-chip-text: #4f46e5;
        --convo-chip-kv-bg: rgba(59, 130, 246, 0.15);
        --convo-chip-kv-text: #2563eb;
        --convo-divider: rgba(128, 128, 128, 0.2);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        border: 1px solid var(--convo-border);
        border-radius: 10px;
        overflow: hidden;
        max-width: 820px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08);
        margin: 12px 0;
        background: var(--convo-bg);
    ">
        <style>
            @media (prefers-color-scheme: dark) {{
                .convo-view-container {{
                    --convo-bg: #1e1e2e !important;
                    --convo-header-bg: #11111b !important;
                    --convo-header-text: #cdd6f4 !important;
                    --convo-header-badge-bg: #1e1e2e !important;
                    --convo-header-badge-text: #a6adc8 !important;
                    --convo-meta-bg: rgba(255, 255, 255, 0.04) !important;
                    --convo-meta-text: #94a3b8 !important;
                    --convo-border: rgba(255, 255, 255, 0.12) !important;
                    --convo-scroll-bg: #181825 !important;
                    --convo-cust-bg: rgba(14, 165, 233, 0.15) !important;
                    --convo-cust-border: rgba(14, 165, 233, 0.35) !important;
                    --convo-cust-header: #38bdf8 !important;
                    --convo-cust-text: #f0f9ff !important;
                    --convo-agent-bg: rgba(255, 255, 255, 0.06) !important;
                    --convo-agent-border: rgba(255, 255, 255, 0.14) !important;
                    --convo-agent-header: #94a3b8 !important;
                    --convo-agent-text: #f1f5f9 !important;
                    --convo-act-bg: rgba(245, 158, 11, 0.15) !important;
                    --convo-act-border: rgba(245, 158, 11, 0.35) !important;
                    --convo-act-header: #fbbf24 !important;
                    --convo-act-text: #fef3c7 !important;
                    --convo-act-subtext: #fcd34d !important;
                    --convo-neutral-bg: rgba(255, 255, 255, 0.05) !important;
                    --convo-neutral-border: rgba(255, 255, 255, 0.12) !important;
                    --convo-neutral-header: #a6adc8 !important;
                    --convo-neutral-text: #e2e8f0 !important;
                    --convo-chip-bg: rgba(129, 140, 248, 0.22) !important;
                    --convo-chip-text: #c7d2fe !important;
                    --convo-chip-kv-bg: rgba(96, 165, 250, 0.22) !important;
                    --convo-chip-kv-text: #bfdbfe !important;
                    --convo-divider: rgba(255, 255, 255, 0.12) !important;
                }}
            }}
        </style>
        {header_html}
        {meta_html}
        <div style="padding: 16px; max-height: {max_height}; overflow-y: auto; background: var(--convo-scroll-bg);">
            {stream_html}
        </div>
    </div>
    """

    return ConversationView(
        html_content=final_html,
        plain_text=full_plain_text,
        turns=records,
        metadata=metadata,
    )
