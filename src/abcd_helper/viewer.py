"""
Interactive Conversation Viewer for Jupyter Notebooks and CLI.
Provides pretty HTML rendering for ABCD conversations with speaker chat bubbles,
action badges, scenario metadata, and resolution status, delegating to the generic
viewer in `utils.display`.
"""

from typing import Any

import pandas as pd

from utils.display import ConversationView, render_conversation


class Conversation:
    """
    Domain-specific representation of a single ABCD conversation session.
    Provides interactive HTML rendering for Jupyter notebooks via _repr_html_().
    """

    def __init__(self, convo_meta: dict[str, Any], turns_df: pd.DataFrame):
        self.convo_id = convo_meta.get("convo_id")
        self.split = convo_meta.get("split")
        self.flow = convo_meta.get("flow")
        self.intent = convo_meta.get("intent")
        self.resolved = convo_meta.get("resolved")
        self.has_escalation = convo_meta.get("has_escalation")
        self.meta = convo_meta
        self.turns = turns_df.sort_values("turn_id").reset_index(drop=True)

    @property
    def actions(self) -> list[str]:
        """List of system actions executed in this conversation."""
        acts = (
            self.turns[self.turns["speaker"] == "action"]["action_name"]
            .dropna()
            .tolist()
        )
        return acts

    def to_dataframe(self) -> pd.DataFrame:
        """Return the turn-level DataFrame for this conversation."""
        return self.turns.copy()

    def render(
        self,
        labels: str | pd.Series | dict[Any, Any] | list[Any] | None = None,
        max_height: str = "480px",
    ) -> ConversationView:
        """
        Renders this ABCD conversation with optional turn labels (e.g., active slots).

        Parameters
        ----------
        labels : str | pd.Series | dict | list, optional
            Turn-level labels or slots to display (e.g. inferred_slots mapping).
        max_height : str, default='480px'
            Maximum height for the scrollable dialogue container.

        Returns
        -------
        ConversationView
            Renderable view that displays interactive HTML in Jupyter cells.
        """
        is_res = self.resolved == 1
        res_label = "Resolved (Y=1)" if is_res else "Unresolved (Y=0)"
        status_color = "success" if is_res else "danger"

        metadata = {
            "Flow": self.flow,
            "Intent": self.intent,
            "Customer": self.meta.get("customer_name") or "N/A",
            "Order ID": self.meta.get("order_id") or "N/A",
            "Escalated": "Yes" if self.has_escalation else None,
        }

        title = f"Conversation #{self.convo_id} ({self.split})"

        return render_conversation(
            turns=self.turns,
            title=title,
            metadata=metadata,
            labels=labels,
            status=res_label,
            status_color=status_color,
            max_height=max_height,
        )

    def print_transcript(self, delexed: bool = False) -> None:
        """Print plain text transcript to stdout."""
        status = "RESOLVED (Y=1)" if self.resolved == 1 else "UNRESOLVED (Y=0)"
        esc = " [ESCALATED]" if self.has_escalation else ""
        print("============================================================")
        print(
            f"Conversation ID: {self.convo_id} | Intent: {self.intent} | Status: {status}{esc}"
        )
        print(
            f"Customer: {self.meta.get('customer_name')} | Order ID: {self.meta.get('order_id')}"
        )
        print("============================================================")

        for _, row in self.turns.iterrows():
            t_idx = row["turn_id"]
            text = row["text_delexed"] if delexed else row["text"]

            if row["speaker"] == "action":
                act_name = row["action_name"]
                vals = row["action_values"]
                print(f'[{t_idx:02d}] ⚙️  ACTION: {act_name} (args: {vals}) -> "{text}"')
            elif row["speaker"] == "agent":
                print(f"[{t_idx:02d}] 🧑‍💼 AGENT: {text}")
            else:
                print(f"[{t_idx:02d}] 👤 CUSTOMER: {text}")
        print("============================================================\n")

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter Notebooks."""
        return self.render()._repr_html_()
