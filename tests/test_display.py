"""
Unit tests for generic conversation display and ABCD viewer integration.
"""

import pandas as pd

import abcd_helper as abcd
from utils.display import ConversationView, render_conversation


def test_render_generic_dataframe():
    """Verify generic conversation rendering from a custom DataFrame."""
    df_sample = pd.DataFrame(
        [
            {"speaker": "agent", "text": "Hello, how can I help you?", "turn_id": 0},
            {"speaker": "customer", "text": "I want to track my order.", "turn_id": 1},
            {
                "speaker": "action",
                "text": "Executed lookup",
                "action_name": "lookup_order",
                "action_values": "order=123",
                "turn_id": 2,
            },
        ]
    )

    view = render_conversation(
        turns=df_sample,
        title="Custom Chat Log",
        metadata={"Channel": "Web Chat", "Priority": "High"},
        status="Resolved",
    )

    assert isinstance(view, ConversationView)
    html_out = view._repr_html_()
    assert "Custom Chat Log" in html_out
    assert "Channel" in html_out
    assert "Web Chat" in html_out
    assert "Resolved" in html_out
    assert "Hello, how can I help you?" in html_out
    assert "I want to track my order." in html_out
    assert "lookup_order" in html_out


def test_render_with_labels():
    """Verify attaching label badges (dict, list, and column name)."""
    df_sample = pd.DataFrame(
        [
            {
                "speaker": "agent",
                "text": "What is your email?",
                "turn_id": 0,
                "slots": [],
            },
            {
                "speaker": "customer",
                "text": "test@example.com",
                "turn_id": 1,
                "slots": ["email"],
            },
        ]
    )

    # 1. Labels from column name
    view_col = render_conversation(df_sample, labels="slots")
    html_col = view_col._repr_html_()
    assert "🏷️ email" in html_col

    # 2. Labels from dict mapping with string values
    slot_map = {1: {"email": "test@example.com", "verified": True}}
    view_dict = render_conversation(df_sample, labels=slot_map)
    html_dict = view_dict._repr_html_()
    assert "🏷️ email:" in html_dict
    assert "test@example.com" in html_dict
    assert "🏷️ verified" in html_dict


def test_abcd_conversation_render_integration():
    """Verify that abcd_helper.Conversation seamlessly delegates to generic viewer."""
    convo = abcd.get_conversation(3592)

    # Default HTML representation
    html_default = convo._repr_html_()
    assert "Conversation #3592" in html_default
    assert "product_defect" in html_default

    # Custom render with slot labels
    labels = {4: ["customer_name"], 10: ["email", "order_id"]}
    view_labeled = convo.render(labels=labels)
    assert isinstance(view_labeled, ConversationView)
    html_labeled = view_labeled._repr_html_()
    assert "🏷️ customer_name" in html_labeled
    assert "🏷️ email" in html_labeled
    assert "🏷️ order_id" in html_labeled
