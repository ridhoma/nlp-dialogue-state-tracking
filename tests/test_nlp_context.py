import pytest

import abcd_helper as abcd
from nlp.context import (
    DialogueTurnContext,
    extract_customer_contexts,
    get_customer_turn_context,
)


def test_get_customer_turn_context_basic():
    # Synthetic mini conversation
    turns = [
        {"turn_id": 0, "speaker": "agent", "text": "Hello!"},
        {"turn_id": 1, "speaker": "agent", "text": "How can I help you today?"},
        {
            "turn_id": 2,
            "speaker": "customer",
            "text": "I want to return something.",
        },
        {"turn_id": 3, "speaker": "agent", "text": "May I have your name?"},
        {"turn_id": 4, "speaker": "customer", "text": "Alice Walker"},
    ]

    # Target turn 4: should roll back to collect 2 agent utterances (turns 3 and 1)
    ctx = get_customer_turn_context(
        turns, current_turn_idx=4, min_prior_agent_utterances=2
    )
    assert isinstance(ctx, DialogueTurnContext)
    assert ctx.turn_id == 4
    assert ctx.customer_text == "Alice Walker"
    assert ctx.num_prior_agent_utterances == 2
    assert [t["turn_id"] for t in ctx.prior_agent_turns] == [1, 3]

    # Turns collected in chronological order: turn 1, turn 2, turn 3
    assert [t["turn_id"] for t in ctx.prior_turns] == [1, 2, 3]
    assert "[AGENT]: How can I help you today?" in ctx.formatted_context
    assert "[CUSTOMER]: I want to return something." in ctx.formatted_context
    assert "[AGENT]: May I have your name?" in ctx.formatted_context
    assert "[CUSTOMER (CURRENT)]: Alice Walker" in ctx.full_document


def test_get_customer_turn_context_early_turns():
    # Customer turn at turn 0 or 1 with fewer than 2 prior agent utterances
    turns = [
        {"turn_id": 0, "speaker": "agent", "text": "Welcome to store."},
        {"turn_id": 1, "speaker": "customer", "text": "I have an issue."},
    ]

    ctx = get_customer_turn_context(
        turns, current_turn_idx=1, min_prior_agent_utterances=2
    )
    assert ctx.num_prior_agent_utterances == 1
    assert len(ctx.prior_turns) == 1
    assert ctx.prior_turns[0]["turn_id"] == 0


def test_consecutive_customer_bursts():
    # Agent asks for multiple things, customer sends 3 consecutive messages
    turns = [
        {"turn_id": 0, "speaker": "agent", "text": "Hello."},
        {
            "turn_id": 1,
            "speaker": "agent",
            "text": "Please provide username and email.",
        },
        {"turn_id": 2, "speaker": "customer", "text": "user_123"},
        {"turn_id": 3, "speaker": "customer", "text": "user@example.com"},
    ]

    ctx_turn3 = get_customer_turn_context(
        turns, current_turn_idx=3, min_prior_agent_utterances=2
    )
    # Rolls back through turn 2 (cust), turn 1 (agent #1), turn 0 (agent #2)
    assert ctx_turn3.num_prior_agent_utterances == 2
    assert [t["turn_id"] for t in ctx_turn3.prior_turns] == [0, 1, 2]
    assert "[AGENT]: Please provide username and email." in ctx_turn3.formatted_context
    assert "[CUSTOMER]: user_123" in ctx_turn3.formatted_context
    assert "[CUSTOMER (CURRENT)]: user@example.com" in ctx_turn3.full_document


def test_action_turns_skipped_by_default():
    turns = [
        {"turn_id": 0, "speaker": "agent", "text": "Please give your name."},
        {"turn_id": 1, "speaker": "customer", "text": "Bob."},
        {"turn_id": 2, "speaker": "agent", "text": "What is the reason?"},
        {"turn_id": 3, "speaker": "action", "text": "Account pulled up."},
        {"turn_id": 4, "speaker": "customer", "text": "Defective zipper."},
    ]

    ctx_no_action = get_customer_turn_context(
        turns, current_turn_idx=4, min_prior_agent_utterances=2, include_actions=False
    )
    assert all(t["speaker"] != "action" for t in ctx_no_action.prior_turns)

    ctx_with_action = get_customer_turn_context(
        turns, current_turn_idx=4, min_prior_agent_utterances=2, include_actions=True
    )
    assert any(t["speaker"] == "action" for t in ctx_with_action.prior_turns)


def test_extract_customer_contexts_dataframe():
    df_turns = abcd.load_turns(convo_id=3592)
    contexts = extract_customer_contexts(df_turns, min_prior_agent_utterances=2)
    assert len(contexts) == 13

    # All returned contexts must belong to customer turns
    cust_turn_ids = df_turns[df_turns["speaker"] == "customer"]["turn_id"].tolist()
    assert [c.turn_id for c in contexts] == cust_turn_ids

    # For turn 11 (Order ID), it has captured the prompt
    ctx_11 = next(c for c in contexts if c.turn_id == 11)
    assert "3348917502" in ctx_11.customer_text
    assert ctx_11.num_prior_agent_utterances >= 2
    assert "order ID" in ctx_11.formatted_context


def test_error_handling():
    turns = [{"turn_id": 0, "speaker": "agent", "text": "Hi"}]
    with pytest.raises(IndexError):
        get_customer_turn_context(turns, current_turn_idx=99)

    with pytest.raises(ValueError, match="empty turns"):
        get_customer_turn_context([], current_turn_idx=0)


def test_extract_customer_contexts_multi_conversation():
    # Synthetic two-conversation dataset
    turns = [
        # Convo 1
        {"convo_id": 101, "turn_id": 0, "speaker": "agent", "text": "Convo1 Agent 0"},
        {"convo_id": 101, "turn_id": 1, "speaker": "agent", "text": "Convo1 Agent 1"},
        {"convo_id": 101, "turn_id": 2, "speaker": "customer", "text": "Convo1 Cust 2"},
        # Convo 2
        {"convo_id": 102, "turn_id": 0, "speaker": "agent", "text": "Convo2 Agent 0"},
        {"convo_id": 102, "turn_id": 1, "speaker": "customer", "text": "Convo2 Cust 1"},
        {"convo_id": 102, "turn_id": 2, "speaker": "agent", "text": "Convo2 Agent 2"},
        {"convo_id": 102, "turn_id": 3, "speaker": "customer", "text": "Convo2 Cust 3"},
    ]

    # Test flat extraction
    flat = extract_customer_contexts(
        turns, min_prior_agent_utterances=2, group_by_convo=False
    )
    assert len(flat) == 3
    assert [c.convo_id for c in flat] == [101, 102, 102]

    # Test boundary isolation: Convo 102 Cust 1 must NEVER see Convo 101!
    ctx_102_1 = flat[1]
    assert ctx_102_1.convo_id == 102
    assert ctx_102_1.customer_text == "Convo2 Cust 1"
    assert ctx_102_1.num_prior_agent_utterances == 1
    assert len(ctx_102_1.prior_turns) == 1
    assert ctx_102_1.prior_turns[0]["text"] == "Convo2 Agent 0"
    # Must NOT have any turns from convo 101
    assert "Convo1" not in ctx_102_1.full_document

    # Test group_by_convo=True
    grouped = extract_customer_contexts(
        turns, min_prior_agent_utterances=2, group_by_convo=True
    )
    assert set(grouped.keys()) == {101, 102}
    assert len(grouped[101]) == 1
    assert len(grouped[102]) == 2


def test_get_customer_turn_context_convo_filter():
    turns = [
        {"convo_id": 1, "turn_id": 0, "speaker": "agent", "text": "A1"},
        {"convo_id": 1, "turn_id": 1, "speaker": "customer", "text": "C1"},
        {"convo_id": 2, "turn_id": 0, "speaker": "agent", "text": "A2"},
        {"convo_id": 2, "turn_id": 1, "speaker": "customer", "text": "C2"},
    ]
    ctx = get_customer_turn_context(turns, current_turn_idx=1, convo_id=2)
    assert ctx.convo_id == 2
    assert ctx.customer_text == "C2"
    assert ctx.prior_turns[0]["text"] == "A2"


def test_get_customer_turn_context_non_customer_warning():
    turns = [
        {"convo_id": 1, "turn_id": 0, "speaker": "agent", "text": "Agent greeting"},
        {"convo_id": 1, "turn_id": 1, "speaker": "customer", "text": "Customer reply"},
    ]
    with pytest.warns(UserWarning, match="has speaker 'agent', expected 'customer'"):
        ctx = get_customer_turn_context(turns, current_turn_idx=0, convo_id=1)
    assert ctx is None
