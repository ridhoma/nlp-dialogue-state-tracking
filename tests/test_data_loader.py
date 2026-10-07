import pandas as pd

import abcd_helper as abcd
from abcd_helper.resolution import evaluate_conversation_resolution


def test_load_conversations():
    df = abcd.load_conversations()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10042
    assert "convo_id" in df.columns
    assert "intent" in df.columns
    assert "resolved" in df.columns
    assert set(df["split"].unique()) == {"train", "dev", "test"}


def test_load_conversations_filter():
    df_train = abcd.load_conversations(split="train")
    assert len(df_train) == 8034
    assert (df_train["split"] == "train").all()

    df_res = abcd.load_conversations(resolved=1)
    assert (df_res["resolved"] == 1).all()


def test_load_turns():
    df_turns = abcd.load_turns(convo_id=3592)
    assert len(df_turns) == 29
    assert set(df_turns["speaker"].unique()).issubset({"agent", "customer", "action"})
    assert "text" in df_turns.columns
    assert "text_delexed" in df_turns.columns


def test_subflows_and_actions():
    df_sf = abcd.load_subflows()
    assert len(df_sf) == 55
    assert "canonical_actions" in df_sf.columns

    df_act = abcd.load_actions()
    assert len(df_act) == 30
    assert "required_slots" in df_act.columns


def test_get_conversation():
    c = abcd.get_conversation(3592)
    assert c.convo_id == 3592
    assert c.intent == "return_size"
    assert c.resolved == 0
    assert c.has_escalation is True
    assert "pull-up-account" in c.actions
    html = c._repr_html_()
    assert "Conversation #3592" in html
    assert "Unresolved" in html


def test_evaluate_conversation_resolution():
    canonical = ["pull-up-account", "validate-purchase", "offer-refund"]

    # Clean positive resolution
    res1, diag1 = evaluate_conversation_resolution(
        actions_taken=["pull-up-account", "validate-purchase", "offer-refund"],
        canonical_actions=canonical,
    )
    assert res1 == 1
    assert diag1["reached_canonical_terminal"] is True
    assert diag1["has_escalation"] is False

    # Escalated resolution
    res2, diag2 = evaluate_conversation_resolution(
        actions_taken=["pull-up-account", "notify-team"],
        canonical_actions=canonical,
    )
    assert res2 == 0
    assert diag2["has_escalation"] is True

    # Premature termination
    res3, diag3 = evaluate_conversation_resolution(
        actions_taken=["pull-up-account"],
        canonical_actions=canonical,
    )
    assert res3 == 0
    assert diag3["reached_canonical_terminal"] is False


def test_load_intent_slots_full():
    df = abcd.load_intent_slots()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 55
    expected_cols = {
        "flow",
        "intent",
        "canonical_actions",
        "mandatory_slots",
        "num_mandatory_slots",
        "agent_action_slots",
        "all_slots",
        "sop_summary",
    }
    assert expected_cols.issubset(df.columns)
    assert df["flow"].nunique() == 10


def test_load_intent_slots_filter():
    # Single subflow lookup returning dict
    schema = abcd.load_intent_slots(intent="refund_initiate")
    assert isinstance(schema, dict)
    assert schema["flow"] == "product_defect"
    assert schema["intent"] == "refund_initiate"
    assert schema["num_mandatory_slots"] == 5
    assert len(schema["mandatory_slots"]) == 5

    # Filter with as_df=True
    df_single = abcd.load_intent_slots(intent="refund_initiate", as_df=True)
    assert isinstance(df_single, pd.DataFrame)
    assert len(df_single) == 1

    # Filter by flow returning multiple subflows
    df_flow = abcd.load_intent_slots(flow="single_item_query")
    assert isinstance(df_flow, pd.DataFrame)
    assert len(df_flow) == 4

    # Compound key lookup
    schema_compound = abcd.load_intent_slots(
        intent="membership", flow="storewide_query"
    )
    assert isinstance(schema_compound, dict)
    assert schema_compound["flow"] == "storewide_query"


def test_intent_slot_cnf_satisfaction():
    schema = abcd.load_intent_slots(intent="refund_initiate")
    mandatory = schema["mandatory_slots"]

    # Customer provides account_id (alternative to customer_name), username, and order_id
    extracted = {"account_id": "acc_123", "username": "alice", "order_id": "ord_999"}
    satisfied = [any(s in extracted for s in group) for group in mandatory]

    # Group 0: ['customer_name', 'account_id'] -> True (matched account_id)
    # Group 1: ['username'] -> True
    # Group 2: ['email'] -> False
    # Group 3: ['order_id'] -> True
    # Group 4: ['full_address', 'street_address'] -> False
    assert satisfied == [True, True, False, True, False]
    density = sum(satisfied) / len(mandatory)
    assert round(density, 2) == 0.60


def test_intent_schema_ontology_validation(monkeypatch):
    import pytest

    from abcd_helper.intent_schema import INTENT_SCHEMA_DATA, build_intent_schema

    # Normal build passes cleanly
    df = build_intent_schema()
    assert len(df) == 55

    # Tampered slot raises ValueError
    bad_entry = dict(INTENT_SCHEMA_DATA[0])
    bad_entry["mandatory_slots"] = [["non_existent_slot_xyz"]]
    tampered_data = [bad_entry] + INTENT_SCHEMA_DATA[1:]

    monkeypatch.setattr("abcd_helper.intent_schema.INTENT_SCHEMA_DATA", tampered_data)
    with pytest.raises(ValueError, match="non_existent_slot_xyz"):
        build_intent_schema()

    # Tampered action raises ValueError
    bad_action_entry = dict(INTENT_SCHEMA_DATA[0])
    bad_action_entry["canonical_actions"] = ["non_existent_action_xyz"]
    tampered_action_data = [bad_action_entry] + INTENT_SCHEMA_DATA[1:]

    monkeypatch.setattr(
        "abcd_helper.intent_schema.INTENT_SCHEMA_DATA", tampered_action_data
    )
    with pytest.raises(ValueError, match="non_existent_action_xyz"):
        build_intent_schema()
