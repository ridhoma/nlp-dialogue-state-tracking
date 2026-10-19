"""
Unit tests for data_science.features (DST and AST feature engineering).
"""

import numpy as np
import pandas as pd
import pytest

from data_science.features.ast import (
    compute_action_thrashing,
    extract_ast_features_for_conversation,
    token_levenshtein,
)
from data_science.features.dst import (
    evaluate_cnf_clauses,
    extract_dst_features_for_conversation,
)


def test_token_levenshtein():
    # Identical
    assert token_levenshtein(["a", "b", "c"], ["a", "b", "c"]) == 0
    # Empty
    assert token_levenshtein([], []) == 0
    assert token_levenshtein(["a"], []) == 1
    assert token_levenshtein([], ["a", "b"]) == 2
    # Deletion, insertion, substitution
    assert token_levenshtein(["a", "b"], ["a"]) == 1
    assert token_levenshtein(["a"], ["a", "b"]) == 1
    assert token_levenshtein(["a", "b"], ["a", "c"]) == 1
    # Complex sequence
    obs = ["pull-up-account", "validate-purchase", "enter-details", "notify-team"]
    canon = ["pull-up-account", "validate-purchase", "membership", "enter-details", "update-order"]
    # Deletion of membership (1), substitution/deletion of notify-team + update-order
    dist = token_levenshtein(obs, canon)
    assert dist == 2


def test_compute_action_thrashing():
    assert compute_action_thrashing([]) == 0
    assert compute_action_thrashing(["a"]) == 0
    assert compute_action_thrashing(["a", "b", "c"]) == 0
    assert compute_action_thrashing(["a", "a", "b", "b", "b"]) == 3  # 1 from a->a, 2 from b->b->b


def test_evaluate_cnf_clauses():
    cnf = [
        ["customer_name", "account_id"],
        ["order_id"],
        ["email"],
    ]
    # Clause 0 satisfied by account_id, Clause 1 by order_id, Clause 2 missing
    slots = {"account_id", "order_id"}
    res = evaluate_cnf_clauses(slots, cnf)
    assert res == [True, True, False]

    # All satisfied
    slots_all = {"customer_name", "order_id", "email"}
    assert evaluate_cnf_clauses(slots_all, cnf) == [True, True, True]

    # Empty CNF
    assert evaluate_cnf_clauses(slots_all, []) == []


def test_extract_dst_features_for_conversation():
    schema = {
        "mandatory_slots": [["username"], ["order_id"]],
        "all_slots": ["username", "order_id", "email"],
    }
    # Turn 2: username (customer turn 1)
    # Turn 4: order_id, phone (customer turn 2)
    # Turn 6: order_id repeated (customer turn 3)
    slots_df = pd.DataFrame([
        {"turn_id": 2, "inferred_slots": ["username"]},
        {"turn_id": 4, "inferred_slots": ["order_id", "phone"]},
        {"turn_id": 6, "inferred_slots": ["order_id"]},
    ])

    feat = extract_dst_features_for_conversation(
        convo_id=101,
        convo_slots_df=slots_df,
        schema=schema,
        num_total_turns=10,
    )

    assert feat["convo_id"] == 101
    assert feat["total_slots_disclosed"] == 3  # username, order_id, phone
    assert feat["mandatory_slots_count"] == 2
    assert feat["mandatory_slots_satisfied"] == 2
    assert feat["mandatory_slot_coverage"] == 1.0
    assert feat["all_mandatory_slots_filled"] == 1
    # Latency: start_turn is 2, all satisfied at turn 4 -> 4 - 2 = 2
    assert feat["slot_filling_latency"] == 2.0
    # Both clauses satisfied in turns 2 and 4 (first 2 customer turns) -> 1.0
    assert feat["upfront_slot_density"] == 1.0
    # phone is irrelevant to all_slots ['username', 'order_id', 'email'] -> 1/3
    assert pytest.approx(feat["unused_slot_ratio"], 0.01) == 0.333
    # order_id repeated at turn 6
    assert feat["slot_repetition_count"] == 1


def test_extract_dst_features_never_satisfied():
    schema = {
        "mandatory_slots": [["username"], ["order_id"]],
        "all_slots": ["username", "order_id"],
    }
    slots_df = pd.DataFrame([
        {"turn_id": 2, "inferred_slots": ["username"]},
    ])

    feat = extract_dst_features_for_conversation(
        convo_id=102,
        convo_slots_df=slots_df,
        schema=schema,
        num_total_turns=8,
    )

    assert feat["all_mandatory_slots_filled"] == 0
    assert feat["mandatory_slots_satisfied"] == 1
    assert feat["mandatory_slot_coverage"] == 0.5
    # Capped at total turns
    assert feat["slot_filling_latency"] == 8.0


def test_extract_dst_features_zero_mandatory_schema():
    schema = {
        "mandatory_slots": [],
        "all_slots": ["amount"],
    }
    slots_df = pd.DataFrame([
        {"turn_id": 2, "inferred_slots": ["amount"]},
    ])

    feat = extract_dst_features_for_conversation(
        convo_id=103,
        convo_slots_df=slots_df,
        schema=schema,
        num_total_turns=6,
    )

    assert feat["mandatory_slots_count"] == 0
    assert feat["all_mandatory_slots_filled"] == 1
    assert feat["slot_filling_latency"] == 0.0
    assert feat["mandatory_slot_coverage"] == 1.0
    assert feat["upfront_slot_density"] == 1.0


def test_extract_ast_features_for_conversation():
    # Terminal action 'select-faq' and any 'notify-team' are stripped to prevent target leakage
    actions_taken = ["search-faq", "search-pricing", "select-faq"]
    canonical_actions = ["search-faq", "search-pricing", "select-faq"]

    feat = extract_ast_features_for_conversation(
        convo_id=201,
        actions_taken=actions_taken,
        canonical_actions=canonical_actions,
    )

    assert feat["num_actions_taken"] == 2
    assert feat["num_canonical_actions"] == 2
    assert feat["sop_edit_distance_raw"] == 0
    assert feat["sop_edit_distance_norm"] == 0.0
    assert feat["action_thrashing_count"] == 0
    assert feat["action_count_ratio"] == 1.0
    assert feat["action_count_delta"] == 0
    assert "reached_canonical_terminal" not in feat
    assert "has_escalation" not in feat


def test_extract_ast_features_premature_action():
    # Agent executes pre-terminal validate-purchase at turn 5, but order_id was only provided at turn 7
    actions_taken = ["validate-purchase", "offer-refund"]
    canonical_actions = ["validate-purchase", "offer-refund"]
    action_turns_list = [(5, "validate-purchase"), (10, "offer-refund")]
    slots_timeline = [
        (2, ["username", "email"]),
        (7, ["order_id"]),
    ]

    feat = extract_ast_features_for_conversation(
        convo_id=202,
        actions_taken=actions_taken,
        canonical_actions=canonical_actions,
        action_turns_list=action_turns_list,
        customer_slots_timeline=slots_timeline,
    )

    # Only pre-terminal 'validate-purchase' is evaluated; terminal 'offer-refund' is excluded
    assert feat["premature_action_attempts"] == 1
    assert feat["action_execution_latency_mean"] == 0.0


def test_extract_ast_features_execution_latency_prereq_only():
    # pull-up-account (turn 3) is not in A_prereq; ask-the-oracle (turn 15) is canonical_terminal
    # validate-purchase (turn 8) has prerequisites satisfied at turn 6 -> latency = 8 - 6 = 2.0
    actions_taken = ["pull-up-account", "validate-purchase", "ask-the-oracle"]
    canonical_actions = ["pull-up-account", "validate-purchase", "ask-the-oracle"]
    action_turns_list = [
        (3, "pull-up-account"),
        (8, "validate-purchase"),
        (15, "ask-the-oracle"),
    ]
    slots_timeline = [
        (2, ["username"]),
        (4, ["email"]),
        (6, ["order_id"]),
    ]

    feat = extract_ast_features_for_conversation(
        convo_id=203,
        actions_taken=actions_taken,
        canonical_actions=canonical_actions,
        action_turns_list=action_turns_list,
        customer_slots_timeline=slots_timeline,
    )

    assert feat["premature_action_attempts"] == 0
    assert feat["action_execution_latency_mean"] == 2.0
