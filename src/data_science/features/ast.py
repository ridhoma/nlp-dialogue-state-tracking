"""
Action State Tracking (AST) Feature Extraction.

Extracts conversation-level tabular features reflecting agent Standard Operating Procedure (SOP)
compliance, workflow deviations, action thrashing, and prerequisite coordination against
normative guidelines.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def token_levenshtein(seq1: Sequence[str], seq2: Sequence[str]) -> int:
    """
    Compute token-level Levenshtein edit distance between two sequences of action tokens.

    Operations:
        - Insertion of an action (cost = 1)
        - Deletion of an action (cost = 1)
        - Substitution of an action (cost = 1)

    Args:
        seq1: Observed sequence of action tokens.
        seq2: Canonical / expected sequence of action tokens.

    Returns:
        Minimum number of edit operations to transform seq1 into seq2.
    """
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n

    # Dynamic programming matrix (O(N * M))
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # Deletion
                    dp[i][j - 1],      # Insertion
                    dp[i - 1][j - 1],  # Substitution
                )

    return dp[n][m]


def compute_action_thrashing(actions: Sequence[str]) -> int:
    """
    Compute the count of consecutive identical actions executed in a sequence.

    Args:
        actions: Sequence of action names.

    Returns:
        Count of times action[t] == action[t+1].
    """
    if len(actions) < 2:
        return 0
    return sum(1 for i in range(len(actions) - 1) if actions[i] == actions[i + 1])


def check_action_prerequisites(action_name: str, cumulative_slots: set[str]) -> bool:
    """
    Check whether tracked customer entity prerequisites for an action are satisfied.

    Handles conjunctive requirements (e.g. validate-purchase requiring username, email, order_id)
    and disjunctive verification requirements across flows (e.g. verify-identity and enter-details).
    Note: `pull-up-account` is excluded from prerequisite gating because 97.1% of calls use
    synthetic `customer_name` (which is not in ABCD's ground-truth `slot_tags`).

    Args:
        action_name: Name of the executed system action.
        cumulative_slots: Set of unique customer slots disclosed prior to the action turn.

    Returns:
        True if prerequisite customer entities were present, False otherwise.
    """
    if action_name == "validate-purchase":
        return {"username", "email", "order_id"}.issubset(cumulative_slots)
    elif action_name == "pull-up-account":
        return True
    elif action_name == "verify-identity":
        return bool(
            {"order_id", "zip_code", "account_id", "username", "email", "phone", "pin_number"}
            & cumulative_slots
        )
    elif action_name == "offer-refund":
        return "amount" in cumulative_slots
    elif action_name == "enter-details":
        return bool(
            {"full_address", "street_address", "phone", "email", "pin_number", "account_id", "username"}
            & cumulative_slots
        )
    return True


def extract_ast_features_for_conversation(
    convo_id: int,
    actions_taken: Sequence[str],
    canonical_actions: Sequence[str],
    action_turns_list: list[tuple[int, str]] | None = None,
    customer_slots_timeline: list[tuple[int, list[str]]] | None = None,
) -> dict[str, Any]:
    """
    Extract conversation-level AST features for a single conversation.

    To prevent target leakage with the resolution label Y (which is defined by reaching
    `canonical_actions[-1]` without `"notify-team"`), all AST metrics are evaluated strictly
    on the pre-terminal prerequisite action sequence:
      - `canonical_pre`: `canonical_actions[:-1]` excluding `"notify-team"`
      - `observed_pre`: `actions_taken` excluding `canonical_terminal` and `"notify-team"`

    Args:
        convo_id: Conversation identifier.
        actions_taken: List/array of observed system actions executed by agent.
        canonical_actions: List/array of guideline actions from SOP.
        action_turns_list: Optional list of (turn_id, action_name) tuples.
        customer_slots_timeline: Optional list of (turn_id, inferred_slots) tuples.

    Returns:
        Dictionary of engineered pre-terminal AST features.
    """
    raw_actions_taken = list(actions_taken) if actions_taken is not None else []
    raw_canonical_actions = list(canonical_actions) if canonical_actions is not None else []

    canonical_terminal = raw_canonical_actions[-1] if len(raw_canonical_actions) > 0 else None

    # Pre-terminal prerequisite sequences (strip outcome-defining actions to prevent target leakage)
    canonical_actions_list = [
        act for act in raw_canonical_actions[:-1] if act != "notify-team"
    ]
    actions_taken_list = [
        act for act in raw_actions_taken if act != canonical_terminal and act != "notify-team"
    ]

    num_actions = len(actions_taken_list)
    num_canonical = len(canonical_actions_list)

    # 1. Levenshtein Sequence Alignment (on pre-terminal prerequisite sequence)
    raw_edit = token_levenshtein(actions_taken_list, canonical_actions_list)
    max_len = max(num_actions, num_canonical, 1)
    norm_edit = raw_edit / max_len

    # 2. Action Thrashing (Consecutive Duplicates)
    thrashing_count = compute_action_thrashing(actions_taken_list)

    # 3. Action Counts & Ratios
    action_count_ratio = (
        num_actions / num_canonical
        if num_canonical > 0
        else (1.0 if num_actions == 0 else float(num_actions))
    )
    action_count_delta = num_actions - num_canonical

    # 4. Premature Actions & Execution Latency (over pre-terminal prerequisite-gated actions A_prereq)
    prereq_actions = {"validate-purchase", "verify-identity", "offer-refund", "enter-details"}
    premature_count = 0
    execution_latencies = []

    if action_turns_list and customer_slots_timeline:
        for turn_act, act_name in action_turns_list:
            if act_name == canonical_terminal or act_name == "notify-team":
                continue
            if act_name not in prereq_actions:
                continue

            # Slots available prior to action execution
            prior_slots: set[str] = set()
            for turn_slot, s_list in customer_slots_timeline:
                if turn_slot < turn_act:
                    prior_slots.update(s_list)

            # Check if action was triggered before prerequisites were satisfied
            if not check_action_prerequisites(act_name, prior_slots):
                premature_count += 1
            else:
                # Find earliest turn where prerequisites were satisfied
                cum_eval: set[str] = set()
                satisfied_turn = None
                for turn_slot, s_list in customer_slots_timeline:
                    cum_eval.update(s_list)
                    if check_action_prerequisites(act_name, cum_eval):
                        satisfied_turn = turn_slot
                        break
                if satisfied_turn is not None and turn_act >= satisfied_turn:
                    execution_latencies.append(turn_act - satisfied_turn)

    mean_latency = float(np.mean(execution_latencies)) if execution_latencies else 0.0

    return {
        "convo_id": convo_id,
        "num_actions_taken": int(num_actions),
        "num_canonical_actions": int(num_canonical),
        "is_optimal_depth_2_or_3": int(num_canonical in (2, 3)),
        "sop_edit_distance_raw": int(raw_edit),
        "sop_edit_distance_norm": float(norm_edit),
        "action_thrashing_count": int(thrashing_count),
        "action_count_ratio": float(action_count_ratio),
        "action_count_delta": int(action_count_delta),
        "premature_action_attempts": int(premature_count),
        "action_execution_latency_mean": float(mean_latency),
    }


def extract_ast_features_batch(
    df_convos: pd.DataFrame,
    df_turns: pd.DataFrame | None = None,
    df_slots: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Extract conversation-level AST features in batch across all conversations.

    Args:
        df_convos: DataFrame of conversations (must contain 'convo_id', 'actions_taken', 'canonical_actions').
        df_turns: Optional DataFrame of turns for turn-level action timestamps.
        df_slots: Optional DataFrame of extracted customer slots for prerequisite checks.

    Returns:
        DataFrame of conversation-level AST features indexed by convo_id.
    """
    # Pre-index action turns by convo_id if df_turns is provided
    actions_by_convo = {}
    if df_turns is not None:
        action_turns = df_turns[df_turns["speaker"] == "action"][["convo_id", "turn_id", "action_name"]]
        for cid, group in action_turns.groupby("convo_id", sort=False):
            actions_by_convo[int(cid)] = group[["turn_id", "action_name"]].values.tolist()

    # Pre-index slots by convo_id if df_slots is provided
    slots_by_convo = {}
    if df_slots is not None:
        for cid, group in df_slots.groupby("convo_id", sort=False):
            slots_by_convo[int(cid)] = [
                (int(r["turn_id"]), list(r["inferred_slots"]))
                for _, r in group.iterrows()
            ]

    records = []
    for _, convo in df_convos.iterrows():
        cid = int(convo["convo_id"])
        actions_taken = convo.get("actions_taken", [])
        canonical_actions = convo.get("canonical_actions", [])

        action_turns_list = actions_by_convo.get(cid, None)
        slots_timeline = slots_by_convo.get(cid, None)

        feat = extract_ast_features_for_conversation(
            convo_id=cid,
            actions_taken=actions_taken,
            canonical_actions=canonical_actions,
            action_turns_list=action_turns_list,
            customer_slots_timeline=slots_timeline,
        )
        records.append(feat)

    df_ast = pd.DataFrame(records)
    return df_ast
