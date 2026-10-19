"""
Dialogue State Tracking (DST / CST) Feature Extraction.

Extracts conversation-level tabular features reflecting customer communication efficiency,
slot disclosure dynamics, and normative schema compliance based on Conjunctive Normal Form
(CNF) intent requirements and turn-level extracted slots.
"""

from typing import Any

import numpy as np
import pandas as pd


def evaluate_cnf_clauses(
    cumulative_slots: set[str],
    mandatory_clauses: list[list[str]] | np.ndarray,
) -> list[bool]:
    """
    Evaluate whether each clause in a Conjunctive Normal Form (CNF) schema is satisfied.

    Each clause is an iterable of alternative acceptable slots (logical OR).
    The clause is satisfied if at least one of its slots is present in cumulative_slots.

    Args:
        cumulative_slots: Set of unique slot names disclosed up to a given point.
        mandatory_clauses: List of lists/arrays of alternative slot names.

    Returns:
        List of boolean flags indicating satisfaction for each clause.
    """
    if len(mandatory_clauses) == 0:
        return []
    return [
        any(slot in cumulative_slots for slot in clause)
        for clause in mandatory_clauses
    ]


def extract_dst_features_for_conversation(
    convo_id: int,
    convo_slots_df: pd.DataFrame,
    schema: dict[str, Any] | pd.Series,
    num_total_turns: int | None = None,
) -> dict[str, Any]:
    """
    Extract conversation-level DST / CST features for a single conversation.

    Args:
        convo_id: Conversation identifier.
        convo_slots_df: DataFrame containing turn-level extracted slots for this conversation
            (expected columns: 'turn_id', 'inferred_slots').
        schema: Intent schema dict containing 'mandatory_slots' and 'all_slots'.
        num_total_turns: Total number of turns in conversation (for latency capping).

    Returns:
        Dictionary of engineered DST features.
    """
    mandatory_clauses = schema.get("mandatory_slots", [])
    num_mandatory = len(mandatory_clauses)
    all_schema_slots = set(schema.get("all_slots", []))

    # Sort turns chronologically
    sorted_slots = convo_slots_df.sort_values("turn_id") if not convo_slots_df.empty else convo_slots_df

    cumulative_slots: set[str] = set()
    first_two_turns_slots: set[str] = set()
    seen_slots: set[str] = set()

    slot_repetition_count = 0
    all_mandatory_turn: int | None = None
    start_turn: int | None = None

    for customer_turn_idx, (_, row) in enumerate(sorted_slots.iterrows()):
        turn_id = int(row["turn_id"])
        if start_turn is None:
            start_turn = turn_id

        inferred = row.get("inferred_slots", [])
        if isinstance(inferred, np.ndarray):
            inferred = inferred.tolist()
        inferred_set = set(inferred)

        # Track repetitions (customer mentioning slots already disclosed earlier)
        for s in inferred_set:
            if s in seen_slots:
                slot_repetition_count += 1
            else:
                seen_slots.add(s)

        cumulative_slots.update(inferred_set)

        # Track upfront slots (first 2 customer turns)
        if customer_turn_idx < 2:
            first_two_turns_slots.update(inferred_set)

        # Check CNF satisfaction for the first time
        if all_mandatory_turn is None and num_mandatory > 0:
            satisfied = evaluate_cnf_clauses(cumulative_slots, mandatory_clauses)
            if all(satisfied):
                all_mandatory_turn = turn_id

    # Handle cases where intent requires 0 mandatory slots (e.g. FAQ / storewide queries)
    if num_mandatory == 0:
        all_mandatory_filled = 1
        slot_filling_latency = 0.0
        mandatory_slot_coverage = 1.0
        upfront_slot_density = 1.0
        mandatory_satisfied_count = 0
    else:
        # Final clause satisfaction
        final_satisfied = evaluate_cnf_clauses(cumulative_slots, mandatory_clauses)
        mandatory_satisfied_count = sum(final_satisfied)
        mandatory_slot_coverage = mandatory_satisfied_count / num_mandatory
        all_mandatory_filled = 1 if mandatory_satisfied_count == num_mandatory else 0

        # Upfront slot density (within customer turns 1 and 2)
        upfront_satisfied = evaluate_cnf_clauses(first_two_turns_slots, mandatory_clauses)
        upfront_slot_density = sum(upfront_satisfied) / num_mandatory

        # Slot filling latency (measured in number of dialogue turns elapsed)
        if all_mandatory_turn is not None and start_turn is not None:
            slot_filling_latency = float(all_mandatory_turn - start_turn)
        else:
            # Impute latency to total turn count (N_turns) if mandatory slots were never satisfied
            fallback_turns = float(num_total_turns) if num_total_turns is not None else float(convo_slots_df["turn_id"].max() + 1 if not convo_slots_df.empty else 0)
            slot_filling_latency = fallback_turns

    # Unused / irrelevant slot ratio
    irrelevant_slots = cumulative_slots - all_schema_slots
    unused_slot_ratio = len(irrelevant_slots) / len(cumulative_slots) if len(cumulative_slots) > 0 else 0.0

    return {
        "convo_id": convo_id,
        "total_slots_disclosed": len(cumulative_slots),
        "mandatory_slots_count": num_mandatory,
        "mandatory_slots_satisfied": mandatory_satisfied_count,
        "mandatory_slot_coverage": float(mandatory_slot_coverage),
        "all_mandatory_slots_filled": int(all_mandatory_filled),
        "slot_filling_latency": float(slot_filling_latency),
        "upfront_slot_density": float(upfront_slot_density),
        "unused_slot_ratio": float(unused_slot_ratio),
        "slot_repetition_count": int(slot_repetition_count),
    }


def extract_dst_features_batch(
    df_convos: pd.DataFrame,
    df_slots: pd.DataFrame,
    df_schemas: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract conversation-level DST / CST features in batch across all conversations.

    Args:
        df_convos: DataFrame of conversations (must contain 'convo_id', 'intent', 'num_turns').
        df_slots: DataFrame of extracted turn-level slots (must contain 'convo_id', 'turn_id', 'inferred_slots').
        df_schemas: Normative intent schema DataFrame (must contain 'intent', 'mandatory_slots', 'all_slots').

    Returns:
        DataFrame of conversation-level DST features indexed by convo_id.
    """
    schemas_by_intent = df_schemas.set_index("intent").to_dict("index")

    # Group extracted slots by convo_id for fast in-memory dictionary slicing
    slots_grouped = {
        cid: group
        for cid, group in df_slots.groupby("convo_id", sort=False)
    }

    records = []
    empty_df = pd.DataFrame(columns=["turn_id", "inferred_slots"])

    for _, convo in df_convos.iterrows():
        cid = int(convo["convo_id"])
        intent = convo["intent"]
        num_turns = int(convo["num_turns"]) if "num_turns" in convo else None

        schema = schemas_by_intent.get(intent, {})
        convo_slots = slots_grouped.get(cid, empty_df)

        feat = extract_dst_features_for_conversation(
            convo_id=cid,
            convo_slots_df=convo_slots,
            schema=schema,
            num_total_turns=num_turns,
        )
        records.append(feat)

    df_dst = pd.DataFrame(records)
    return df_dst
