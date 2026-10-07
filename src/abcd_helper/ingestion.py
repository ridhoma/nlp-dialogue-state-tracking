"""
Data Ingestion Pipeline for ABCD Dataset.

Downloads raw files if needed, extracts conversations and turns into normalized
tabular schemas, evaluates resolution target Y, and serializes to Parquet and SQLite.
"""

import gzip
import json
import logging
import re
import sqlite3
import urllib.request
from pathlib import Path

import pandas as pd

from abcd_helper.config import (
    ACTIONS_PARQUET,
    CONVERSATIONS_PARQUET,
    INTENT_SCHEMA_PARQUET,
    PROCESSED_DATA_DIR,
    RAW_ABCD_GZ,
    RAW_DATA_DIR,
    RAW_GUIDELINES,
    RAW_KB,
    RAW_ONTOLOGY,
    REMOTE_FILES,
    SQLITE_DB,
    SUBFLOWS_PARQUET,
    TURNS_PARQUET,
)
from abcd_helper.intent_schema import build_intent_schema
from abcd_helper.resolution import evaluate_conversation_resolution

logger = logging.getLogger(__name__)
SLOT_PATTERN = re.compile(r"<([a-zA-Z0-9_]+)>")


def download_raw_files(force: bool = False) -> None:
    """Download ABCD raw files from GitHub repository if not already present."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, url in REMOTE_FILES.items():
        dest = RAW_DATA_DIR / filename
        if not dest.exists() or force:
            logger.info("Downloading %s -> %s ...", url, dest)
            urllib.request.urlretrieve(url, dest)
            logger.info(
                "Saved %s (%.2f MB)", dest.name, dest.stat().st_size / (1024 * 1024)
            )
        else:
            logger.info("Found existing raw file: %s", dest.name)


def load_raw_metadata() -> tuple[dict, dict, dict]:
    """Load raw ontology, guidelines, and KB mappings."""
    with open(RAW_ONTOLOGY, "r", encoding="utf-8") as f:
        ontology = json.load(f)
    with open(RAW_GUIDELINES, "r", encoding="utf-8") as f:
        guidelines = json.load(f)
    with open(RAW_KB, "r", encoding="utf-8") as f:
        kb = json.load(f)
    return ontology, guidelines, kb


def parse_metadata_tables(
    ontology: dict, guidelines: dict, kb: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct normalized subflows and actions reference DataFrames.
    """
    # Actions table
    actions_rows = []
    for category, act_dict in ontology.get("actions", {}).items():
        for act_name, slots in act_dict.items():
            actions_rows.append(
                {
                    "action_name": act_name,
                    "action_category": category,
                    "required_slots": slots,
                    "num_required_slots": len(slots),
                }
            )
    df_actions = pd.DataFrame(actions_rows)

    # Invert ontology intents to get flow for each subflow
    subflow_to_flow = {}
    for flow, subflow_list in ontology.get("intents", {}).get("subflows", {}).items():
        for sf in subflow_list:
            subflow_to_flow[sf] = flow

    # Subflows table (from KB and ontology)
    subflow_rows = []
    for intent, canonical_acts in kb.items():
        flow = subflow_to_flow.get(intent, "")
        canonical_terminal = canonical_acts[-1] if canonical_acts else None
        subflow_rows.append(
            {
                "intent": intent,
                "flow": flow,
                "canonical_actions": canonical_acts,
                "num_canonical_actions": len(canonical_acts),
                "canonical_terminal": canonical_terminal,
            }
        )
    df_subflows = pd.DataFrame(subflow_rows)

    return df_subflows, df_actions


def parse_conversations_and_turns(
    raw_data: dict, kb: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse conversation splits into conversations and turns DataFrames.
    """
    convo_rows = []
    turn_rows = []

    for split in ["train", "dev", "test"]:
        conversations = raw_data.get(split, [])
        logger.info("Parsing split %s (%d conversations)...", split, len(conversations))

        for d in conversations:
            convo_id = d["convo_id"]
            scenario = d.get("scenario", {})
            personal = scenario.get("personal", {})
            order = scenario.get("order", {})
            product = scenario.get("product", {})

            orig_turns = d.get("original", [])
            del_turns = d.get("delexed", [])

            # Extract canonical intent from turn target
            canonical_intent = None
            if del_turns and del_turns[0].get("targets"):
                canonical_intent = del_turns[0]["targets"][0]
            if not canonical_intent:
                canonical_intent = scenario.get("subflow", "")

            flow = scenario.get("flow", "")
            scenario_subflow = scenario.get("subflow", "")

            # Aggregate turns and actions
            n_cust = 0
            n_agent = 0
            n_action = 0
            actions_taken = []

            for turn_idx, (orig, delex) in enumerate(zip(orig_turns, del_turns)):
                spk = delex.get("speaker", "")
                if spk == "customer":
                    n_cust += 1
                elif spk == "agent":
                    n_agent += 1
                elif spk == "action":
                    n_action += 1

                targets = delex.get("targets") or [None] * 5
                act_name = targets[2]
                act_vals = targets[3] if len(targets) > 3 else []
                if act_name:
                    actions_taken.append(act_name)

                text_orig = orig[1] if isinstance(orig, list) and len(orig) > 1 else ""
                text_delex = delex.get("text", "")
                slot_tags = SLOT_PATTERN.findall(text_delex) if text_delex else []

                turn_rows.append(
                    {
                        "convo_id": convo_id,
                        "turn_id": turn_idx,
                        "turn_count": delex.get("turn_count", turn_idx + 1),
                        "split": split,
                        "speaker": spk,
                        "text": text_orig,
                        "text_delexed": text_delex,
                        "flow": flow,
                        "intent": canonical_intent,
                        "next_step": targets[1],
                        "action_name": act_name,
                        "action_values": json.dumps(act_vals) if act_vals else None,
                        "slot_tags": ",".join(slot_tags) if slot_tags else None,
                        "char_len": len(text_orig),
                        "word_len": len(text_orig.split()) if text_orig else 0,
                    }
                )

            canonical_actions = kb.get(canonical_intent, [])
            resolved, diagnostics = evaluate_conversation_resolution(
                actions_taken=actions_taken,
                canonical_actions=canonical_actions,
            )

            # Extract product / order totals
            product_names = product.get("names", [])
            product_amounts = product.get("amounts", [])
            order_amount = sum(product_amounts) if product_amounts else None

            convo_rows.append(
                {
                    "convo_id": convo_id,
                    "split": split,
                    "flow": flow,
                    "intent": canonical_intent,
                    "scenario_subflow": scenario_subflow,
                    "num_turns": len(orig_turns),
                    "num_customer_turns": n_cust,
                    "num_agent_turns": n_agent,
                    "num_action_turns": n_action,
                    "customer_name": personal.get("customer_name"),
                    "username": personal.get("username"),
                    "email": personal.get("email"),
                    "phone": personal.get("phone"),
                    "member_level": personal.get("member_level"),
                    "account_id": personal.get("account_id"),
                    "order_id": order.get("order_id"),
                    "order_amount": order_amount,
                    "order_products": ", ".join(product_names)
                    if product_names
                    else None,
                    "order_city": order.get("city"),
                    "order_state": order.get("state"),
                    "order_zip": order.get("zip_code"),
                    "actions_taken": actions_taken,
                    "num_actions": len(actions_taken),
                    "canonical_actions": canonical_actions,
                    "canonical_terminal": diagnostics["canonical_terminal"],
                    "actual_terminal": diagnostics["actual_terminal"],
                    "has_escalation": diagnostics["has_escalation"],
                    "reached_canonical_terminal": diagnostics[
                        "reached_canonical_terminal"
                    ],
                    "resolved": resolved,
                }
            )

    df_conversations = pd.DataFrame(convo_rows)
    df_turns = pd.DataFrame(turn_rows)
    return df_conversations, df_turns


def export_to_sqlite(
    df_conversations: pd.DataFrame,
    df_turns: pd.DataFrame,
    df_subflows: pd.DataFrame,
    df_actions: pd.DataFrame,
    df_intent_schema: pd.DataFrame | None = None,
    db_path: Path = SQLITE_DB,
) -> None:
    """Save DataFrames into an indexed SQLite database."""
    logger.info("Writing SQLite database to %s ...", db_path)
    conn = sqlite3.connect(db_path)
    try:

        def _to_json(val):
            if hasattr(val, "tolist"):
                val = val.tolist()
            elif isinstance(val, (list, tuple)):
                val = [
                    item.tolist() if hasattr(item, "tolist") else item for item in val
                ]
            return json.dumps(val)

        # Prepare copies with string serialized lists for SQLite
        df_convos_sql = df_conversations.copy()
        df_convos_sql["actions_taken"] = df_convos_sql["actions_taken"].apply(_to_json)
        df_convos_sql["canonical_actions"] = df_convos_sql["canonical_actions"].apply(
            _to_json
        )
        df_convos_sql.to_sql("conversations", conn, if_exists="replace", index=False)

        df_turns.to_sql("turns", conn, if_exists="replace", index=False)

        df_sf_sql = df_subflows.copy()
        df_sf_sql["canonical_actions"] = df_sf_sql["canonical_actions"].apply(_to_json)
        df_sf_sql.to_sql("subflows", conn, if_exists="replace", index=False)

        df_act_sql = df_actions.copy()
        df_act_sql["required_slots"] = df_act_sql["required_slots"].apply(_to_json)
        df_act_sql.to_sql("actions", conn, if_exists="replace", index=False)

        if df_intent_schema is not None:
            df_is_sql = df_intent_schema.copy()
            df_is_sql["canonical_actions"] = df_is_sql["canonical_actions"].apply(
                _to_json
            )
            df_is_sql["mandatory_slots"] = df_is_sql["mandatory_slots"].apply(_to_json)
            df_is_sql["agent_action_slots"] = df_is_sql["agent_action_slots"].apply(
                _to_json
            )
            df_is_sql["all_slots"] = df_is_sql["all_slots"].apply(_to_json)
            df_is_sql.to_sql("intent_schema", conn, if_exists="replace", index=False)

        # Create indexes for high performance querying
        cur = conn.cursor()
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_convos_id ON conversations(convo_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_convos_split ON conversations(split)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_convos_intent ON conversations(intent)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_convos_resolved ON conversations(resolved)"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_turns_convo_id ON turns(convo_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_turns_speaker ON turns(speaker)")
        if df_intent_schema is not None:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_is_flow_intent ON intent_schema(flow, intent)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_is_intent ON intent_schema(intent)"
            )
        conn.commit()
    finally:
        conn.close()
    logger.info("SQLite database ready at %s", db_path)


def run_ingestion(force_download: bool = False, write_sqlite: bool = True) -> None:
    """
    Full ingestion pipeline orchestrator.
    Downloads raw ABCD files, parses them into conversations and turns tables,
    builds the normative intent schema, and stores them as Parquet files and SQLite database.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Download raw files
    download_raw_files(force=force_download)

    # 2. Load metadata
    ontology, guidelines, kb = load_raw_metadata()
    df_subflows, df_actions = parse_metadata_tables(ontology, guidelines, kb)

    # 3. Build intent schema
    df_intent_schema = build_intent_schema(actions_df=df_actions)

    # 4. Load conversations
    logger.info("Decompressing and loading raw ABCD conversations...")
    with gzip.open(RAW_ABCD_GZ, "rt", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 5. Parse conversations & turns
    df_conversations, df_turns = parse_conversations_and_turns(raw_data, kb)

    # 6. Save Parquet files
    logger.info("Saving conversations to Parquet: %s", CONVERSATIONS_PARQUET)
    df_conversations.to_parquet(CONVERSATIONS_PARQUET, index=False)

    logger.info("Saving turns to Parquet: %s", TURNS_PARQUET)
    df_turns.to_parquet(TURNS_PARQUET, index=False)

    logger.info("Saving subflows metadata to Parquet: %s", SUBFLOWS_PARQUET)
    df_subflows.to_parquet(SUBFLOWS_PARQUET, index=False)

    logger.info("Saving actions metadata to Parquet: %s", ACTIONS_PARQUET)
    df_actions.to_parquet(ACTIONS_PARQUET, index=False)

    logger.info("Saving intent schema to Parquet: %s", INTENT_SCHEMA_PARQUET)
    df_intent_schema.to_parquet(INTENT_SCHEMA_PARQUET, index=False)

    # 7. Save SQLite database
    if write_sqlite:
        export_to_sqlite(
            df_conversations,
            df_turns,
            df_subflows,
            df_actions,
            df_intent_schema=df_intent_schema,
        )

    logger.info("Ingestion completed successfully!")
    logger.info("Conversations: %d rows", len(df_conversations))
    logger.info("Turns: %d rows", len(df_turns))
    logger.info("Subflows: %d rows", len(df_subflows))
    logger.info("Actions: %d rows", len(df_actions))
    logger.info("Intent Schema: %d rows", len(df_intent_schema))
