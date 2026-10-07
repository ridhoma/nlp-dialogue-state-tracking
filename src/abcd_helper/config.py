from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Raw file paths
RAW_ABCD_GZ = RAW_DATA_DIR / "abcd_v1.1.json.gz"
RAW_ONTOLOGY = RAW_DATA_DIR / "ontology.json"
RAW_GUIDELINES = RAW_DATA_DIR / "guidelines.json"
RAW_KB = RAW_DATA_DIR / "kb.json"
RAW_UTTERANCES = RAW_DATA_DIR / "utterances.json"
RAW_SAMPLE = RAW_DATA_DIR / "abcd_sample.json"

# Processed file paths (Parquet & SQLite)
CONVERSATIONS_PARQUET = PROCESSED_DATA_DIR / "conversations.parquet"
TURNS_PARQUET = PROCESSED_DATA_DIR / "turns.parquet"
SUBFLOWS_PARQUET = PROCESSED_DATA_DIR / "subflows.parquet"
ACTIONS_PARQUET = PROCESSED_DATA_DIR / "actions.parquet"
INTENT_SCHEMA_PARQUET = PROCESSED_DATA_DIR / "intent_schema.parquet"
SQLITE_DB = PROCESSED_DATA_DIR / "abcd.db"

# Remote URLs
ABCD_GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/asappresearch/abcd/master/data"
)
REMOTE_FILES = {
    "abcd_v1.1.json.gz": f"{ABCD_GITHUB_RAW_BASE}/abcd_v1.1.json.gz",
    "ontology.json": f"{ABCD_GITHUB_RAW_BASE}/ontology.json",
    "guidelines.json": f"{ABCD_GITHUB_RAW_BASE}/guidelines.json",
    "kb.json": f"{ABCD_GITHUB_RAW_BASE}/kb.json",
    "utterances.json": f"{ABCD_GITHUB_RAW_BASE}/utterances.json",
    "abcd_sample.json": f"{ABCD_GITHUB_RAW_BASE}/abcd_sample.json",
}
