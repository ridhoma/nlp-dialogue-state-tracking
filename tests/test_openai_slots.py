"""
Unit tests for OpenAI Batch API and Decisions API slot extraction.
All network calls are mocked; no OpenAI API keys or internet connection required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from nlp.slots import (
    OpenAIBatchSlotExtractor,
    OpenAIDecisionSlotExtractor,
)
from utils.openai_helper import (
    build_batch_request_line,
    build_decision_predicate_questions,
    format_conversation_for_llm,
    get_openai_client,
    parse_batch_output_jsonl,
    parse_decision_answers,
    parse_llm_slot_annotations,
    prepare_batch_input_jsonl,
)

# =============================================================================
# 1. Tests for openai_helper.py
# =============================================================================


def test_import_from_utils_module() -> None:
    """Verify both `from utils import openai_helper` and direct submodule imports work."""
    from utils import openai_helper
    from utils.openai_helper import get_openai_client as helper_get_client

    assert hasattr(openai_helper, "get_openai_client")
    assert hasattr(openai_helper, "build_decision_predicate_questions")
    assert helper_get_client is openai_helper.get_openai_client


def test_build_slots_extraction_prompt_custom() -> None:
    """Verify dynamic prompt formatting from slot definitions dictionary."""
    from nlp.prompts import (
        DEFAULT_SLOT_DEFINITIONS,
        SLOTS_EXTRACTION_PROMPT,
        build_slots_extraction_prompt,
    )

    # 1. Default prompt should contain all 11 canonical slot definitions
    for slot_name in DEFAULT_SLOT_DEFINITIONS:
        assert f'"{slot_name}"' in SLOTS_EXTRACTION_PROMPT

    # 2. Custom dictionary should format cleanly into prompt
    custom_dict = {"custom_slot": "The customer provides a custom identifier."}
    prompt = build_slots_extraction_prompt(custom_dict)
    assert '"custom_slot": "The customer provides a custom identifier."' in prompt
    assert "# The Task" in prompt
    assert "Notice that output file only contains CUSTOMER's turns" in prompt


def test_get_openai_client_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ValueError is raised when no API key is passed or present in env."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OpenAI API key is missing"):
        get_openai_client(api_key=None)


def test_get_openai_client_explicit_key() -> None:
    """Verify client is created when API key is explicitly supplied."""
    client = get_openai_client(api_key="sk-test-key-12345")
    assert client.api_key == "sk-test-key-12345"


def test_format_conversation_for_llm() -> None:
    """Verify conversation formatting adheres to SLOTS_EXTRACTION_PROMPT schema."""
    turns = [
        {
            "convo_id": 101,
            "turn_id": 0,
            "speaker": "agent",
            "text": "Hello, how can I help?",
        },
        {"convo_id": 101, "turn_id": 1, "speaker": "action", "text": "pull-up-account"},
        {
            "convo_id": 101,
            "turn_id": 2,
            "speaker": "customer",
            "text": "My email is bob@acme.com",
        },
    ]
    # Without action turns
    res = format_conversation_for_llm(turns, convo_id=101, include_actions=False)
    assert "101" in res
    assert len(res["101"]) == 2
    assert res["101"][0] == {
        "turn_id": 0,
        "speaker": "AGENT",
        "text": "Hello, how can I help?",
    }
    assert res["101"][1] == {
        "turn_id": 2,
        "speaker": "CUSTOMER",
        "text": "My email is bob@acme.com",
    }

    # With action turns
    res_actions = format_conversation_for_llm(turns, convo_id=101, include_actions=True)
    assert len(res_actions["101"]) == 3


def test_build_batch_request_line_responses() -> None:
    """Verify /v1/responses request line conforms to OpenAI Batch API specification."""
    payload = {"101": [{"turn_id": 0, "speaker": "CUSTOMER", "text": "order #123"}]}
    line = build_batch_request_line(
        custom_id="convo_101",
        conversations_payload=payload,
        model="gpt-6-luna",
        endpoint="/v1/responses",
    )
    assert line["custom_id"] == "convo_101"
    assert line["method"] == "POST"
    assert line["url"] == "/v1/responses"
    assert line["body"]["model"] == "gpt-6-luna"
    assert "instructions" in line["body"]
    assert "input" in line["body"]
    assert line["body"]["text"]["format"]["type"] == "json_schema"


def test_build_batch_request_line_chat_completions() -> None:
    """Verify /v1/chat/completions request line conforms to OpenAI Batch API specification."""
    payload = {"101": [{"turn_id": 0, "speaker": "CUSTOMER", "text": "order #123"}]}
    line = build_batch_request_line(
        custom_id="convo_101",
        conversations_payload=payload,
        model="gpt-6-luna",
        endpoint="/v1/chat/completions",
    )
    assert line["url"] == "/v1/chat/completions"
    assert "messages" in line["body"]
    assert len(line["body"]["messages"]) == 2
    assert line["body"]["response_format"]["type"] == "json_schema"


def test_prepare_batch_input_jsonl_offline(tmp_path: Path) -> None:
    """Verify offline JSONL preparation writes valid JSONL lines to disk."""
    df_sample = pd.DataFrame(
        [
            {"convo_id": 1, "turn_id": 0, "speaker": "agent", "text": "Hello"},
            {"convo_id": 1, "turn_id": 1, "speaker": "customer", "text": "I am Alice"},
            {"convo_id": 2, "turn_id": 0, "speaker": "agent", "text": "Hi"},
            {"convo_id": 2, "turn_id": 1, "speaker": "customer", "text": "I am Bob"},
        ]
    )
    out_file = tmp_path / "batch.jsonl"
    lines = prepare_batch_input_jsonl(
        df_turns=df_sample,
        output_path=out_file,
        model="gpt-6-luna",
        endpoint="/v1/responses",
    )
    assert len(lines) == 2
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(content) == 2
    parsed_0 = json.loads(content[0])
    assert parsed_0["custom_id"] == "convo_1"


def test_parse_llm_slot_annotations() -> None:
    """Verify parsing handles markdown code blocks, root lists, and root objects."""
    # Case 1: Markdown codeblock with root array
    raw_md = """```json
    [
        {"conversation_id": 101, "turn_id": 2, "slots": ["email", "account_id"]},
        {"conversation_id": 101, "turn_id": 4, "slots": []}
    ]
    ```"""
    ann_1 = parse_llm_slot_annotations(raw_md)
    assert len(ann_1) == 2
    assert ann_1[0].conversation_id == 101
    assert ann_1[0].turn_id == 2
    assert ann_1[0].slots == ["account_id", "email"]  # Sorted unique

    # Case 2: Root object with "annotations" key
    raw_obj = json.dumps(
        {"annotations": [{"conversation_id": 202, "turn_id": 1, "slots": ["order_id"]}]}
    )
    ann_2 = parse_llm_slot_annotations(raw_obj)
    assert len(ann_2) == 1
    assert ann_2[0].conversation_id == 202
    assert ann_2[0].slots == ["order_id"]


def test_parse_batch_output_jsonl() -> None:
    """Verify downloaded batch output JSONL is parsed into canonical DataFrame."""
    batch_output_text = "\n".join(
        [
            json.dumps(
                {
                    "id": "batch_req_1",
                    "custom_id": "convo_101",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "output": [
                                {
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": json.dumps(
                                                {
                                                    "annotations": [
                                                        {
                                                            "conversation_id": 101,
                                                            "turn_id": 1,
                                                            "slots": ["order_id"],
                                                        },
                                                        {
                                                            "conversation_id": 101,
                                                            "turn_id": 3,
                                                            "slots": ["email"],
                                                        },
                                                    ]
                                                }
                                            ),
                                        }
                                    ]
                                }
                            ]
                        },
                    },
                    "error": None,
                }
            ),
            json.dumps(
                {
                    "id": "batch_req_2",
                    "custom_id": "convo_102",
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "content": json.dumps(
                                            [
                                                {
                                                    "conversation_id": 102,
                                                    "turn_id": 2,
                                                    "slots": ["zip_code"],
                                                }
                                            ]
                                        )
                                    }
                                }
                            ]
                        },
                    },
                    "error": None,
                }
            ),
        ]
    )

    df = parse_batch_output_jsonl(batch_output_text)
    assert len(df) == 3
    assert list(df.columns) == ["convo_id", "turn_id", "inferred_slots"]
    assert df.iloc[0]["convo_id"] == 101
    assert df.iloc[0]["turn_id"] == 1
    assert df.iloc[0]["inferred_slots"] == ["order_id"]
    assert df.iloc[2]["convo_id"] == 102
    assert df.iloc[2]["inferred_slots"] == ["zip_code"]


def test_build_decision_predicate_questions() -> None:
    """Verify predicate questions are created with proper name, type, and instructions."""
    slots = ["order_id", "email", "custom_slot"]
    questions = build_decision_predicate_questions(slots)
    assert len(questions) == 3
    for q in questions:
        assert q["type"] == "predicate"
        assert "instructions" in q
    assert questions[0]["name"] == "order_id"
    assert questions[2]["name"] == "custom_slot"
    assert "custom slot" in questions[2]["instructions"].lower()


def test_parse_decision_answers() -> None:
    """Verify probability thresholding on predicate answers."""
    mock_answers = [
        {"name": "order_id", "probability": 0.95, "type": "predicate"},
        {"name": "email", "probability": 0.30, "type": "predicate"},
        {"name": "zip_code", "probability": 0.85, "type": "predicate"},
    ]
    # Flat threshold = 0.50
    scores_flat = parse_decision_answers(mock_answers, thresholds=0.50)
    assert "order_id" in scores_flat
    assert "zip_code" in scores_flat
    assert "email" not in scores_flat
    assert scores_flat["order_id"] == 0.95

    # Slot-specific threshold dict
    thresholds_dict = {"order_id": 0.98, "zip_code": 0.80, "default": 0.50}
    scores_dict = parse_decision_answers(mock_answers, thresholds=thresholds_dict)
    assert "order_id" not in scores_dict  # 0.95 < 0.98 cutoff
    assert "zip_code" in scores_dict  # 0.85 >= 0.80 cutoff


# =============================================================================
# 2. Tests for OpenAIBatchSlotExtractor
# =============================================================================


def test_openai_batch_extractor_prepare_file(tmp_path: Path) -> None:
    """Verify OpenAIBatchSlotExtractor prepares batch file offline."""
    df_sample = pd.DataFrame(
        [
            {"convo_id": 10, "turn_id": 0, "speaker": "agent", "text": "Welcome!"},
            {
                "convo_id": 10,
                "turn_id": 1,
                "speaker": "customer",
                "text": "My phone is 555-1234",
            },
        ]
    )
    extractor = OpenAIBatchSlotExtractor(
        model_name="gpt-6-luna", endpoint="/v1/responses"
    )
    out_file = tmp_path / "batch_test.jsonl"
    lines = extractor.prepare_batch_file(df_sample, output_path=out_file)
    assert len(lines) == 1
    assert out_file.exists()


def test_openai_batch_extractor_submit_and_poll_mocked(tmp_path: Path) -> None:
    """Verify batch upload, submission, and polling logic with a mocked client."""
    mock_client = MagicMock()
    mock_file = MagicMock()
    mock_file.id = "file_123"
    mock_client.files.create.return_value = mock_file

    mock_batch = MagicMock()
    mock_batch.id = "batch_abc999"
    mock_batch.status = "completed"
    mock_client.batches.create.return_value = mock_batch
    mock_client.batches.retrieve.return_value = mock_batch

    input_file = tmp_path / "dummy.jsonl"
    input_file.write_text('{"dummy": true}\n')

    extractor = OpenAIBatchSlotExtractor(client=mock_client)
    batch_id = extractor.submit_batch(input_file)
    assert batch_id == "batch_abc999"
    mock_client.files.create.assert_called_once()
    mock_client.batches.create.assert_called_once()

    completed_batch = extractor.poll_batch(batch_id, poll_interval_seconds=0)
    assert completed_batch.status == "completed"


def test_openai_batch_extractor_download_and_parse_mocked(tmp_path: Path) -> None:
    """Verify downloading and parsing output file with a mocked client."""
    mock_client = MagicMock()
    mock_content = MagicMock()
    mock_content.text = (
        json.dumps(
            {
                "id": "batch_req_1",
                "custom_id": "convo_999",
                "response": {
                    "status_code": 200,
                    "body": {
                        "output": [
                            {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps(
                                            [
                                                {
                                                    "conversation_id": 999,
                                                    "turn_id": 2,
                                                    "slots": ["phone"],
                                                }
                                            ]
                                        ),
                                    }
                                ]
                            }
                        ]
                    },
                },
                "error": None,
            }
        )
        + "\n"
    )
    mock_client.files.content.return_value = mock_content

    extractor = OpenAIBatchSlotExtractor(client=mock_client)
    dl_path = tmp_path / "downloaded_batch.jsonl"
    df = extractor.download_and_parse_results(
        output_file_id="file_output_999", download_path=dl_path
    )

    assert dl_path.exists()
    assert len(df) == 1
    assert df.iloc[0]["convo_id"] == 999
    assert df.iloc[0]["inferred_slots"] == ["phone"]


def test_openai_batch_extractor_extract_slots_mocked() -> None:
    """Verify synchronous extract_slots call on OpenAIBatchSlotExtractor."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_part = MagicMock()
    mock_part.text = json.dumps(
        [{"conversation_id": 0, "turn_id": 0, "slots": ["email"]}]
    )
    mock_item = MagicMock()
    mock_item.content = [mock_part]
    mock_resp.output = [mock_item]
    mock_client.responses.create.return_value = mock_resp

    extractor = OpenAIBatchSlotExtractor(client=mock_client, endpoint="/v1/responses")
    slots = extractor.extract_slots(
        "My email is test@test.com", available_slots=["email", "phone"]
    )
    assert slots == ["email"]


# =============================================================================
# 3. Tests for OpenAIDecisionSlotExtractor
# =============================================================================


def test_openai_decision_extractor_extract_slots_mocked() -> None:
    """Verify single turn decision evaluation with mocked Decision object."""
    mock_client = MagicMock()
    mock_decision = MagicMock()

    class MockAnswer:
        def __init__(self, name: str, probability: float):
            self.name = name
            self.probability = probability
            self.type = "predicate"

    mock_decision.answers = [
        MockAnswer("order_id", 0.94),
        MockAnswer("email", 0.12),
        MockAnswer("zip_code", 0.05),
    ]
    mock_client.decisions.create.return_value = mock_decision

    extractor = OpenAIDecisionSlotExtractor(client=mock_client, threshold=0.50)
    res = extractor.extract_slots(
        "My order is 99881", available_slots=["order_id", "email", "zip_code"]
    )
    assert res == ["order_id"]

    scores = extractor.extract_slots(
        "My order is 99881",
        available_slots=["order_id", "email", "zip_code"],
        return_scores=True,
    )
    assert scores == {"order_id": 0.94}


def test_openai_decision_extractor_extract_batch_mocked() -> None:
    """Verify concurrent batch evaluation with mocked Decisions API."""
    mock_client = MagicMock()

    class MockAnswer:
        def __init__(self, name: str, probability: float):
            self.name = name
            self.probability = probability
            self.type = "predicate"

    # Turn 0 -> email active (0.91)
    # Turn 1 -> no slots active
    decision_0 = MagicMock(
        answers=[MockAnswer("email", 0.91), MockAnswer("order_id", 0.01)]
    )
    decision_1 = MagicMock(
        answers=[MockAnswer("email", 0.02), MockAnswer("order_id", 0.05)]
    )

    mock_client.decisions.create.side_effect = [decision_0, decision_1]

    extractor = OpenAIDecisionSlotExtractor(
        client=mock_client, threshold=0.50, max_workers=2
    )
    texts = ["Here is my email: a@b.com", "Thanks for your help!"]
    results = extractor.extract_batch(texts, available_slots=["email", "order_id"])

    assert len(results) == 2
    assert results[0] == ["email"]
    assert results[1] == []


def test_openai_decision_extractor_extract_from_conversations_mocked() -> None:
    """Verify full DataFrame extraction with DialogueTurnContext integration."""
    mock_client = MagicMock()

    class MockAnswer:
        def __init__(self, name: str, probability: float):
            self.name = name
            self.probability = probability
            self.type = "predicate"

    decision = MagicMock(answers=[MockAnswer("zip_code", 0.96)])
    mock_client.decisions.create.return_value = decision

    df_turns = pd.DataFrame(
        [
            {
                "convo_id": 500,
                "turn_id": 0,
                "speaker": "agent",
                "text": "What is your zip?",
            },
            {"convo_id": 500, "turn_id": 1, "speaker": "customer", "text": "90210"},
        ]
    )

    extractor = OpenAIDecisionSlotExtractor(client=mock_client, threshold=0.50)
    df_res = extractor.extract_from_conversations(
        df_turns, available_slots=["zip_code"]
    )

    assert len(df_res) == 1
    assert df_res.iloc[0]["convo_id"] == 500
    assert df_res.iloc[0]["turn_id"] == 1
    assert df_res.iloc[0]["inferred_slots"] == ["zip_code"]


# =============================================================================
# 4. Tests for CLI Inference Scripts
# =============================================================================


def test_cli_batch_script_prepare(tmp_path: Path) -> None:
    """Verify CLI batch extraction script with --action prepare."""
    import subprocess
    import sys

    out_jsonl = tmp_path / "test_cli_batch.jsonl"
    cmd = [
        sys.executable,
        "scripts/run_openai_batch_slot_extraction.py",
        "--dataset",
        "benchmark",
        "--sample",
        "2",
        "--action",
        "prepare",
        "--batch-input-jsonl",
        str(out_jsonl),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Script failed: {res.stderr}"
    assert out_jsonl.exists()
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_cli_decision_script_dry_run() -> None:
    """Verify CLI decision extraction script with --dry-run."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "scripts/run_openai_decision_slot_extraction.py",
        "--dataset",
        "benchmark",
        "--sample",
        "2",
        "--dry-run",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"Script failed: {res.stderr}"
    assert "OPENAI DECISIONS API: DRY RUN INSPECTION & COST ESTIMATE" in res.stdout
    assert "Candidate Slots:" in res.stdout
