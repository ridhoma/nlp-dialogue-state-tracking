"""
Unit tests for SemanticMatcher (Option 1 Calibration) and string-level slot extraction.
"""

from unittest.mock import MagicMock, patch

import torch

from nlp.slots import SemanticMatcher


def test_semantic_matcher_single_string_mocked():
    """Verify extract_slots takes raw str and returns list of active slot names."""
    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_cls,
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_model_cls,
    ):
        mock_model = MagicMock()
        mock_model.config.label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
        mock_model_cls.return_value.to.return_value = mock_model

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tok_cls.return_value = mock_tok

        matcher = SemanticMatcher(model_name="dummy-nli", threshold=0.5)

        # Mock high entailment for email
        mock_model.return_value.logits = torch.tensor([[-2.0, -1.0, 3.0]])
        res = matcher.extract_slots("my email is test@domain.com", ["email"])
        assert isinstance(res, list)
        assert res == ["email"]


def test_semantic_matcher_batch_strings_mocked():
    """Verify extract_batch takes list[str] and returns list[list[str]]."""
    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_cls,
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_model_cls,
    ):
        mock_model = MagicMock()
        mock_model.config.label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
        mock_model_cls.return_value.to.return_value = mock_model

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tok_cls.return_value = mock_tok

        matcher = SemanticMatcher(model_name="dummy-nli", threshold=0.5)

        # 2 chunks with batch_size=1: chunk 0 entailment high, chunk 1 neutral high
        mock_model.side_effect = [
            MagicMock(
                logits=torch.tensor([[-2.0, -1.0, 3.0]])
            ),  # chunk 0 -> email active
            MagicMock(
                logits=torch.tensor([[0.0, 3.0, 1.0]])
            ),  # chunk 1 -> email inactive
        ]

        texts = ["Here is my email: bob@acme.com", "I have an issue with delivery"]
        results = matcher.extract_batch(texts, ["email"], batch_size=1)

        assert len(results) == 2
        assert results[0] == ["email"]
        assert results[1] == []


def test_semantic_matcher_option_1_calibration_mocked():
    """Verify Option 1 3-class probability calibration logic."""
    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_cls,
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_model_cls,
    ):
        mock_model = MagicMock()
        mock_model.config.label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
        mock_model_cls.return_value.to.return_value = mock_model

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tok_cls.return_value = mock_tok

        matcher = SemanticMatcher(model_name="dummy-nli", threshold=0.5)

        # Case A: Neutral dominates (e.g. C=0.0, N=3.0, E=1.0) -> should NOT be active
        mock_model.return_value.logits = torch.tensor([[0.0, 3.0, 1.0]])
        res_a = matcher.extract_batch(["hello"], ["email"])
        assert "email" not in res_a[0], (
            "Neutral dominant turn must not be flagged active"
        )

        # Case B: Entailment dominates but <= threshold (e.g. C=0.0, N=1.0, E=1.2) -> threshold is 0.5
        mock_model.return_value.logits = torch.tensor([[0.0, 1.0, 1.2]])
        res_b = matcher.extract_batch(["my email maybe"], ["email"])
        assert "email" not in res_b[0], (
            "Entailment below threshold must not be flagged active"
        )

        # Case C: Entailment dominates and > threshold (e.g. C=-2.0, N=-1.0, E=3.0)
        mock_model.return_value.logits = torch.tensor([[-2.0, -1.0, 3.0]])
        res_c = matcher.extract_batch(["alice@example.com"], ["email"])
        assert res_c[0] == ["email"], (
            "Entailment > Neutral and > 0.5 must be flagged active"
        )


def test_semantic_matcher_return_scores_mocked():
    """Verify return_scores=True yields dict of probability floats instead of list of strings."""
    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_cls,
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_model_cls,
    ):
        mock_model = MagicMock()
        mock_model.config.label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
        mock_model_cls.return_value.to.return_value = mock_model

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tok_cls.return_value = mock_tok

        matcher = SemanticMatcher(model_name="dummy-nli", threshold=0.5)

        # High entailment
        mock_model.return_value.logits = torch.tensor([[-2.0, -1.0, 3.0]])
        res = matcher.extract_slots("test@domain.com", ["email"], return_scores=True)
        assert isinstance(res, dict)
        assert isinstance(res.get("email"), float)
        assert res["email"] > 0.5


def test_semantic_matcher_empty_handling():
    """Verify empty text or empty slots returns empty lists immediately."""
    with (
        patch("transformers.AutoTokenizer.from_pretrained"),
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_model_cls,
    ):
        mock_model = MagicMock()
        mock_model.config.label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
        mock_model_cls.return_value.to.return_value = mock_model

        matcher = SemanticMatcher(model_name="dummy-nli")

        assert matcher.extract_batch([], ["email"]) == []
        assert matcher.extract_batch(["hello"], []) == [[]]
        assert matcher.extract_batch(["   ", ""], ["email"]) == [[], []]


def test_semantic_matcher_personalized_hypotheses():
    """Verify hypothesis_template supports dict, callable, and DEFAULT_SLOT_HYPOTHESES."""
    from nlp.slots import DEFAULT_SLOT_HYPOTHESES

    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_cls,
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_model_cls,
    ):
        mock_model = MagicMock()
        mock_model.config.label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
        mock_model_cls.return_value.to.return_value = mock_model

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tok_cls.return_value = mock_tok

        # Test 1: Dict mapping
        custom_hypotheses = {
            "name": "The customer states their own personal first or last name.",
            "account_id": "The customer provides an account number.",
        }
        matcher_dict = SemanticMatcher(
            model_name="dummy-nli", hypothesis_template=custom_hypotheses
        )
        assert (
            matcher_dict.format_hypothesis("name")
            == "The customer states their own personal first or last name."
        )
        assert (
            matcher_dict.format_hypothesis("account_id")
            == "The customer provides an account number."
        )
        # Fallback for slot not in dict
        assert (
            matcher_dict.format_hypothesis("zip_code")
            == "The customer specifies their zip code."
        )

        # Test 2: Callable mapping
        matcher_fn = SemanticMatcher(
            model_name="dummy-nli",
            hypothesis_template=lambda s: f"Slot {s.upper()} is mentioned.",
        )
        assert matcher_fn.format_hypothesis("email") == "Slot EMAIL is mentioned."

        # Test 3: DEFAULT_SLOT_HYPOTHESES contains all 10 core slots
        matcher_default_dict = SemanticMatcher(
            model_name="dummy-nli", hypothesis_template=DEFAULT_SLOT_HYPOTHESES
        )
        for slot in ["name", "email", "order_id", "account_id", "username", "zip_code"]:
            assert slot in DEFAULT_SLOT_HYPOTHESES
            assert (
                matcher_default_dict.format_hypothesis(slot)
                == DEFAULT_SLOT_HYPOTHESES[slot]
            )

        # Test 4: Verify tokenizer actually receives the custom hypothesis in extract_batch
        mock_model.return_value.logits = torch.tensor([[-2.0, -1.0, 3.0]])
        matcher_dict.extract_batch(["My name is Alice"], ["name"])
        call_args = mock_tok.call_args[0]
        assert call_args[0] == ["My name is Alice"]
        assert call_args[1] == [
            "The customer states their own personal first or last name."
        ]


def test_semantic_matcher_dynamic_thresholds():
    """Verify threshold parameter supports per-slot dict mappings and fallbacks."""
    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_cls,
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_model_cls,
    ):
        mock_model = MagicMock()
        mock_model.config.label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
        mock_model_cls.return_value.to.return_value = mock_model

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tok_cls.return_value = mock_tok

        # Test 1: Uniform float threshold
        matcher_uniform = SemanticMatcher(model_name="dummy-nli", threshold=0.75)
        assert matcher_uniform.get_threshold("email") == 0.75
        assert matcher_uniform.get_threshold("phone") == 0.75

        # Test 2: Dynamic dict thresholds with "default" key
        thresholds_dict = {
            "email": 0.95,
            "phone": 0.80,
            "default": 0.90,
        }
        matcher_dict = SemanticMatcher(
            model_name="dummy-nli",
            threshold=thresholds_dict,
            default_threshold=0.50,
        )
        assert matcher_dict.get_threshold("email") == 0.95
        assert matcher_dict.get_threshold("phone") == 0.80
        assert matcher_dict.get_threshold("order_id") == 0.90  # falls back to "default" key

        # Test 3: Fallback to default_threshold parameter when "default" key is not in dict
        matcher_no_default_key = SemanticMatcher(
            model_name="dummy-nli",
            threshold={"email": 0.98},
            default_threshold=0.85,
        )
        assert matcher_no_default_key.get_threshold("email") == 0.98
        assert matcher_no_default_key.get_threshold("zip_code") == 0.85

        # Test 4: End-to-end extraction with different thresholds per slot
        # Mock logits producing ~0.85 entailment (softmax([-10.0, 0.0, 1.74]) -> ~0.85 entailment)
        # With threshold email=0.90 and phone=0.80, only phone should be active
        import torch.nn.functional as F

        # Target probs: C=0.05, N=0.10, E=0.85
        # logits corresponding to roughly this:
        logits = torch.tensor([[-2.0, 0.0, 2.14]])  # softmax has E ≈ 0.85, N ≈ 0.10, C ≈ 0.05
        mock_model.return_value.logits = logits

        matcher_split = SemanticMatcher(
            model_name="dummy-nli",
            threshold={"email": 0.90, "phone": 0.80},
        )
        # Single text, two candidate slots
        # Chunks evaluated in batch
        mock_model.return_value.logits = torch.cat([logits, logits], dim=0)
        res = matcher_split.extract_batch(["contact info here"], ["email", "phone"])

        assert "phone" in res[0], "Phone with threshold 0.80 should pass (P_e ≈ 0.85)"
        assert "email" not in res[0], "Email with threshold 0.90 should be filtered (P_e ≈ 0.85)"


def test_semantic_matcher_informative_gate_mocked():
    """Verify that enable_informative_gate short-circuits uninformative turns."""
    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_cls,
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_model_cls,
    ):
        mock_model = MagicMock()
        mock_model.config.label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
        mock_model_cls.return_value.to.return_value = mock_model

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tok_cls.return_value = mock_tok

        matcher = SemanticMatcher(
            model_name="dummy-nli",
            threshold=0.5,
            enable_informative_gate=True,
            gate_threshold=0.10,
        )

        texts = [
            "[AGENT]: Can you give your email?\n[CUSTOMER]: Okay sure",
            "[AGENT]: Can you give your email?\n[CUSTOMER]: test@example.com",
        ]

        # Call 1: Gate evaluation on [Okay sure, test@example.com]
        # Text 0: low entailment (0.01) -> filtered
        # Text 1: high entailment (0.95) -> passed
        gate_logits = torch.tensor([
            [1.0, 2.0, -2.0],  # Text 0: entailment ≈ 0.01
            [-2.0, -1.0, 3.0], # Text 1: entailment ≈ 0.95
        ])

        # Call 2: Slot evaluation only for Text 1 on slot 'email'
        slot_logits = torch.tensor([
            [-2.0, -1.0, 3.0]  # Text 1: email active
        ])

        mock_model.side_effect = [
            MagicMock(logits=gate_logits),
            MagicMock(logits=slot_logits),
        ]

        results = matcher.extract_batch(texts, ["email"])

        assert len(results) == 2
        assert results[0] == [], "Turn 0 failed gate and must return empty slots"
        assert results[1] == ["email"], "Turn 1 passed gate and must extract email"


def test_semantic_matcher_disambiguation_gate_mocked():
    """Verify that enable_contextual_disambiguation prunes candidate slots not backed by customer utterance."""
    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_cls,
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_model_cls,
    ):
        mock_model = MagicMock()
        mock_model.config.label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
        mock_model_cls.return_value.to.return_value = mock_model

        mock_tok = MagicMock()
        mock_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tok_cls.return_value = mock_tok

        matcher = SemanticMatcher(
            model_name="dummy-nli",
            threshold=0.5,
            enable_contextual_disambiguation=True,
            verification_threshold=0.05,
        )

        texts = [
            "[AGENT]: What is your order number and phone?\n[CUSTOMER]: My order is 99881",
        ]

        # Call 1: Contextual evaluation on pairs [(text, order_id), (text, phone)]
        # Both pass Option 1 calibration (p_e > p_n and p_e > 0.5)
        # Logits: [contra, neutral, entail]
        stage2_logits = torch.tensor([
            [-2.0, -1.0, 3.0],  # order_id: P_e ≈ 0.95 > 0.5
            [-2.0, -1.0, 3.0],  # phone: P_e ≈ 0.95 > 0.5 (prompt bleed)
        ])

        # Call 2: Stage 3 verification on customer-only text "My order is 99881"
        # order_id passes verification (> 0.05)
        # phone fails verification (< 0.05)
        stage3_logits = torch.tensor([
            [-2.0, -1.0, 3.0],  # order_id on customer text: P_e ≈ 0.95
            [2.0, 1.0, -3.0],   # phone on customer text: P_e ≈ 0.004 (< 0.05)
        ])

        mock_model.side_effect = [
            MagicMock(logits=stage2_logits),
            MagicMock(logits=stage3_logits),
        ]

        results = matcher.extract_batch(texts, ["order_id", "phone"])

        assert len(results) == 1
        assert "order_id" in results[0], "order_id was verified by customer text"
        assert "phone" not in results[0], "phone was pruned because customer text had P_e < 0.05"


