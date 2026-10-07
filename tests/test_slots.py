"""
Unit tests for SemanticMatcher (Option 1 Calibration) and string-level slot extraction.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from nlp.slots import SemanticMatcher, TwoStageSlotExtractor


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

        # 2 texts x 1 slot = 2 pairs: text 0 entailment high, text 1 neutral high
        mock_model.return_value.logits = torch.tensor(
            [
                [-2.0, -1.0, 3.0],  # text 0 -> email active
                [0.0, 3.0, 1.0],  # text 1 -> email inactive
            ]
        )

        texts = ["Here is my email: bob@acme.com", "I have an issue with delivery"]
        results = matcher.extract_batch(texts, ["email"])

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


def test_deprecated_two_stage_extractor_alias():
    """Verify TwoStageSlotExtractor issues a deprecation warning and delegates cleanly."""
    with (
        patch("transformers.AutoTokenizer.from_pretrained"),
        patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_model_cls,
        pytest.deprecated_call(),
    ):
        mock_model = MagicMock()
        mock_model.config.label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
        mock_model_cls.return_value.to.return_value = mock_model

        extractor = TwoStageSlotExtractor(
            semantic_model_name="dummy-nli",
            route_by_structure=True,
        )
        assert isinstance(extractor, SemanticMatcher)
