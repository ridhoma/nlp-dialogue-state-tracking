import logging
import warnings
from abc import ABC, abstractmethod
from itertools import batched
from typing import Any

logger = logging.getLogger(__name__)


class SlotMatcher(ABC):
    """Abstract base class for slot filling algorithms."""

    @abstractmethod
    def extract_slots(
        self,
        text: str,
        available_slots: list[str],
        return_scores: bool = False,
    ) -> list[str] | dict[str, float]:
        """Extract active slots from a single text utterance."""

    @abstractmethod
    def extract_batch(
        self,
        texts: list[str],
        available_slots: list[str],
        return_scores: bool = False,
    ) -> list[list[str]] | list[dict[str, float]]:
        """Extract active slots from a batch of text utterances."""


class SemanticMatcher(SlotMatcher):
    """
    Semantic Slot Matcher using zero-shot / one-shot Multi-Genre NLI (MNLI).
    Evaluates slots from raw text inputs using Option 1 3-class probability calibration:
        P(Entailment) > P(Neutral) and P(Entailment) > Threshold.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-mnli",
        batch_size: int = 32,
        threshold: float = 0.5,
        hypothesis_template: str = "The customer specifies their {}.",
        device: Any | None = None,
        **kwargs: Any,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Handle legacy or alternative argument names
        if "semantic_model_name" in kwargs:
            model_name = kwargs["semantic_model_name"]

        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.batch_size = batch_size
        self.threshold = threshold
        self.hypothesis_template = hypothesis_template
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

        # Dynamic label index resolution from model config (e.g. contradiction, neutral, entailment)
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.entailment_id = label2id.get("entailment", 2)
        self.neutral_id = label2id.get("neutral", 1)
        self.contradiction_id = label2id.get("contradiction", 0)

    def extract_slots(
        self,
        text: str,
        available_slots: list[str],
        return_scores: bool = False,
    ) -> list[str] | dict[str, float]:
        """
        Extract active slots from a single text utterance.

        Args:
            text: Input text utterance.
            available_slots: List of slot names to evaluate.
            return_scores: If True, returns dict of {slot: prob_entailment}.
                           If False (default), returns list of active slot names.

        Returns:
            list[str] of active slot names, or dict[str, float] if return_scores=True.
        """
        return self.extract_batch([text], available_slots, return_scores=return_scores)[
            0
        ]

    def extract_batch(
        self,
        texts: list[str],
        available_slots: list[str],
        return_scores: bool = False,
    ) -> list[list[str]] | list[dict[str, float]]:
        """
        Extract active slots from a batch of text utterances.

        Args:
            texts: List of input text utterances (preconditioned outside this matcher).
            available_slots: List of slot names to evaluate.
            return_scores: If True, returns list of dicts mapping {slot: prob_entailment}.
                           If False (default), returns list of lists of active slot names.

        Returns:
            list[list[str]] of active slot names per utterance, or
            list[dict[str, float]] if return_scores=True.
        """
        import torch

        if return_scores:
            score_results: list[dict[str, float]] = [{} for _ in texts]
        else:
            list_results: list[list[str]] = [[] for _ in texts]

        if not available_slots or not texts:
            return score_results if return_scores else list_results

        # Construct all (text, slot) premise-hypothesis pairs for non-empty texts
        pairs: list[tuple[str, str]] = []
        pair_mapping: list[tuple[int, str]] = []  # (text_index, slot_name)

        for text_idx, text in enumerate(texts):
            if not text or not str(text).strip():
                continue
            clean_text = str(text)
            for slot in available_slots:
                clean_slot = slot.replace("_", " ")
                hyp = self.hypothesis_template.format(clean_slot)
                pairs.append((clean_text, hyp))
                pair_mapping.append((text_idx, slot))

        if not pairs:
            return score_results if return_scores else list_results

        with torch.no_grad():
            for batch_indices in batched(range(len(pairs)), self.batch_size):
                chunk_premises = [pairs[i][0] for i in batch_indices]
                chunk_hypotheses = [pairs[i][1] for i in batch_indices]

                inputs = self.tokenizer(
                    chunk_premises,
                    chunk_hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)

                p_neutral = probs[:, self.neutral_id].cpu()
                p_entail = probs[:, self.entailment_id].cpu()

                for k, pair_idx in enumerate(batch_indices):
                    text_idx, slot = pair_mapping[pair_idx]
                    p_e = float(p_entail[k].item())
                    p_n = float(p_neutral[k].item())

                    # Option 1 Calibration Rule:
                    # Entailment must exceed Neutral AND pass the confidence threshold
                    if p_e > p_n and p_e > self.threshold:
                        if return_scores:
                            score_results[text_idx][slot] = p_e
                        else:
                            list_results[text_idx].append(slot)

        return score_results if return_scores else list_results


class TwoStageSlotExtractor(SemanticMatcher):
    """
    Deprecated alias for SemanticMatcher.
    RegexMatcher has been removed in favor of direct calibrated SemanticMatcher inference.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "TwoStageSlotExtractor is deprecated; RegexMatcher has been removed. "
            "Please use SemanticMatcher directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        kwargs.pop("route_by_structure", None)
        super().__init__(*args, **kwargs)

    def extract(self, *args: Any, **kwargs: Any) -> list[str] | dict[str, float]:
        """Deprecated compatibility method mapping to extract_slots."""
        return self.extract_slots(*args, **kwargs)
