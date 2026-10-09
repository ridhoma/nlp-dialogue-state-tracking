import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from itertools import batched
from typing import Any

from nlp.context import extract_customer_utterance

logger = logging.getLogger(__name__)

DEFAULT_GATE_HYPOTHESIS: str = (
    "The customer provides an identifier, contact info, code, name, or numeric amount."
)

DEFAULT_SLOT_HYPOTHESES: dict[str, str] = {
    "account_id": "The customer enters their account ID.",
    "amount": "The customer specifies a dollar amount or price.",
    "email": "The customer enters their email address.",
    "name": "The customer specifies a merchandise product name.",
    "order_id": "The customer specifies an order ID number.",
    "phone": "The customer provides a telephone phone number.",
    "pin_number": "The customer enters their PIN number.",
    "street_address": "The customer specifies a street address.",
    "username": "The customer writes their account username in the message.",
    "zip_code": "The customer provides a postal zip code.",
}

DEFAULT_SLOT_THRESHOLDS: dict[str, float] = {
    "email": 0.98,
    "phone": 0.97,
    "order_id": 0.95,
    "account_id": 0.96,
    "zip_code": 0.85,
    "amount": 0.90,
    "street_address": 0.90,
}

# Alias for backwards compatibility / shorthand
DEFAULT_HYPOTHESES = DEFAULT_SLOT_HYPOTHESES



class SlotMatcher(ABC):
    """Abstract base class for slot filling algorithms."""

    @abstractmethod
    def extract_slots(
        self,
        text: str,
        available_slots: list[str],
        batch_size: int = 128,
        return_scores: bool = False,
    ) -> list[str] | dict[str, float]:
        """Extract active slots from a single text utterance."""

    @abstractmethod
    def extract_batch(
        self,
        texts: list[str],
        available_slots: list[str],
        batch_size: int = 128,
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
        threshold: float | dict[str, float] = 0.5,
        default_threshold: float = 0.5,
        max_length: int = 128,
        use_fp16: bool = True,
        hypothesis_template: str | dict[str, str] | Callable[[str], str] = "The customer specifies their {}.",
        enable_informative_gate: bool = False,
        gate_hypothesis: str | None = None,
        gate_threshold: float = 0.02,
        enable_contextual_disambiguation: bool = False,
        verification_threshold: float = 0.02,
        device: Any | None = None,
    ):

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device is not None:
            self.device = torch.device(device) if isinstance(device, str) else device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model_name = model_name
        self.threshold = threshold
        self.default_threshold = default_threshold
        self.max_length = max_length
        self.hypothesis_template = hypothesis_template
        self.enable_informative_gate = enable_informative_gate
        self.gate_hypothesis = gate_hypothesis or DEFAULT_GATE_HYPOTHESIS
        self.gate_threshold = float(gate_threshold)
        self.enable_contextual_disambiguation = enable_contextual_disambiguation
        self.verification_threshold = float(verification_threshold)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

        # Enable FP16 acceleration on hardware accelerators (MPS / CUDA)
        self.use_fp16 = bool(use_fp16 and self.device.type in ("cuda", "mps"))
        if self.use_fp16:
            self.model.half()

        threshold_desc = (
            f"dynamic ({len(self.threshold)} thresholds)"
            if isinstance(self.threshold, dict)
            else f"{float(self.threshold):.2f}"
        )
        logger.info(
            "Initialized SemanticMatcher with model '%s' on device '%s' (threshold=%s, fp16=%s, gate=%s, disambig=%s)",
            self.model_name,
            self.device,
            threshold_desc,
            self.use_fp16,
            self.enable_informative_gate,
            self.enable_contextual_disambiguation,
        )

        # Dynamic label index resolution from model config (e.g. contradiction, neutral, entailment)
        label2id = {k.lower(): v for k, v in self.model.config.label2id.items()}
        self.entailment_id = label2id.get("entailment", 2)
        self.neutral_id = label2id.get("neutral", 1)
        self.contradiction_id = label2id.get("contradiction", 0)

    def __repr__(self) -> str:
        return (
            f"SemanticMatcher(model='{self.model_name}', "
            f"device={self.device}, threshold={self.threshold}, fp16={self.use_fp16}, "
            f"gate={self.enable_informative_gate}, disambig={self.enable_contextual_disambiguation})"
        )



    def format_hypothesis(self, slot: str) -> str:
        """Resolve NLI hypothesis premise for a given candidate slot name."""
        if isinstance(self.hypothesis_template, dict):
            if slot in self.hypothesis_template:
                return self.hypothesis_template[slot]
            clean_slot = slot.replace("_", " ")
            return f"The customer specifies their {clean_slot}."
        if callable(self.hypothesis_template):
            return self.hypothesis_template(slot)
        clean_slot = slot.replace("_", " ")
        try:
            return self.hypothesis_template.format(clean_slot)
        except (KeyError, IndexError):
            return self.hypothesis_template.format(slot=clean_slot)

    def get_threshold(self, slot: str) -> float:
        """Resolve confidence threshold for a specific slot name."""
        if isinstance(self.threshold, dict):
            return float(
                self.threshold.get(
                    slot, self.threshold.get("default", self.default_threshold)
                )
            )
        return float(self.threshold)

    def extract_slots(
        self,
        text: str,
        available_slots: list[str],
        batch_size: int = 128,
        return_scores: bool = False,
        customer_text: str | None = None,
    ) -> list[str] | dict[str, float]:
        """
        Extract active slots from a single text utterance.

        Args:
            text: Input text utterance.
            available_slots: List of slot names to evaluate.
            batch_size: Number of (text, slot) pairs to evaluate per GPU forward pass.
            return_scores: If True, returns dict of {slot: prob_entailment}.
                           If False (default), returns list of active slot names.
            customer_text: Optional explicit customer-only utterance. If omitted,
                           automatically extracted from text if formatted with [CUSTOMER]:.

        Returns:
            list[str] of active slot names, or dict[str, float] if return_scores=True.
        """
        return self.extract_batch(
            [text],
            available_slots,
            batch_size=batch_size,
            return_scores=return_scores,
            customer_texts=[customer_text] if customer_text is not None else None,
        )[0]

    def extract_batch(
        self,
        texts: list[str],
        available_slots: list[str],
        batch_size: int = 128,
        return_scores: bool = False,
        customer_texts: list[str] | None = None,
    ) -> list[list[str]] | list[dict[str, float]]:
        """
        Extract active slots from a batch of text utterances.

        Args:
            texts: List of input text utterances (preconditioned outside this matcher).
            available_slots: List of slot names to evaluate.
            batch_size: Number of (text, slot) pairs to evaluate per GPU forward pass.
            return_scores: If True, returns list of dicts mapping {slot: prob_entailment}.
                           If False (default), returns list of lists of active slot names.
            customer_texts: Optional explicit customer-only utterances. If omitted,
                           automatically extracted from texts if formatted with [CUSTOMER]:.

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

        # Stage 1: Optional Informative Utterance Gate
        if self.enable_informative_gate:
            resolved_cust_texts = (
                customer_texts
                if customer_texts is not None
                else [extract_customer_utterance(t) for t in texts]
            )
            gate_eval_indices = [
                idx
                for idx, ct in enumerate(resolved_cust_texts)
                if ct and ct.strip()
            ]
            passed_gate_indices: set[int] = set()

            if gate_eval_indices:
                with torch.inference_mode():
                    for batch_indices in batched(gate_eval_indices, batch_size):
                        chunk_premises = [
                            resolved_cust_texts[i] for i in batch_indices
                        ]
                        chunk_hyps = [
                            self.gate_hypothesis
                        ] * len(chunk_premises)

                        inputs = self.tokenizer(
                            chunk_premises,
                            chunk_hyps,
                            padding=True,
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt",
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        logits = self.model(**inputs).logits
                        probs = torch.softmax(logits, dim=-1)
                        p_entail = probs[:, self.entailment_id].cpu()

                        for k, text_idx in enumerate(batch_indices):
                            if float(p_entail[k].item()) >= self.gate_threshold:
                                passed_gate_indices.add(text_idx)

            candidate_indices = [
                idx for idx in gate_eval_indices if idx in passed_gate_indices
            ]
        else:
            candidate_indices = [
                text_idx
                for text_idx, text in enumerate(texts)
                if text and str(text).strip()
            ]

        # Stage 2: Slot Extraction Evaluation for candidate texts
        pairs: list[tuple[str, str]] = []
        pair_mapping: list[tuple[int, str]] = []  # (text_index, slot_name)

        for text_idx in candidate_indices:
            clean_text = str(texts[text_idx])
            for slot in available_slots:
                hyp = self.format_hypothesis(slot)
                pairs.append((clean_text, hyp))
                pair_mapping.append((text_idx, slot))

        if not pairs:
            return score_results if return_scores else list_results

        with torch.inference_mode():
            for batch_indices in batched(range(len(pairs)), batch_size):
                chunk_premises = [pairs[i][0] for i in batch_indices]
                chunk_hypotheses = [pairs[i][1] for i in batch_indices]

                inputs = self.tokenizer(
                    chunk_premises,
                    chunk_hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
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
                    # Entailment must exceed Neutral AND pass the slot-specific confidence threshold
                    slot_threshold = self.get_threshold(slot)
                    if p_e > p_n and p_e > slot_threshold:
                        if return_scores:
                            score_results[text_idx][slot] = p_e
                        else:
                            list_results[text_idx].append(slot)

        # Stage 3: Optional Contextual Disambiguation Gate (Customer-Support Verification)
        if self.enable_contextual_disambiguation:
            if "resolved_cust_texts" not in locals():
                resolved_cust_texts = (
                    customer_texts
                    if customer_texts is not None
                    else [extract_customer_utterance(t) for t in texts]
                )

            verify_pairs: list[tuple[str, str]] = []
            verify_mapping: list[tuple[int, str]] = []

            for text_idx in candidate_indices:
                predicted_slots = (
                    list(score_results[text_idx].keys())
                    if return_scores
                    else list(list_results[text_idx])
                )
                if not predicted_slots:
                    continue
                c_text = resolved_cust_texts[text_idx]
                if not c_text or not str(c_text).strip():
                    if return_scores:
                        score_results[text_idx].clear()
                    else:
                        list_results[text_idx].clear()
                    continue

                for slot in predicted_slots:
                    hyp = self.format_hypothesis(slot)
                    verify_pairs.append((str(c_text), hyp))
                    verify_mapping.append((text_idx, slot))

            if verify_pairs:
                with torch.inference_mode():
                    for batch_indices in batched(range(len(verify_pairs)), batch_size):
                        chunk_premises = [verify_pairs[i][0] for i in batch_indices]
                        chunk_hypotheses = [verify_pairs[i][1] for i in batch_indices]

                        inputs = self.tokenizer(
                            chunk_premises,
                            chunk_hypotheses,
                            padding=True,
                            truncation=True,
                            max_length=self.max_length,
                            return_tensors="pt",
                        )
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                        logits = self.model(**inputs).logits
                        probs = torch.softmax(logits, dim=-1)
                        p_entail = probs[:, self.entailment_id].cpu()

                        for k, pair_idx in enumerate(batch_indices):
                            text_idx, slot = verify_mapping[pair_idx]
                            p_cust = float(p_entail[k].item())

                            # Prune candidate slot if customer-only utterance provides insufficient support
                            if p_cust < self.verification_threshold:
                                if return_scores:
                                    score_results[text_idx].pop(slot, None)
                                else:
                                    if slot in list_results[text_idx]:
                                        list_results[text_idx].remove(slot)

        return score_results if return_scores else list_results


