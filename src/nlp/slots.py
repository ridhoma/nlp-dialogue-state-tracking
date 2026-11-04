import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from nlp.context import extract_customer_contexts, extract_customer_utterance
from nlp.prompts import (
    DEFAULT_GATE_HYPOTHESIS,
    DEFAULT_HYPOTHESES,
    DEFAULT_SLOT_DEFINITIONS,
    DEFAULT_SLOT_HYPOTHESES,
    DEFAULT_SLOT_PREDICATES,
    SLOTS_EXTRACTION_PROMPT,
)
from utils.openai_helper import (
    DEFAULT_OPENAI_BATCH_MODEL,
    DEFAULT_OPENAI_DECISION_MODEL,
    build_decision_predicate_questions,
    get_openai_client,
    parse_batch_output_jsonl,
    parse_decision_answers,
    parse_llm_slot_annotations,
    prepare_batch_input_jsonl,
)

logger = logging.getLogger(__name__)

DEFAULT_SLOT_THRESHOLDS: dict[str, float] = {
    "email": 0.98,
    "phone": 0.97,
    "order_id": 0.95,
    "account_id": 0.96,
    "zip_code": 0.85,
    "amount": 0.90,
    "street_address": 0.90,
    "payment_method": 0.90,
}

__all__ = [
    "DEFAULT_GATE_HYPOTHESIS",
    "DEFAULT_HYPOTHESES",
    "DEFAULT_SLOT_DEFINITIONS",
    "DEFAULT_SLOT_HYPOTHESES",
    "DEFAULT_SLOT_PREDICATES",
    "DEFAULT_SLOT_THRESHOLDS",
    "OpenAIBatchSlotExtractor",
    "OpenAIDecisionSlotExtractor",
    "SemanticMatcher",
    "SlotMatcher",
]


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
        hypothesis_template: str
        | dict[str, str]
        | Callable[[str], str] = "The customer specifies their {}.",
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
                idx for idx, ct in enumerate(resolved_cust_texts) if ct and ct.strip()
            ]
            passed_gate_indices: set[int] = set()

            if gate_eval_indices:
                with torch.inference_mode():
                    for batch_indices in batched(gate_eval_indices, batch_size):
                        chunk_premises = [resolved_cust_texts[i] for i in batch_indices]
                        chunk_hyps = [self.gate_hypothesis] * len(chunk_premises)

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


class OpenAIBatchSlotExtractor(SlotMatcher):
    """
    Dialogue State Tracking (Slot Extraction) via OpenAI Batch API.

    Leverages whole-dialogue prompts with structured outputs (Pydantic / JSON Schema)
    to annotate customer turns across conversations asynchronously at a 50% discount rate.
    Defaults to `gpt-6-luna` using the `/v1/responses` endpoint.

    Attributes
    ----------
    model_name : str
        OpenAI model identifier (default: 'gpt-6-luna').
    endpoint : {'/v1/responses', '/v1/chat/completions'}
        Target OpenAI API endpoint for the batch requests.
    convos_per_request : int
        Number of conversations bundled per API request line.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_OPENAI_BATCH_MODEL,
        endpoint: Literal["/v1/responses", "/v1/chat/completions"] = "/v1/responses",
        api_key: str | None = None,
        client: Any | None = None,
        convos_per_request: int = 1,
        prompt: str | None = None,
    ):
        self.model_name = model_name
        self.endpoint = endpoint
        self.convos_per_request = convos_per_request
        self.prompt = prompt or SLOTS_EXTRACTION_PROMPT
        self._api_key = api_key
        self._client = client

    @property
    def client(self) -> Any:
        """Lazy-loaded OpenAI client instance."""
        if self._client is None:
            self._client = get_openai_client(api_key=self._api_key)
        return self._client

    def prepare_batch_file(
        self,
        df_turns: pd.DataFrame,
        convo_ids: list[int | str] | None = None,
        output_path: str | Path | None = None,
        prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Prepare `.jsonl` input lines and optionally save to disk for OpenAI Batch API submission.
        Does NOT require an active OpenAI API key or internet connection.
        """
        return prepare_batch_input_jsonl(
            df_turns=df_turns,
            convo_ids=convo_ids,
            output_path=output_path,
            model=self.model_name,
            endpoint=self.endpoint,
            convos_per_request=self.convos_per_request,
            prompt=prompt or self.prompt,
        )

    def submit_batch(
        self,
        input_file_path: str | Path,
        completion_window: Literal["24h"] = "24h",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """
        Upload input JSONL file and create an asynchronous batch job.

        Parameters
        ----------
        input_file_path : str | Path
            Path to the local `.jsonl` file prepared by `prepare_batch_file`.
        completion_window : {'24h'}, default '24h'
            Target batch processing completion window.
        metadata : dict[str, str], optional
            Arbitrary key-value metadata to attach to the batch.

        Returns
        -------
        str
            The created batch ID (e.g. 'batch_abc123').
        """
        input_p = Path(input_file_path)
        if not input_p.exists():
            raise FileNotFoundError(f"Batch input file not found: {input_p}")

        logger.info("Uploading batch input file '%s' to OpenAI...", input_p)
        with open(input_p, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="batch")

        logger.info(
            "File uploaded successfully (file_id=%s). Creating batch...", file_obj.id
        )
        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint=self.endpoint,
            completion_window=completion_window,
            metadata=metadata
            or {"project": "nlp-dialogue-state-tracking", "model": self.model_name},
        )
        logger.info(
            "Batch created successfully! Batch ID: %s (status=%s)",
            batch.id,
            batch.status,
        )
        return str(batch.id)

    def poll_batch(
        self,
        batch_id: str,
        poll_interval_seconds: int = 30,
        timeout_seconds: int = 86400,
    ) -> Any:
        """
        Poll OpenAI Batch status until completion, cancellation, or failure.

        Parameters
        ----------
        batch_id : str
            The batch identifier returned by `submit_batch`.
        poll_interval_seconds : int, default 30
            Time to sleep between status checks.
        timeout_seconds : int, default 86400 (24h)
            Maximum time to wait before raising TimeoutError.

        Returns
        -------
        Any
            The completed batch object.
        """
        import time

        start_time = time.time()
        logger.info(
            "Polling status for batch '%s' every %ds...",
            batch_id,
            poll_interval_seconds,
        )

        while True:
            batch = self.client.batches.retrieve(batch_id)
            status = getattr(batch, "status", None)
            logger.info(
                "Batch '%s' status: %s (elapsed: %.1fs)",
                batch_id,
                status,
                time.time() - start_time,
            )

            if status in ("completed", "failed", "expired", "cancelled"):
                if status != "completed":
                    raise RuntimeError(
                        f"Batch '{batch_id}' finished with non-successful status: '{status}'. "
                        f"Errors: {getattr(batch, 'errors', None)}"
                    )
                return batch

            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(
                    f"Batch polling timed out after {timeout_seconds} seconds."
                )

            time.sleep(poll_interval_seconds)

    def download_and_parse_results(
        self,
        output_file_id: str | None = None,
        batch_id: str | None = None,
        download_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Download the output file from a completed batch and parse into
        a standardized DataFrame with columns `['convo_id', 'turn_id', 'inferred_slots']`.
        """
        if output_file_id is None:
            if batch_id is None:
                raise ValueError("Must provide either `output_file_id` or `batch_id`.")
            batch = self.client.batches.retrieve(batch_id)
            output_file_id = getattr(batch, "output_file_id", None)
            if not output_file_id:
                raise ValueError(
                    f"Batch '{batch_id}' does not have an output_file_id (status: {getattr(batch, 'status', None)})."
                )

        logger.info("Downloading batch output content (file_id=%s)...", output_file_id)
        file_response = self.client.files.content(output_file_id)
        raw_text = file_response.text

        if download_path:
            dl_p = Path(download_path)
            dl_p.parent.mkdir(parents=True, exist_ok=True)
            dl_p.write_text(raw_text, encoding="utf-8")
            logger.info("Saved raw batch output to '%s'", dl_p)

        return parse_batch_output_jsonl(raw_text)

    def extract_from_conversations(
        self,
        df_turns: pd.DataFrame,
        convo_ids: list[int | str] | None = None,
        batch_input_path: str | Path | None = None,
        batch_output_path: str | Path | None = None,
        poll: bool = True,
        poll_interval_seconds: int = 30,
    ) -> pd.DataFrame | str:
        """
        End-to-end convenience workflow: prepare, submit, poll, and parse results.
        If poll=False, returns the batch ID after submission.
        """
        jsonl_path = batch_input_path or "data/processed/batch_input_openai.jsonl"
        self.prepare_batch_file(df_turns, convo_ids=convo_ids, output_path=jsonl_path)
        batch_id = self.submit_batch(jsonl_path)

        if not poll:
            return batch_id

        self.poll_batch(batch_id, poll_interval_seconds=poll_interval_seconds)
        return self.download_and_parse_results(
            batch_id=batch_id, download_path=batch_output_path
        )

    def extract_slots(
        self,
        text: str,
        available_slots: list[str],
        batch_size: int = 128,
        return_scores: bool = False,
    ) -> list[str] | dict[str, float]:
        """Synchronous slot extraction on a single utterance."""
        return self.extract_batch(
            [text], available_slots, batch_size=batch_size, return_scores=return_scores
        )[0]

    def extract_batch(
        self,
        texts: list[str],
        available_slots: list[str],
        batch_size: int = 128,
        return_scores: bool = False,
    ) -> list[list[str]] | list[dict[str, float]]:
        """
        Synchronous batch slot extraction using `client.responses.create` or `chat.completions.create`.
        """
        import json

        from nlp.prompts import SLOTS_EXTRACTION_PROMPT
        from utils.openai_helper import ConversationSlotExtractionBatch

        results: list[list[str]] | list[dict[str, float]] = [
            {} if return_scores else [] for _ in texts
        ]
        if not texts or not available_slots:
            return results

        mock_payload = {
            str(idx): [{"turn_id": 0, "speaker": "CUSTOMER", "text": t}]
            for idx, t in enumerate(texts)
        }

        json_schema = ConversationSlotExtractionBatch.model_json_schema()
        serialized_input = json.dumps(mock_payload, indent=2)

        if self.endpoint == "/v1/responses":
            resp = self.client.responses.create(
                model=self.model_name,
                instructions=SLOTS_EXTRACTION_PROMPT,
                input=serialized_input,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "slot_extraction_batch",
                        "strict": True,
                        "schema": json_schema,
                    }
                },
            )
            output_items = getattr(resp, "output", [])
            content_text = ""
            for it in output_items:
                for part in getattr(it, "content", []):
                    if hasattr(part, "text"):
                        content_text = part.text
                        break
        else:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SLOTS_EXTRACTION_PROMPT},
                    {"role": "user", "content": serialized_input},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "slot_extraction_batch",
                        "strict": True,
                        "schema": json_schema,
                    },
                },
            )
            content_text = resp.choices[0].message.content or ""

        annotations = parse_llm_slot_annotations(content_text)
        for ann in annotations:
            idx = int(ann.conversation_id)
            if 0 <= idx < len(results):
                active = [s for s in ann.slots if s in available_slots]
                if return_scores:
                    results[idx] = {s: 1.0 for s in active}
                else:
                    results[idx] = active

        return results


class OpenAIDecisionSlotExtractor(SlotMatcher):
    """
    Dialogue State Tracking (Slot Extraction) via OpenAI Decisions API (`client.decisions.create`).

    Uses `gpt-6-luna` to evaluate multiple slot predicates simultaneously in a single
    decision request per customer turn with zero output token fees ($0.10/1M input).

    Attributes
    ----------
    model_name : str
        Target Decisions API model (default: 'gpt-6-luna').
    threshold : float | dict[str, float]
        Confidence cutoff(s) for predicate probability.
    default_threshold : float
        Fallback cutoff for slots not present in a threshold dictionary.
    max_workers : int
        Thread pool concurrency for turn evaluation (default: 8).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_OPENAI_DECISION_MODEL,
        threshold: float | dict[str, float] = 0.50,
        default_threshold: float = 0.50,
        api_key: str | None = None,
        client: Any | None = None,
        max_workers: int = 8,
        custom_instructions: dict[str, str] | None = None,
    ):
        self.model_name = model_name
        self.threshold = threshold
        self.default_threshold = default_threshold
        self.max_workers = max_workers
        self.custom_instructions = custom_instructions or DEFAULT_SLOT_PREDICATES
        self._api_key = api_key
        self._client = client

    @property
    def client(self) -> Any:
        """Lazy-loaded OpenAI client instance."""
        if self._client is None:
            self._client = get_openai_client(api_key=self._api_key)
        return self._client

    def extract_slots(
        self,
        text: str,
        available_slots: list[str],
        batch_size: int = 128,
        return_scores: bool = False,
    ) -> list[str] | dict[str, float]:
        """
        Extract active slots from a single utterance using `client.decisions.create`.
        Evaluates all candidate slots simultaneously as predicate questions.
        """
        if not text or not str(text).strip() or not available_slots:
            return {} if return_scores else []

        questions = build_decision_predicate_questions(
            available_slots=available_slots,
            custom_instructions=self.custom_instructions,
        )

        decision = self.client.decisions.create(
            model=self.model_name,
            input=str(text),
            questions=questions,
        )

        answers = getattr(decision, "answers", [])
        scores = parse_decision_answers(
            answers,
            thresholds=self.threshold,
            default_threshold=self.default_threshold,
        )

        if return_scores:
            return scores
        return sorted(scores.keys())

    def extract_batch(
        self,
        texts: list[str],
        available_slots: list[str],
        batch_size: int = 128,
        return_scores: bool = False,
        max_workers: int | None = None,
    ) -> list[list[str]] | list[dict[str, float]]:
        """
        Extract active slots across multiple text utterances concurrently
        using a thread pool over the Decisions API.
        """
        if not texts:
            return []
        workers = max_workers or self.max_workers

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    self.extract_slots,
                    text,
                    available_slots,
                    batch_size=batch_size,
                    return_scores=return_scores,
                )
                for text in texts
            ]
            results = [f.result() for f in futures]

        return results

    def extract_from_conversations(
        self,
        df_turns: pd.DataFrame,
        available_slots: list[str] | None = None,
        min_prior_agent_utterances: int = 1,
        return_scores: bool = False,
        max_workers: int | None = None,
    ) -> pd.DataFrame:
        """
        Extract contextual dialogue windows and run decision slot extraction
        across all customer turns in `df_turns`. Returns a standardized DataFrame
        with columns `['convo_id', 'turn_id', 'inferred_slots']`.
        """
        all_documents = extract_customer_contexts(
            df_turns,
            min_prior_agent_utterances=min_prior_agent_utterances,
            group_by_convo=False,
        )

        target_slots = available_slots or sorted(self.custom_instructions.keys())
        batch_texts = [
            doc.get_full_document(strip_prior_customer_utterances=True)
            for doc in all_documents
        ]

        logger.info(
            "Running Decisions API slot extraction on %d customer documents (%d slots, workers=%d)...",
            len(batch_texts),
            len(target_slots),
            max_workers or self.max_workers,
        )

        batch_results = self.extract_batch(
            batch_texts,
            available_slots=target_slots,
            return_scores=return_scores,
            max_workers=max_workers,
        )

        records: list[dict[str, Any]] = []
        for doc, res in zip(all_documents, batch_results):
            rec: dict[str, Any] = {
                "convo_id": doc.convo_id,
                "turn_id": doc.turn_id,
                "inferred_slots": sorted(res.keys()) if return_scores else sorted(res),
            }
            if return_scores:
                rec["confidence_scores"] = [
                    float(res[s]) for s in rec["inferred_slots"]
                ]
            records.append(rec)

        df_out = pd.DataFrame(records)
        df_out["convo_id"] = df_out["convo_id"].astype("int64")
        df_out["turn_id"] = df_out["turn_id"].astype("int64")
        return df_out
