"""
NLP utilities package for dialogue processing, context extraction, and slot tracking.
"""

from nlp.context import (
    DialogueTurnContext,
    extract_customer_contexts,
    extract_customer_utterance,
    get_customer_turn_context,
)
from nlp.prompts import (
    DEFAULT_GATE_HYPOTHESIS,
    DEFAULT_SLOT_DEFINITIONS,
    DEFAULT_SLOT_HYPOTHESES,
    DEFAULT_SLOT_PREDICATES,
    SLOTS_EXTRACTION_PROMPT,
    build_slots_extraction_prompt,
)
from nlp.slots import (
    DEFAULT_SLOT_THRESHOLDS,
    OpenAIBatchSlotExtractor,
    OpenAIDecisionSlotExtractor,
    SemanticMatcher,
    SlotMatcher,
)

__all__ = [
    "DEFAULT_GATE_HYPOTHESIS",
    "DEFAULT_SLOT_DEFINITIONS",
    "DEFAULT_SLOT_HYPOTHESES",
    "DEFAULT_SLOT_PREDICATES",
    "DEFAULT_SLOT_THRESHOLDS",
    "SLOTS_EXTRACTION_PROMPT",
    "DialogueTurnContext",
    "OpenAIBatchSlotExtractor",
    "OpenAIDecisionSlotExtractor",
    "SemanticMatcher",
    "SlotMatcher",
    "build_slots_extraction_prompt",
    "extract_customer_contexts",
    "extract_customer_utterance",
    "get_customer_turn_context",
]
