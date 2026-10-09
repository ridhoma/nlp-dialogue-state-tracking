"""
NLP utilities package for dialogue processing, context extraction, and slot tracking.
"""

from nlp.context import (
    DialogueTurnContext,
    extract_customer_contexts,
    extract_customer_utterance,
    get_customer_turn_context,
)
from nlp.slots import (
    DEFAULT_GATE_HYPOTHESIS,
    DEFAULT_SLOT_HYPOTHESES,
    DEFAULT_SLOT_THRESHOLDS,
    SemanticMatcher,
    SlotMatcher,
)

__all__ = [
    "DEFAULT_GATE_HYPOTHESIS",
    "DEFAULT_SLOT_HYPOTHESES",
    "DEFAULT_SLOT_THRESHOLDS",
    "DialogueTurnContext",
    "SemanticMatcher",
    "SlotMatcher",
    "extract_customer_contexts",
    "extract_customer_utterance",
    "get_customer_turn_context",
]

