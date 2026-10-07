"""
Resolution Determinator logic as specified in ANALYSIS_PLAN.md (§2).

Objective:
Map conversation action sequences and terminal states to a binary target variable Y:
    Y = 1: Positive resolution (canonical resolving action executed, no escalation)
    Y = 0: Negative resolution (escalation to supervisor/team, premature termination,
           unresolved policy block)
"""


def evaluate_conversation_resolution(
    actions_taken: list[str],
    canonical_actions: list[str],
    escalation_actions: list[str] | None = None,
) -> tuple[int, dict]:
    """
    Evaluates whether a conversation achieved positive resolution (Y=1) or negative (Y=0).

    Parameters
    ----------
    actions_taken : list[str]
        Ordered list of system action names executed during the conversation.
    canonical_actions : list[str]
        Canonical SOP sequence of actions defined for this intent in kb.json / guidelines.json.
    escalation_actions : list[str], optional
        List of actions designated as escalations / fallbacks. Defaults to ['notify-team'].

    Returns
    -------
    resolved : int
        1 if positive resolution, 0 if negative resolution.
    diagnostics : dict
        Detailed boolean flags and strings explaining the resolution verdict:
        - 'canonical_terminal': expected final action
        - 'actual_terminal': observed final action
        - 'reached_canonical_terminal': bool
        - 'has_escalation': bool
        - 'num_actions_taken': int
        - 'expected_num_actions': int
    """
    if escalation_actions is None:
        escalation_actions = ["notify-team"]

    canonical_terminal = canonical_actions[-1] if canonical_actions else None
    actual_terminal = actions_taken[-1] if actions_taken else None

    has_escalation = any(act in escalation_actions for act in actions_taken)
    reached_canonical_terminal = (
        actual_terminal is not None
        and canonical_terminal is not None
        and actual_terminal == canonical_terminal
    )

    resolved = 1 if (reached_canonical_terminal and not has_escalation) else 0

    diagnostics = {
        "canonical_terminal": canonical_terminal,
        "actual_terminal": actual_terminal,
        "reached_canonical_terminal": reached_canonical_terminal,
        "has_escalation": has_escalation,
        "num_actions_taken": len(actions_taken),
        "expected_num_actions": len(canonical_actions),
    }

    return resolved, diagnostics
