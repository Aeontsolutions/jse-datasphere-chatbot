"""Shared helpers for introspecting raw Gemini SDK response objects.

Kept separate from any one call site (agent_v2, document_selector, ...) since
more than one module needs to read how a generation ended without importing
each other and risking a cycle.
"""

from typing import Any, Optional


def extract_finish_reason(response: Any) -> Optional[str]:
    """Bare finish-reason name from a Gemini response, or None if unavailable.

    The SDK yields an enum whose str() is 'FinishReason.MAX_TOKENS', so prefer
    .name. Never raises: this feeds logging, which must not break a request.
    """
    try:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        reason = getattr(candidates[0], "finish_reason", None)
        if reason is None:
            return None
        return getattr(reason, "name", None) or str(reason).rsplit(".", 1)[-1]
    except Exception:  # pragma: no cover - defensive
        return None
