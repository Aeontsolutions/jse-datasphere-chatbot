"""
Langfuse tracing for observability/cost dashboards only — not used for
caching (Langfuse doesn't do response caching) or as the permanent
interaction-log store (see app/interaction_log.py for that).

The installed Langfuse SDK (v3+) is fully OpenTelemetry-based: there is no
`.trace()`/`.generation()` API. Observations are created via
`client.start_as_current_observation(...)`, which sets the new observation as
the "current" one in OTel context for the duration of a `with` block — any
observation started inside that block (even from a different module, deeper
in the call stack) automatically nests under it. That means Gemini call
sites (financial_utils.py, agent_v2.py) don't need a `trace` object threaded
through function signatures: they just call `traced_observation(...)`
themselves, and it nests correctly whenever called from within a request's
outer span opened in main.py.

Google GenAI has no Langfuse auto-instrumentation, so every call site wraps
its own `generate_content(...)` call manually.
"""

from contextlib import contextmanager
from typing import Any, Optional

from app.logging_config import get_logger

logger = get_logger(__name__)

_client: Optional[Any] = None
_resolved = False


def get_langfuse_client() -> Optional[Any]:
    """Return a singleton Langfuse client, or None if tracing isn't configured.

    No-ops safely (returns None) whenever LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY
    are unset or LANGFUSE_ENABLED=false — matches the app's default test env,
    which sets neither, so tracing is a pure no-op in the test suite.
    """
    global _client, _resolved
    if not _resolved:
        from app.config import get_config

        cfg = get_config().langfuse
        if cfg.configured:
            try:
                from langfuse import Langfuse

                _client = Langfuse(
                    public_key=cfg.public_key, secret_key=cfg.secret_key, host=cfg.host
                )
            except Exception as exc:
                logger.warning("langfuse_init_failed", extra={"error": str(exc)})
                _client = None
        _resolved = True
    return _client


def reset_langfuse_client() -> None:
    """Reset the cached client. Test-only — production never needs to reconfigure."""
    global _client, _resolved
    _client = None
    _resolved = False


def log_completed_generation(
    name: str,
    model: Optional[str] = None,
    input: Optional[Any] = None,
    output: Optional[Any] = None,
    usage_details: Optional[dict] = None,
) -> None:
    """Log a generation that has already completed (e.g. from a cost-tracking
    choke point that only sees the response after the call). Unlike
    `traced_observation`, there's no `with` block to wrap around a call that
    already happened — this creates, updates, and ends the observation in one
    shot. No-ops when tracing isn't configured.
    """
    client = get_langfuse_client()
    if client is None:
        return
    try:
        obs = client.start_observation(name=name, as_type="generation", model=model, input=input)
        obs.update(output=output, usage_details=usage_details)
        obs.end()
    except Exception as exc:
        logger.warning("langfuse_log_generation_failed", extra={"name": name, "error": str(exc)})


@contextmanager
def traced_observation(name: str, as_type: str = "span", **kwargs: Any):
    """Context manager yielding a Langfuse observation, or None when tracing
    is unconfigured. Callers must guard `.update(...)` calls on `is not None`.
    """
    client = get_langfuse_client()
    if client is None:
        yield None
        return
    try:
        with client.start_as_current_observation(name=name, as_type=as_type, **kwargs) as obs:
            yield obs
    except Exception as exc:
        # Tracing must never break the request it's observing.
        logger.warning("langfuse_observation_failed", extra={"name": name, "error": str(exc)})
        yield None
