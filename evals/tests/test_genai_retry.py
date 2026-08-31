"""Tests for the Gemini SDK retry helper.

The failure this exists for: on 2026-08-31 the CI eval gate failed with
`2 judge_failed_count (max allowed 0)` while every quality dimension improved.
Both failures were `ServerError: 503 UNAVAILABLE -- "This model is currently
experiencing high demand. Spikes in demand are usually temporary. Please try
again later."` Gemini was asking to be retried and nothing retried it, because
judge.py called generate_content outside its try block.
"""

from __future__ import annotations

import pytest
from google.genai import errors

from evals._genai_retry import call_with_retry


def _server_error(code: int = 503) -> errors.ServerError:
    return errors.ServerError(code, {"error": {"message": "high demand", "status": "UNAVAILABLE"}})


def _client_error(code: int) -> errors.ClientError:
    return errors.ClientError(code, {"error": {"message": f"status {code}"}})


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    """Record backoff delays instead of waiting them out."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("evals._genai_retry.asyncio.sleep", fake_sleep)
    return slept


async def test_returns_value_without_retrying_on_success():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        return "judged"

    assert await call_with_retry(fn) == "judged"
    assert calls == 1


async def test_retries_transient_503_then_succeeds():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _server_error(503)
        return "judged"

    assert await call_with_retry(fn) == "judged"
    assert calls == 2


async def test_retries_rate_limit_429():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _client_error(429)
        return "judged"

    assert await call_with_retry(fn) == "judged"
    assert calls == 2


async def test_gives_up_after_max_retries_and_reraises():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise _server_error(503)

    with pytest.raises(errors.ServerError):
        await call_with_retry(fn, max_retries=2)
    assert calls == 3  # one initial attempt plus two retries


async def test_does_not_retry_a_bad_request():
    # A malformed prompt is not going to fix itself. Retrying burns budget and
    # delays the real error.
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise _client_error(400)

    with pytest.raises(errors.ClientError):
        await call_with_retry(fn)
    assert calls == 1


async def test_does_not_retry_auth_failure():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise _client_error(403)

    with pytest.raises(errors.ClientError):
        await call_with_retry(fn)
    assert calls == 1


async def test_does_not_retry_unrelated_exception():
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        raise ValueError("bug in our own code")

    with pytest.raises(ValueError):
        await call_with_retry(fn)
    assert calls == 1


async def test_backoff_grows_between_attempts(_no_real_sleeping):
    async def fn():
        raise _server_error(503)

    with pytest.raises(errors.ServerError):
        await call_with_retry(fn, max_retries=2)

    assert len(_no_real_sleeping) == 2
    assert _no_real_sleeping[1] > _no_real_sleeping[0]


async def test_logs_each_retry_with_label(capsys, _no_real_sleeping):
    calls = 0

    async def fn():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _server_error(503)
        return "judged"

    await call_with_retry(fn, label="judge:ncb_revenue")
    out = capsys.readouterr().out
    assert "judge:ncb_revenue" in out
    assert "503" in out
