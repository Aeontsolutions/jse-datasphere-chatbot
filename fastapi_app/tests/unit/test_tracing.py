"""Unit tests for the Langfuse tracing module."""

from unittest.mock import MagicMock, patch

import pytest

import app.tracing as tracing


@pytest.fixture(autouse=True)
def reset_client():
    tracing.reset_langfuse_client()
    yield
    tracing.reset_langfuse_client()


def test_get_langfuse_client_none_when_unconfigured():
    mock_config = MagicMock()
    mock_config.langfuse.configured = False
    with patch("app.config.get_config", return_value=mock_config):
        assert tracing.get_langfuse_client() is None


def test_get_langfuse_client_returns_client_when_configured():
    mock_config = MagicMock()
    mock_config.langfuse.configured = True
    mock_config.langfuse.public_key = "pub"
    mock_config.langfuse.secret_key = "sec"
    mock_config.langfuse.host = "https://cloud.langfuse.com"

    mock_instance = MagicMock()
    with patch("app.config.get_config", return_value=mock_config):
        with patch("langfuse.Langfuse", return_value=mock_instance) as mock_ctor:
            client = tracing.get_langfuse_client()
    assert client is mock_instance
    mock_ctor.assert_called_once_with(
        public_key="pub", secret_key="sec", host="https://cloud.langfuse.com"
    )


def test_get_langfuse_client_is_cached():
    mock_config = MagicMock()
    mock_config.langfuse.configured = True
    mock_config.langfuse.public_key = "pub"
    mock_config.langfuse.secret_key = "sec"
    mock_config.langfuse.host = "https://cloud.langfuse.com"

    with patch("app.config.get_config", return_value=mock_config):
        with patch("langfuse.Langfuse", return_value=MagicMock()) as mock_ctor:
            tracing.get_langfuse_client()
            tracing.get_langfuse_client()
    assert mock_ctor.call_count == 1


def test_get_langfuse_client_none_on_init_failure():
    mock_config = MagicMock()
    mock_config.langfuse.configured = True
    mock_config.langfuse.public_key = "pub"
    mock_config.langfuse.secret_key = "sec"
    mock_config.langfuse.host = "https://cloud.langfuse.com"

    with patch("app.config.get_config", return_value=mock_config):
        with patch("langfuse.Langfuse", side_effect=Exception("boom")):
            assert tracing.get_langfuse_client() is None


def test_traced_observation_yields_none_when_unconfigured():
    with patch("app.tracing.get_langfuse_client", return_value=None):
        with tracing.traced_observation("test-span") as obs:
            assert obs is None


def test_traced_observation_yields_observation_when_configured():
    mock_obs = MagicMock()
    mock_client = MagicMock()
    mock_client.start_as_current_observation.return_value.__enter__.return_value = mock_obs
    mock_client.start_as_current_observation.return_value.__exit__.return_value = False

    with patch("app.tracing.get_langfuse_client", return_value=mock_client):
        with tracing.traced_observation(
            "test-span", as_type="generation", model="gemini-2.5-flash"
        ) as obs:
            assert obs is mock_obs

    mock_client.start_as_current_observation.assert_called_once_with(
        name="test-span", as_type="generation", model="gemini-2.5-flash"
    )


def test_traced_observation_swallows_client_errors():
    mock_client = MagicMock()
    mock_client.start_as_current_observation.side_effect = Exception("boom")

    with patch("app.tracing.get_langfuse_client", return_value=mock_client):
        with tracing.traced_observation("test-span") as obs:
            assert obs is None


def test_log_completed_generation_noop_when_unconfigured():
    with patch("app.tracing.get_langfuse_client", return_value=None):
        tracing.log_completed_generation("phase", model="gemini-2.5-flash")  # must not raise


def test_log_completed_generation_creates_updates_and_ends():
    mock_obs = MagicMock()
    mock_client = MagicMock()
    mock_client.start_observation.return_value = mock_obs

    with patch("app.tracing.get_langfuse_client", return_value=mock_client):
        tracing.log_completed_generation(
            "phase",
            model="gemini-2.5-flash",
            input="prompt",
            output="answer",
            usage_details={"input": 10, "output": 5},
        )

    mock_client.start_observation.assert_called_once_with(
        name="phase", as_type="generation", model="gemini-2.5-flash", input="prompt"
    )
    mock_obs.update.assert_called_once_with(
        output="answer", usage_details={"input": 10, "output": 5}
    )
    mock_obs.end.assert_called_once()


def test_log_completed_generation_swallows_client_errors():
    mock_client = MagicMock()
    mock_client.start_observation.side_effect = Exception("boom")

    with patch("app.tracing.get_langfuse_client", return_value=mock_client):
        tracing.log_completed_generation("phase")  # must not raise
