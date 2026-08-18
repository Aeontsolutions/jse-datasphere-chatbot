"""Tests for the eval-suite CLI argparse layer."""

import pytest

from evals.cli import build_arg_parser, financial_tool_coverage, parse_args_to_overrides
from evals.persona import PersonaSpec


def test_parser_accepts_all_documented_flags():
    parser = build_arg_parser()
    ns = parser.parse_args(
        [
            "--base-url", "http://x:9000",
            "--persona", "a",
            "--persona", "b",
            "--category", "positive",
            "--endpoint", "fast_chat_v2",
            "--replicates", "2",
            "--concurrency", "8",
            "--max-cost-usd", "3.5",
            "--run-id", "smoke",
            "--request-timeout-s", "45",
        ]
    )
    assert ns.base_url == "http://x:9000"
    assert ns.personas == ["a", "b"]
    assert ns.category == "positive"
    assert ns.endpoint == "fast_chat_v2"
    assert ns.replicates == 2
    assert ns.concurrency == 8
    assert ns.max_cost_usd == 3.5
    assert ns.run_id == "smoke"
    assert ns.request_timeout_s == 45.0


def test_overrides_skip_none_values():
    parser = build_arg_parser()
    ns = parser.parse_args(["--replicates", "1"])
    overrides = parse_args_to_overrides(ns)
    assert overrides["replicates"] == 1
    assert "base_url" not in overrides
    assert "concurrency" not in overrides


def test_invalid_endpoint_rejected():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--endpoint", "bogus"])


def test_parser_accepts_resume_flag():
    parser = build_arg_parser()
    ns = parser.parse_args(["--resume", "2026-06-01T10-00-00"])
    assert ns.resume_run_id == "2026-06-01T10-00-00"


def test_resume_flag_defaults_to_none():
    parser = build_arg_parser()
    ns = parser.parse_args([])
    assert ns.resume_run_id is None


# ---------------------------------------------------------------------------
# financial_tool_coverage — surface how thinly a filtered persona set
# exercises the financial DB tool, instead of leaving it a silent gap.
# ---------------------------------------------------------------------------


def _persona(endpoint: str, enable_financial_data: bool = True) -> PersonaSpec:
    return PersonaSpec(
        id=f"p_{endpoint}_{enable_financial_data}",
        name="P",
        category="positive",
        endpoint=endpoint,
        character="X",
        goal="Y",
        max_turns=3,
        api_options={"enable_financial_data": enable_financial_data} if endpoint == "chat_stream" else {},
    )


def test_coverage_counts_fast_chat_v2_as_always_capable():
    personas = [_persona("fast_chat_v2"), _persona("fast_chat_v2")]
    assert financial_tool_coverage(personas) == (2, 2)


def test_coverage_counts_chat_stream_only_when_financial_data_enabled():
    personas = [
        _persona("chat_stream", enable_financial_data=True),
        _persona("chat_stream", enable_financial_data=False),
    ]
    assert financial_tool_coverage(personas) == (1, 2)


def test_coverage_empty_list():
    assert financial_tool_coverage([]) == (0, 0)
