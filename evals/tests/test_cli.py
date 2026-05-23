"""Tests for the eval-suite CLI argparse layer."""

import pytest

from evals.cli import build_arg_parser, parse_args_to_overrides


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
