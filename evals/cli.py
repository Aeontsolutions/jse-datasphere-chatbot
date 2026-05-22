"""CLI entrypoint for the eval suite (`python -m evals.cli`)."""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google import genai

from evals.client.agent_stream import AgentStreamClient
from evals.client.base import ChatClient
from evals.client.financial import FinancialClient
from evals.config import load_config
from evals.judge import Judge
from evals.persona import PersonaSpec, load_personas
from evals.persona_actor import PersonaActor
from evals.report import write_run
from evals.runner import run_simulation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evals", description="Simulation eval suite")
    parser.add_argument("--base-url")
    parser.add_argument("--persona", action="append", dest="personas", default=[])
    parser.add_argument("--category", choices=["positive", "negative"])
    parser.add_argument("--endpoint", choices=["fast_chat_v2", "chat_stream"])
    parser.add_argument("--replicates", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--max-cost-usd", type=float, dest="max_cost_usd")
    parser.add_argument("--request-timeout-s", type=float, dest="request_timeout_s")
    parser.add_argument("--config", dest="config_path", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--personas-dir",
        default=str(Path(__file__).parent / "personas"),
        help="Directory of persona YAML files",
    )
    return parser


def parse_args_to_overrides(ns: argparse.Namespace) -> dict[str, Any]:
    """Convert CLI namespace into config override dict, skipping unset values."""
    mapping = {
        "base_url": ns.base_url,
        "replicates": ns.replicates,
        "concurrency": ns.concurrency,
        "max_cost_usd_per_run": ns.max_cost_usd,
        "request_timeout_s": ns.request_timeout_s,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _filter_personas(
    all_personas: list[PersonaSpec],
    ids: list[str],
    category: str | None,
    endpoint: str | None,
) -> list[PersonaSpec]:
    out = all_personas
    if ids:
        out = [p for p in out if p.id in ids]
    if category:
        out = [p for p in out if p.category == category]
    if endpoint:
        out = [p for p in out if p.endpoint == endpoint]
    return out


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


async def _amain(ns: argparse.Namespace) -> int:
    overrides = parse_args_to_overrides(ns)
    config = load_config(path=ns.config_path, overrides=overrides)

    api_key = os.environ.get(config.gemini_api_key_env)
    if not api_key:
        print(f"ERROR: env var {config.gemini_api_key_env} not set")
        return 2

    personas = _filter_personas(
        load_personas(ns.personas_dir),
        ids=ns.personas,
        category=ns.category,
        endpoint=ns.endpoint,
    )
    if not personas:
        print("ERROR: no personas matched the filters")
        return 2

    genai_client = genai.Client(api_key=api_key)
    persona_actor = PersonaActor(
        client=genai_client,
        model=config.persona_model,
        temperature=config.persona_temperature,
    )
    judge = Judge(
        client=genai_client,
        model=config.judge_model,
        temperature=config.judge_temperature,
    )

    def client_factory(endpoint: str) -> ChatClient:
        if endpoint == "fast_chat_v2":
            return FinancialClient(base_url=config.base_url, timeout_s=config.request_timeout_s)
        return AgentStreamClient(base_url=config.base_url, timeout_s=config.request_timeout_s)

    started_at = datetime.now(timezone.utc).isoformat()
    print(
        f"Running {len(personas)} persona(s) × {config.replicates} replicate(s) "
        f"= {len(personas) * config.replicates} conversation(s)..."
    )
    artifacts = await run_simulation(
        personas=personas,
        replicates=config.replicates,
        concurrency=config.concurrency,
        max_cost_usd_per_run=config.max_cost_usd_per_run,
        max_cost_usd_per_conversation=config.max_cost_usd_per_conversation,
        chat_client_factory=client_factory,
        persona_actor=persona_actor,
        judge=judge,
    )
    ended_at = datetime.now(timezone.utc).isoformat()

    run_id = ns.run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    output_root = Path(ns.output_dir) if ns.output_dir else Path(__file__).parent / "runs"
    run_dir = write_run(
        artifacts=artifacts,
        personas=personas,
        config=config.model_dump(),
        run_id=run_id,
        git_sha=_git_sha(),
        output_root=output_root,
        started_at=started_at,
        ended_at=ended_at,
    )
    print(f"Wrote run to {run_dir}")
    if artifacts.cost_capped:
        print("WARNING: cost cap reached; run is partial")
    return 0


def main() -> int:
    ns = build_arg_parser().parse_args()
    return asyncio.run(_amain(ns))


if __name__ == "__main__":
    raise SystemExit(main())
