"""Writers for manifest.json, summary.json, and per-conversation JSON files."""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.judge import JudgeOutput
from evals.persona import PersonaSpec
from evals.runner import ConversationArtifact, RunArtifacts
from evals.transcript import ChatTurn, TerminationReason, Transcript


def write_run(
    artifacts: RunArtifacts,
    personas: list[PersonaSpec],
    config: dict[str, Any],
    run_id: str,
    git_sha: str | None,
    output_root: Path,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> Path:
    """Write manifest.json, summary.json, and per-conversation JSON files."""
    run_dir = output_root / run_id
    (run_dir / "conversations").mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "run_id": run_id,
        "started_at": started_at or now,
        "ended_at": ended_at or now,
        "git_sha": git_sha,
        "config": config,
        "personas_run": [p.id for p in personas],
        "replicates": _detect_replicates(artifacts),
        "cost_capped": artifacts.cost_capped,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    persona_by_id = {p.id: p for p in personas}
    for c in artifacts.conversations:
        persona = persona_by_id.get(c.transcript.persona_id)
        path = run_dir / "conversations" / f"{c.transcript.conversation_id}.json"
        if path.exists():
            continue
        path.write_text(
            json.dumps(_convo_payload(c, persona), indent=2),
            encoding="utf-8",
        )

    summary = _summarize(artifacts, personas, run_id, manifest)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return run_dir


def write_initial_manifest(
    run_dir: Path,
    run_id: str,
    git_sha: str | None,
    config: dict[str, Any],
    personas: list[PersonaSpec],
    started_at: str,
) -> None:
    """Write manifest.json with status 'in_progress' before the run starts."""
    manifest = {
        "run_id": run_id,
        "status": "in_progress",
        "started_at": started_at,
        "git_sha": git_sha,
        "config": config,
        "personas_run": [p.id for p in personas],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_conversation(
    run_dir: Path,
    artifact: ConversationArtifact,
    persona: PersonaSpec | None,
) -> None:
    """Write a single conversation artifact to conversations/<id>.json."""
    path = run_dir / "conversations" / f"{artifact.transcript.conversation_id}.json"
    path.write_text(json.dumps(_convo_payload(artifact, persona), indent=2), encoding="utf-8")


def load_conversation_artifacts(
    run_dir: Path,
) -> tuple[list[ConversationArtifact], set[str]]:
    """Load completed conversation artifacts from a partial or crashed run."""
    convos_dir = run_dir / "conversations"
    artifacts: list[ConversationArtifact] = []
    ids: set[str] = set()
    if not convos_dir.exists():
        return artifacts, ids
    for path in sorted(convos_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            artifacts.append(_artifact_from_payload(payload))
            ids.add(payload["conversation_id"])
        except Exception as exc:
            print(f"WARNING: skipping unreadable conversation file {path.name}: {exc}", file=sys.stderr)
    return artifacts, ids


def _artifact_from_payload(payload: dict[str, Any]) -> ConversationArtifact:
    persona_data = payload.get("persona") or {}
    persona_id = persona_data.get("id") or payload["conversation_id"].rsplit("__rep", 1)[0]
    rep_str = payload["conversation_id"].rsplit("__rep", 1)[-1]
    replicate_index = int(rep_str) - 1

    transcript = Transcript(
        conversation_id=payload["conversation_id"],
        persona_id=persona_id,
        replicate_index=replicate_index,
        endpoint=payload["endpoint"],
        turns=[ChatTurn.model_validate(t) for t in payload["turns"]],
        termination=TerminationReason.model_validate(payload["termination"]),
    )

    judge_data = payload.get("judge")
    if judge_data is None:
        return ConversationArtifact(transcript, None, False, None)
    if judge_data.get("judge_failed"):
        return ConversationArtifact(transcript, None, True, judge_data.get("error"))
    return ConversationArtifact(transcript, JudgeOutput.model_validate(judge_data), False, None)


def _detect_replicates(artifacts: RunArtifacts) -> int:
    if not artifacts.conversations:
        return 0
    by_persona: dict[str, int] = {}
    for c in artifacts.conversations:
        by_persona.setdefault(c.transcript.persona_id, 0)
        by_persona[c.transcript.persona_id] += 1
    return max(by_persona.values())


def _convo_payload(c: ConversationArtifact, persona: PersonaSpec | None) -> dict[str, Any]:
    t = c.transcript
    payload = {
        "conversation_id": t.conversation_id,
        "persona": persona.model_dump() if persona else None,
        "endpoint": t.endpoint,
        "turns": [turn.model_dump() for turn in t.turns],
        "termination": t.termination.model_dump(),
        "totals": t.totals(),
    }
    if c.judge_failed:
        payload["judge"] = {"judge_failed": True, "error": c.judge_error}
    elif c.judge_output is not None:
        payload["judge"] = c.judge_output.model_dump()
    else:
        payload["judge"] = None
    payload["errors"] = []
    return payload


def _summarize(
    artifacts: RunArtifacts,
    personas: list[PersonaSpec],
    run_id: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    convos = artifacts.conversations
    complete = [c for c in convos if c.transcript.termination.reason != "error"]
    incomplete = [c for c in convos if c.transcript.termination.reason == "error"]

    by_persona: dict[str, dict[str, Any]] = {}
    for p in personas:
        ps = [c for c in complete if c.transcript.persona_id == p.id]
        by_persona[p.id] = _persona_stats(ps)

    incomplete_by_persona: dict[str, dict[str, Any]] = {}
    for p in personas:
        errs = [c for c in incomplete if c.transcript.persona_id == p.id]
        if errs:
            incomplete_by_persona[p.id] = {"incomplete_count": len(errs)}

    by_endpoint = {
        endpoint: _persona_stats([c for c in complete if c.transcript.endpoint == endpoint])
        for endpoint in {c.transcript.endpoint for c in complete}
    }

    by_category = {
        cat: _persona_stats(
            [c for c in complete if _category_for(c.transcript.persona_id, personas) == cat]
        )
        for cat in {p.category for p in personas}
    }

    incomplete_cost = sum(c.transcript.totals()["cost_usd"] for c in incomplete)
    overall = _overall_stats(complete, len(incomplete), incomplete_cost)

    return {
        "run_id": run_id,
        "started_at": manifest["started_at"],
        "ended_at": manifest["ended_at"],
        "git_sha": manifest["git_sha"],
        "config": manifest["config"],
        "conversation_count": len(convos),
        "by_persona": by_persona,
        "incomplete_by_persona": incomplete_by_persona,
        "by_endpoint": by_endpoint,
        "by_category": by_category,
        "overall": overall,
    }


def _category_for(persona_id: str, personas: list[PersonaSpec]) -> str:
    for p in personas:
        if p.id == persona_id:
            return p.category
    return "unknown"


def _persona_stats(convos: list[ConversationArtifact]) -> dict[str, Any]:
    if not convos:
        return {"count": 0}

    judged = [c for c in convos if c.judge_output is not None]
    if not judged:
        return {"count": len(convos), "judged_count": 0}

    def dim(field: str) -> list[float]:
        out: list[float] = []
        for c in judged:
            s = getattr(c.judge_output.scores, field)
            if s.score is not None:
                out.append(float(s.score))
        return out

    def mean_std(field: str) -> tuple[float | None, float | None]:
        values = dim(field)
        if not values:
            return None, None
        m = statistics.mean(values)
        s = statistics.stdev(values) if len(values) > 1 else 0.0
        return m, s

    out: dict[str, Any] = {"count": len(convos), "judged_count": len(judged)}
    for d in (
        "groundedness",
        "factfulness",
        "goal_completion",
        "tool_use_appropriateness",
        "coherence",
        "persona_handling",
    ):
        m, s = mean_std(d)
        out[f"mean_{d}"] = m
        out[f"std_{d}"] = s

    verdict_counts: dict[str, int] = {"pass": 0, "partial": 0, "fail": 0}
    for c in judged:
        verdict_counts[c.judge_output.verdict] = verdict_counts.get(c.judge_output.verdict, 0) + 1
    out["verdict_counts"] = verdict_counts

    out["mean_turns"] = statistics.mean(len(c.transcript.turns) for c in convos)
    out["mean_latency_ms"] = statistics.mean(c.transcript.totals()["latency_ms"] for c in convos)
    out["total_cost_usd"] = sum(c.transcript.totals()["cost_usd"] for c in convos)
    return out


def _overall_stats(convos: list[ConversationArtifact], incomplete_count: int = 0, incomplete_cost: float = 0.0) -> dict[str, Any]:
    if not convos:
        return {"incomplete_count": incomplete_count}

    judged = [c for c in convos if c.judge_output is not None]

    def values(field: str) -> list[float]:
        return [
            float(getattr(c.judge_output.scores, field).score)
            for c in judged
            if getattr(c.judge_output.scores, field).score is not None
        ]

    overall: dict[str, Any] = {}
    for d in (
        "groundedness",
        "factfulness",
        "goal_completion",
        "tool_use_appropriateness",
        "coherence",
        "persona_handling",
    ):
        v = values(d)
        overall[f"mean_{d}"] = statistics.mean(v) if v else None
        overall[f"std_{d}"] = statistics.stdev(v) if len(v) > 1 else (0.0 if v else None)

    verdict_counts = {"pass": 0, "partial": 0, "fail": 0}
    for c in judged:
        verdict_counts[c.judge_output.verdict] = verdict_counts.get(c.judge_output.verdict, 0) + 1
    overall["verdict_counts"] = verdict_counts

    overall["judge_failed_count"] = sum(1 for c in convos if c.judge_failed)
    overall["incomplete_count"] = incomplete_count
    overall["mean_turns"] = statistics.mean(len(c.transcript.turns) for c in convos)
    overall["mean_latency_ms"] = statistics.mean(
        c.transcript.totals()["latency_ms"] for c in convos
    )
    overall["total_cost_usd"] = sum(c.transcript.totals()["cost_usd"] for c in convos) + incomplete_cost
    return overall
