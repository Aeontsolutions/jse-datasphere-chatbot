"""Tests for the report writer."""

import json
from pathlib import Path

import pytest

from evals.judge import (
    DimensionScore, FactfulnessScore, JudgeOutput, JudgeScores, ToolUseScore,
)
from evals.persona import PersonaSpec
from evals.report import write_run
from evals.runner import ConversationArtifact, RunArtifacts
from evals.transcript import ChatTurn, TerminationReason, Transcript


def _persona(id: str = "p1") -> PersonaSpec:
    return PersonaSpec(
        id=id,
        name=id.upper(),
        category="positive",
        endpoint="fast_chat_v2",
        character="X",
        goal="Y",
        max_turns=3,
        expected_facts=["fact one"],
    )


def _transcript(persona_id: str, rep: int, cost: float = 0.001) -> Transcript:
    return Transcript(
        conversation_id=f"{persona_id}__rep{rep + 1:02d}",
        persona_id=persona_id,
        replicate_index=rep,
        endpoint="fast_chat_v2",
        turns=[
            ChatTurn(
                turn_index=0,
                persona_utterance="q",
                chatbot_text="a",
                chatbot_metadata={"data_found": True},
                latency_ms=500,
                cost_usd=cost,
                input_tokens=100,
                output_tokens=40,
            )
        ],
        termination=TerminationReason(reason="done", at_turn=0),
    )


def _judge(score: int = 4, verdict: str = "pass") -> JudgeOutput:
    return JudgeOutput(
        scores=JudgeScores(
            groundedness=DimensionScore(score=score, justification="g"),
            factfulness=FactfulnessScore(score=score, facts_satisfied=[True], justification="f"),
            goal_completion=DimensionScore(score=score, justification="gc"),
            tool_use_appropriateness=ToolUseScore(score=score, justification="t"),
            coherence=DimensionScore(score=score, justification="c"),
            persona_handling=DimensionScore(score=score, justification="ph"),
        ),
        verdict=verdict,
        verdict_reason="r",
    )


def test_write_run_creates_expected_files(tmp_path: Path):
    persona = _persona()
    artifacts = RunArtifacts(
        conversations=[
            ConversationArtifact(_transcript("p1", 0), _judge(), False, None),
            ConversationArtifact(_transcript("p1", 1), _judge(score=3, verdict="partial"), False, None),
        ],
        cost_capped=False,
    )
    run_dir = write_run(
        artifacts=artifacts,
        personas=[persona],
        config={"base_url": "http://localhost:8000"},
        run_id="r1",
        git_sha="abc123",
        output_root=tmp_path,
    )
    assert run_dir == tmp_path / "r1"
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "summary.json").exists()
    convo_files = list((run_dir / "conversations").glob("*.json"))
    assert len(convo_files) == 2


def test_summary_aggregates_per_persona(tmp_path: Path):
    persona = _persona()
    artifacts = RunArtifacts(
        conversations=[
            ConversationArtifact(_transcript("p1", 0), _judge(score=4), False, None),
            ConversationArtifact(_transcript("p1", 1), _judge(score=5), False, None),
            ConversationArtifact(_transcript("p1", 2), _judge(score=3, verdict="partial"), False, None),
        ],
        cost_capped=False,
    )
    run_dir = write_run(
        artifacts=artifacts,
        personas=[persona],
        config={"base_url": "http://localhost:8000"},
        run_id="r2",
        git_sha=None,
        output_root=tmp_path,
    )
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["conversation_count"] == 3
    p1 = summary["by_persona"]["p1"]
    assert p1["mean_groundedness"] == pytest.approx(4.0)
    assert p1["std_groundedness"] >= 0
    assert summary["overall"]["verdict_counts"]["pass"] == 2
    assert summary["overall"]["verdict_counts"]["partial"] == 1


def test_manifest_captures_cost_cap_flag(tmp_path: Path):
    persona = _persona()
    artifacts = RunArtifacts(
        conversations=[ConversationArtifact(_transcript("p1", 0), _judge(), False, None)],
        cost_capped=True,
    )
    run_dir = write_run(
        artifacts=artifacts,
        personas=[persona],
        config={},
        run_id="r3",
        git_sha=None,
        output_root=tmp_path,
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["cost_capped"] is True


def test_conversation_json_inlines_persona_and_judge(tmp_path: Path):
    persona = _persona()
    artifacts = RunArtifacts(
        conversations=[ConversationArtifact(_transcript("p1", 0), _judge(), False, None)],
        cost_capped=False,
    )
    run_dir = write_run(
        artifacts=artifacts,
        personas=[persona],
        config={},
        run_id="r4",
        git_sha=None,
        output_root=tmp_path,
    )
    convo_path = run_dir / "conversations" / "p1__rep01.json"
    data = json.loads(convo_path.read_text())
    assert data["persona"]["id"] == "p1"
    assert data["judge"]["verdict"] == "pass"
    assert data["totals"]["turns"] == 1
    assert data["endpoint"] == "fast_chat_v2"


def test_judge_failed_serialized(tmp_path: Path):
    persona = _persona()
    artifacts = RunArtifacts(
        conversations=[
            ConversationArtifact(
                _transcript("p1", 0),
                judge_output=None,
                judge_failed=True,
                judge_error="parse error",
            )
        ],
        cost_capped=False,
    )
    run_dir = write_run(
        artifacts=artifacts,
        personas=[persona],
        config={},
        run_id="r5",
        git_sha=None,
        output_root=tmp_path,
    )
    convo = json.loads((run_dir / "conversations" / "p1__rep01.json").read_text())
    assert convo["judge"]["judge_failed"] is True
    assert "parse error" in convo["judge"]["error"]
