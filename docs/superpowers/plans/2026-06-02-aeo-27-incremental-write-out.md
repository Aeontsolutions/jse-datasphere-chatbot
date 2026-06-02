# AEO-27: Incremental Write-Out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write each `ConversationArtifact` to disk as it completes so a killed run loses no completed work, and support `--resume` to restart from where it stopped.

**Architecture:** Add an `on_artifact` async callback + `skip_ids` set to `run_simulation`; the CLI creates the run directory before the run starts, writes a partial manifest, and passes a closure that writes each conversation file as it finishes. Three new helpers in `report.py` (`write_initial_manifest`, `write_conversation`, `load_conversation_artifacts`) keep all disk-layout logic in one place.

**Tech Stack:** Python 3.11, asyncio, Pydantic v2, pytest-asyncio, tmp_path fixtures.

---

## File Map

| File | Change |
|---|---|
| `evals/runner.py` | Add `on_artifact` + `skip_ids` params; call callback after each judge block |
| `evals/report.py` | Add `write_initial_manifest`, `write_conversation`, `load_conversation_artifacts`, `_artifact_from_payload`; add `if path.exists(): continue` guard to `write_run`; add `sys` + transcript imports |
| `evals/cli.py` | Add `--resume` arg; create run_dir early; call `write_initial_manifest`; wire `_on_artifact` closure; merge existing artifacts on resume |
| `evals/tests/test_runner.py` | Add `_fake_judge_output()` helper; add two new `run_simulation` tests |
| `evals/tests/test_report.py` | Add four new tests covering the three new functions and the `write_run` skip guard |
| `evals/tests/test_cli.py` | Add `--resume` argparse test |

---

## Task 1: Add `on_artifact` callback and `skip_ids` to `run_simulation`

**Files:**
- Modify: `evals/runner.py`
- Test: `evals/tests/test_runner.py`

- [ ] **Step 1: Add `_fake_judge_output` helper to test_runner.py**

Open `evals/tests/test_runner.py`. After the existing imports block (after `from evals.runner import RunArtifacts, run_simulation`) add:

```python
from evals.judge import DimensionScore, FactfulnessScore, JudgeOutput, JudgeScores, ToolUseScore


def _fake_judge_output() -> JudgeOutput:
    return JudgeOutput(
        scores=JudgeScores(
            groundedness=DimensionScore(score=4, justification="x"),
            factfulness=FactfulnessScore(score=None, facts_satisfied=[], justification="n/a"),
            goal_completion=DimensionScore(score=4, justification="x"),
            tool_use_appropriateness=ToolUseScore(score=4, justification="x"),
            coherence=DimensionScore(score=4, justification="x"),
            persona_handling=DimensionScore(score=4, justification="x"),
        ),
        verdict="pass",
        verdict_reason="ok",
    )
```

- [ ] **Step 2: Write failing test — on_artifact called per completed artifact**

Append to `evals/tests/test_runner.py`:

```python
@pytest.mark.asyncio
async def test_run_simulation_calls_on_artifact_per_completed():
    persona = _persona(max_turns=1).model_copy(update={"id": "a"})
    from evals.persona_actor import PersonaTurn

    actor = MagicMock()
    actor.act = AsyncMock(return_value=PersonaTurn(utterance="q", done=True))
    client = MagicMock()
    client.send = AsyncMock(return_value=_client_result())
    fake_judge = MagicMock()
    fake_judge.evaluate = AsyncMock(return_value=_fake_judge_output())

    received: list[str] = []

    async def capture(artifact):
        received.append(artifact.transcript.conversation_id)

    await run_simulation(
        personas=[persona],
        replicates=2,
        concurrency=2,
        max_cost_usd_per_run=10.0,
        max_cost_usd_per_conversation=1.0,
        chat_client_factory=lambda _: client,
        persona_actor=actor,
        judge=fake_judge,
        on_artifact=capture,
    )

    assert set(received) == {"a__rep01", "a__rep02"}
```

- [ ] **Step 3: Write failing test — skip_ids skips matching conversations**

Append to `evals/tests/test_runner.py`:

```python
@pytest.mark.asyncio
async def test_run_simulation_skip_ids_skips_matching():
    persona = _persona(max_turns=1).model_copy(update={"id": "a"})
    from evals.persona_actor import PersonaTurn

    actor = MagicMock()
    actor.act = AsyncMock(return_value=PersonaTurn(utterance="q", done=True))
    client = MagicMock()
    client.send = AsyncMock(return_value=_client_result())
    fake_judge = MagicMock()
    fake_judge.evaluate = AsyncMock(return_value=_fake_judge_output())

    artifacts = await run_simulation(
        personas=[persona],
        replicates=3,
        concurrency=3,
        max_cost_usd_per_run=10.0,
        max_cost_usd_per_conversation=1.0,
        chat_client_factory=lambda _: client,
        persona_actor=actor,
        judge=fake_judge,
        skip_ids={"a__rep01", "a__rep02"},
    )

    assert len(artifacts.conversations) == 1
    assert artifacts.conversations[0].transcript.conversation_id == "a__rep03"
```

- [ ] **Step 4: Run tests to confirm they fail**

```
cd evals && pytest tests/test_runner.py::test_run_simulation_calls_on_artifact_per_completed tests/test_runner.py::test_run_simulation_skip_ids_skips_matching -v
```

Expected: both FAIL with `TypeError` (unexpected keyword argument).

- [ ] **Step 5: Implement the changes in runner.py**

In `evals/runner.py`, change the `run_simulation` signature and body. The full updated function (replace from `async def run_simulation` through the end of the file):

```python
async def run_simulation(
    personas: list[PersonaSpec],
    replicates: int,
    concurrency: int,
    max_cost_usd_per_run: float,
    max_cost_usd_per_conversation: float,
    chat_client_factory: Callable[[str], ChatClient],
    persona_actor: PersonaActor,
    judge: Judge,
    on_artifact: Callable[[ConversationArtifact], Awaitable[None]] | None = None,
    skip_ids: set[str] | None = None,
) -> RunArtifacts:
    """Run all personas × replicates concurrently with a global cost cap."""
    semaphore = asyncio.Semaphore(concurrency)
    judge_semaphore = asyncio.Semaphore(concurrency * 2)

    running_cost = 0.0
    cost_lock = asyncio.Lock()
    cost_capped = False
    cancel_event = asyncio.Event()

    async def one(persona: PersonaSpec, rep: int) -> ConversationArtifact | None:
        nonlocal running_cost, cost_capped
        conversation_id = f"{persona.id}__rep{rep + 1:02d}"
        if skip_ids and conversation_id in skip_ids:
            return None
        if cancel_event.is_set():
            return None
        async with semaphore:
            if cancel_event.is_set():
                return None
            chat_client = chat_client_factory(persona.endpoint)
            transcript = await run_conversation(
                persona=persona,
                replicate_index=rep,
                chat_client=chat_client,
                persona_actor=persona_actor,
                max_cost_usd=max_cost_usd_per_conversation,
            )

            convo_cost = float(transcript.totals()["cost_usd"])
            async with cost_lock:
                running_cost += convo_cost
                if running_cost > max_cost_usd_per_run:
                    cost_capped = True
                    cancel_event.set()

        async with judge_semaphore:
            try:
                output = await judge.evaluate(persona=persona, transcript=transcript)
                print(f"  {persona.id} rep{rep+1}: {output.verdict} (turns={len(transcript.turns)}, ${convo_cost:.3f})")
                artifact: ConversationArtifact = ConversationArtifact(transcript, output, False, None)
            except Exception as exc:
                artifact = ConversationArtifact(transcript, None, True, f"{type(exc).__name__}: {exc}")
            if on_artifact is not None:
                await on_artifact(artifact)
            return artifact

    tasks = [
        asyncio.create_task(one(persona, rep))
        for persona in personas
        for rep in range(replicates)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    conversations: list[ConversationArtifact] = []
    for r in results:
        if isinstance(r, ConversationArtifact):
            conversations.append(r)
        elif isinstance(r, BaseException):
            print(f"ERROR: task crashed: {type(r).__name__}: {r}")

    return RunArtifacts(conversations=conversations, cost_capped=cost_capped)
```

- [ ] **Step 6: Run tests to confirm they pass**

```
cd evals && pytest tests/test_runner.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add evals/runner.py evals/tests/test_runner.py
git commit -m "feat(AEO-27): add on_artifact callback and skip_ids to run_simulation"
```

---

## Task 2: Add `write_initial_manifest` and `write_conversation` to report.py

**Files:**
- Modify: `evals/report.py`
- Test: `evals/tests/test_report.py`

- [ ] **Step 1: Write failing tests**

Open `evals/tests/test_report.py`. Add at the end:

```python
import json
from pathlib import Path

import pytest

from evals.persona import PersonaSpec
from evals.report import write_conversation, write_initial_manifest
from evals.runner import ConversationArtifact
from evals.transcript import TerminationReason, Transcript


def _transcript(conv_id: str = "p1__rep01") -> Transcript:
    return Transcript(
        conversation_id=conv_id,
        persona_id="p1",
        replicate_index=0,
        endpoint="fast_chat_v2",
        turns=[],
        termination=TerminationReason(reason="done", at_turn=0),
    )


def _persona() -> PersonaSpec:
    return PersonaSpec(
        id="p1",
        name="P1",
        category="positive",
        endpoint="fast_chat_v2",
        character="A tester.",
        goal="Test things.",
        max_turns=3,
    )


def test_write_initial_manifest_creates_in_progress_file(tmp_path: Path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()

    write_initial_manifest(
        run_dir=run_dir,
        run_id="run1",
        git_sha="abc123",
        config={"replicates": 2},
        personas=[_persona()],
        started_at="2026-06-02T00:00:00+00:00",
    )

    data = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert data["status"] == "in_progress"
    assert data["run_id"] == "run1"
    assert data["git_sha"] == "abc123"
    assert data["personas_run"] == ["p1"]
    assert data["started_at"] == "2026-06-02T00:00:00+00:00"


def test_write_conversation_creates_json_file(tmp_path: Path):
    run_dir = tmp_path / "run1"
    (run_dir / "conversations").mkdir(parents=True)

    artifact = ConversationArtifact(_transcript(), None, False, None)
    write_conversation(run_dir, artifact, _persona())

    path = run_dir / "conversations" / "p1__rep01.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["conversation_id"] == "p1__rep01"
    assert data["endpoint"] == "fast_chat_v2"
```

- [ ] **Step 2: Run tests to confirm they fail**

```
cd evals && pytest tests/test_report.py::test_write_initial_manifest_creates_in_progress_file tests/test_report.py::test_write_conversation_creates_json_file -v
```

Expected: both FAIL with `ImportError` (names not defined yet).

- [ ] **Step 3: Implement `write_initial_manifest` and `write_conversation` in report.py**

Add to the top of `evals/report.py`, after the existing imports:

```python
import sys
```

Add these two functions after the `write_run` function (before `_detect_replicates`):

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```
cd evals && pytest tests/test_report.py::test_write_initial_manifest_creates_in_progress_file tests/test_report.py::test_write_conversation_creates_json_file -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add evals/report.py evals/tests/test_report.py
git commit -m "feat(AEO-27): add write_initial_manifest and write_conversation to report"
```

---

## Task 3: Add `load_conversation_artifacts` and update `write_run` skip guard

**Files:**
- Modify: `evals/report.py`
- Test: `evals/tests/test_report.py`

- [ ] **Step 1: Write failing tests**

Append to `evals/tests/test_report.py`:

```python
from evals.judge import (
    DimensionScore, FactfulnessScore, JudgeOutput, JudgeScores, ToolUseScore,
)
from evals.report import load_conversation_artifacts
from evals.runner import RunArtifacts


def _judge_output() -> JudgeOutput:
    return JudgeOutput(
        scores=JudgeScores(
            groundedness=DimensionScore(score=4, justification="ok"),
            factfulness=FactfulnessScore(score=None, facts_satisfied=[], justification="n/a"),
            goal_completion=DimensionScore(score=4, justification="ok"),
            tool_use_appropriateness=ToolUseScore(score=4, justification="ok"),
            coherence=DimensionScore(score=4, justification="ok"),
            persona_handling=DimensionScore(score=4, justification="ok"),
        ),
        verdict="pass",
        verdict_reason="all good",
    )


def test_load_conversation_artifacts_roundtrip(tmp_path: Path):
    run_dir = tmp_path / "run1"
    (run_dir / "conversations").mkdir(parents=True)

    artifact = ConversationArtifact(_transcript(), _judge_output(), False, None)
    write_conversation(run_dir, artifact, _persona())

    loaded, ids = load_conversation_artifacts(run_dir)

    assert ids == {"p1__rep01"}
    assert len(loaded) == 1
    la = loaded[0]
    assert la.transcript.conversation_id == "p1__rep01"
    assert la.transcript.persona_id == "p1"
    assert la.transcript.replicate_index == 0
    assert la.transcript.endpoint == "fast_chat_v2"
    assert la.judge_output is not None
    assert la.judge_output.verdict == "pass"
    assert la.judge_failed is False


def test_load_conversation_artifacts_empty_dir(tmp_path: Path):
    run_dir = tmp_path / "run1"
    (run_dir / "conversations").mkdir(parents=True)

    loaded, ids = load_conversation_artifacts(run_dir)

    assert loaded == []
    assert ids == set()


def test_load_conversation_artifacts_missing_dir(tmp_path: Path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    # no conversations/ subdirectory

    loaded, ids = load_conversation_artifacts(run_dir)

    assert loaded == []
    assert ids == set()


def test_load_conversation_artifacts_skips_corrupt_file(tmp_path: Path):
    run_dir = tmp_path / "run1"
    (run_dir / "conversations").mkdir(parents=True)
    (run_dir / "conversations" / "bad.json").write_text("not json", encoding="utf-8")

    artifact = ConversationArtifact(_transcript(), None, False, None)
    write_conversation(run_dir, artifact, _persona())

    loaded, ids = load_conversation_artifacts(run_dir)

    assert ids == {"p1__rep01"}
    assert len(loaded) == 1


def test_write_run_skips_existing_conversation_files(tmp_path: Path):
    run_dir = tmp_path / "run1"
    (run_dir / "conversations").mkdir(parents=True)

    # Pre-write a file with sentinel content
    existing = run_dir / "conversations" / "p1__rep01.json"
    existing.write_text('{"pre_existing": true}', encoding="utf-8")

    from evals.report import write_run
    artifact = ConversationArtifact(_transcript(), None, False, None)
    write_run(
        artifacts=RunArtifacts(conversations=[artifact], cost_capped=False),
        personas=[_persona()],
        config={},
        run_id="run1",
        git_sha=None,
        output_root=tmp_path,
    )

    data = json.loads(existing.read_text(encoding="utf-8"))
    assert data.get("pre_existing") is True
```

- [ ] **Step 2: Run tests to confirm they fail**

```
cd evals && pytest tests/test_report.py::test_load_conversation_artifacts_roundtrip tests/test_report.py::test_load_conversation_artifacts_empty_dir tests/test_report.py::test_load_conversation_artifacts_missing_dir tests/test_report.py::test_load_conversation_artifacts_skips_corrupt_file tests/test_report.py::test_write_run_skips_existing_conversation_files -v
```

Expected: all FAIL with `ImportError` or assertion error.

- [ ] **Step 3: Add transcript imports to report.py**

In `evals/report.py`, add to the imports block (after existing imports):

```python
from evals.transcript import ChatTurn, TerminationReason, Transcript
```

- [ ] **Step 4: Add `load_conversation_artifacts` and `_artifact_from_payload` to report.py**

Append after `write_conversation`:

```python
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
```

- [ ] **Step 5: Add `if path.exists(): continue` guard to `write_run`**

In `evals/report.py`, find this block inside `write_run`:

```python
    for c in artifacts.conversations:
        persona = persona_by_id.get(c.transcript.persona_id)
        path = run_dir / "conversations" / f"{c.transcript.conversation_id}.json"
        path.write_text(
            json.dumps(_convo_payload(c, persona), indent=2),
            encoding="utf-8",
        )
```

Replace with:

```python
    for c in artifacts.conversations:
        persona = persona_by_id.get(c.transcript.persona_id)
        path = run_dir / "conversations" / f"{c.transcript.conversation_id}.json"
        if path.exists():
            continue
        path.write_text(
            json.dumps(_convo_payload(c, persona), indent=2),
            encoding="utf-8",
        )
```

- [ ] **Step 6: Run all report tests**

```
cd evals && pytest tests/test_report.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add evals/report.py evals/tests/test_report.py
git commit -m "feat(AEO-27): add load_conversation_artifacts and write_run skip guard"
```

---

## Task 4: Wire everything in cli.py

**Files:**
- Modify: `evals/cli.py`
- Test: `evals/tests/test_cli.py`

- [ ] **Step 1: Write failing test for --resume arg**

Append to `evals/tests/test_cli.py`:

```python
def test_parser_accepts_resume_flag():
    parser = build_arg_parser()
    ns = parser.parse_args(["--resume", "2026-06-01T10-00-00"])
    assert ns.resume_run_id == "2026-06-01T10-00-00"


def test_resume_flag_defaults_to_none():
    parser = build_arg_parser()
    ns = parser.parse_args([])
    assert ns.resume_run_id is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```
cd evals && pytest tests/test_cli.py::test_parser_accepts_resume_flag tests/test_cli.py::test_resume_flag_defaults_to_none -v
```

Expected: both FAIL with `AttributeError`.

- [ ] **Step 3: Add `--resume` to the argument parser in cli.py**

In `evals/cli.py`, inside `build_arg_parser`, add after the `--output-dir` line:

```python
    parser.add_argument(
        "--resume",
        dest="resume_run_id",
        default=None,
        metavar="RUN_ID",
        help="Resume a crashed run; skips conversations already written to output-dir/RUN_ID/",
    )
```

- [ ] **Step 4: Run parser tests to confirm they pass**

```
cd evals && pytest tests/test_cli.py -v
```

Expected: all PASS.

- [ ] **Step 5: Update imports in cli.py**

Replace the existing import line:

```python
from evals.report import write_run
```

with:

```python
from evals.report import load_conversation_artifacts, write_conversation, write_initial_manifest, write_run
from evals.runner import ConversationArtifact, RunArtifacts, run_simulation
```

Note: `run_simulation` is already imported — replace the existing `from evals.runner import run_simulation` line with the line above.

- [ ] **Step 6: Rewrite `_amain` to create run_dir early, write partial manifest, wire callback, handle resume**

Replace the entire `_amain` function in `evals/cli.py` with:

```python
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

    run_id = ns.run_id or (
        ns.resume_run_id if ns.resume_run_id
        else datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    )
    output_root = Path(ns.output_dir) if ns.output_dir else Path(__file__).parent / "runs"
    run_dir = output_root / run_id
    (run_dir / "conversations").mkdir(parents=True, exist_ok=True)

    git_sha = _git_sha()
    started_at = datetime.now(timezone.utc).isoformat()

    existing_artifacts: list[ConversationArtifact] = []
    skip_ids: set[str] | None = None
    if ns.resume_run_id:
        existing_artifacts, skip_ids = load_conversation_artifacts(run_dir)
        print(f"Resuming {run_id}: {len(existing_artifacts)} conversation(s) already complete, skipping.")

    write_initial_manifest(
        run_dir=run_dir,
        run_id=run_id,
        git_sha=git_sha,
        config=config.model_dump(),
        personas=personas,
        started_at=started_at,
    )

    persona_by_id = {p.id: p for p in personas}

    async def _on_artifact(artifact: ConversationArtifact) -> None:
        write_conversation(run_dir, artifact, persona_by_id.get(artifact.transcript.persona_id))

    print(
        f"Running {len(personas)} persona(s) × {config.replicates} replicate(s) "
        f"= {len(personas) * config.replicates} conversation(s) "
        f"(concurrency={config.concurrency})..."
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
        on_artifact=_on_artifact,
        skip_ids=skip_ids,
    )
    ended_at = datetime.now(timezone.utc).isoformat()

    merged = RunArtifacts(
        conversations=existing_artifacts + artifacts.conversations,
        cost_capped=artifacts.cost_capped,
    )
    run_dir = write_run(
        artifacts=merged,
        personas=personas,
        config=config.model_dump(),
        run_id=run_id,
        git_sha=git_sha,
        output_root=output_root,
        started_at=started_at,
        ended_at=ended_at,
    )
    print(f"Wrote run to {run_dir}")
    if merged.cost_capped:
        print("WARNING: cost cap reached; run is partial")
    return 0
```

- [ ] **Step 7: Run the full test suite**

```
cd evals && pytest tests/ -v
```

Expected: all PASS. If any test imports `run_simulation` directly and is affected by the new params, it will still pass because both new params are optional with `None` defaults.

- [ ] **Step 8: Commit**

```bash
git add evals/cli.py evals/tests/test_cli.py
git commit -m "feat(AEO-27): wire incremental writes and --resume into cli"
```

---

## Task 5: Final verification

- [ ] **Step 1: Run the full test suite one more time**

```
cd evals && pytest tests/ -v --tb=short
```

Expected: all PASS, no warnings about missing fixtures or unexpected errors.

- [ ] **Step 2: Confirm the partial manifest is written before conversations complete**

Skim `evals/cli.py` to verify:
1. `(run_dir / "conversations").mkdir(...)` is called before `run_simulation`.
2. `write_initial_manifest(...)` is called before `run_simulation`.
3. `_on_artifact` is passed to `run_simulation`.
4. `merged = RunArtifacts(existing_artifacts + artifacts.conversations, ...)` merges resume artifacts.

- [ ] **Step 3: Commit final state and push**

```bash
git add -p   # review any unstaged changes
git commit -m "feat(AEO-27): incremental write-out complete"
```

---

## Self-Review Checklist

- **Spec coverage:**
  - ✅ Write each artifact as it completes → Task 1 (callback) + Task 2 (`write_conversation`)
  - ✅ Partial manifest with `status: "in_progress"` → Task 2 (`write_initial_manifest`)
  - ✅ Recompute `summary.json` at end → unchanged `write_run` call in Task 4
  - ✅ `--resume` to skip already-written files → Task 3 (`load_conversation_artifacts`) + Task 4 (cli wiring)
  - ✅ `write_run` skips existing conversation files → Task 3 (skip guard)
- **No placeholders:** All steps include full code.
- **Type consistency:** `on_artifact: Callable[[ConversationArtifact], Awaitable[None]] | None` used consistently across Task 1 signature and Task 4 closure.
- **Import check:** `Awaitable` already imported in `runner.py` (line 110); `sys` added to `report.py`; new report functions imported in `cli.py`.
