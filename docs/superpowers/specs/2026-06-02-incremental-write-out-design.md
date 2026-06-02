# AEO-27: Incremental Write-Out in the Eval Runner

**Date:** 2026-06-02  
**Ticket:** [AEO-27](https://linear.app/aeontech/issue/AEO-27)  
**Status:** Approved

## Problem

`evals/runner.py` (`run_simulation`) only writes `manifest.json`, `summary.json`, and `conversations/*.json` after the entire run completes. A killed process loses all in-flight progress, including already-paid-for judge calls. This happened twice during baseline analysis, losing 25+ completed turns.

## Goals

1. Write each `ConversationArtifact` to disk as it completes.
2. Write a partial `manifest.json` (with `status: "in_progress"`) before the run starts.
3. Recompute and write `summary.json` at the end (unchanged).
4. Support `--resume <run_id>` to skip already-written conversations on restart.

## Architecture

Three files change; no new files.

| File | Change |
|---|---|
| `evals/runner.py` | Add `on_artifact` callback + `skip_ids` params to `run_simulation` |
| `evals/report.py` | Add `write_initial_manifest`, `write_conversation`, `load_conversation_artifacts`; `write_run` skips existing conversation files |
| `evals/cli.py` | Create run_dir early, wire callback, add `--resume` arg |

## Data Flow

### Normal Run

1. CLI computes `run_id`, creates `run_dir / conversations /` before calling `run_simulation`.
2. CLI calls `write_initial_manifest` → writes `manifest.json` with `status: "in_progress"`.
3. CLI builds `on_artifact` closure (captures `persona_by_id`) and passes it with `skip_ids=None`.
4. Runner calls `await on_artifact(artifact)` immediately after each judge block completes.
5. Closure calls `write_conversation(run_dir, artifact, persona)` → writes `conversations/<id>.json`.
6. After `run_simulation` returns, CLI calls `write_run`:
   - Skips conversation files that already exist on disk.
   - Overwrites `manifest.json` (final version, no `status` field).
   - Writes `summary.json`.

### Resume Run (`--resume <run_id>`)

1. CLI resolves `run_dir = output_root / run_id`.
2. Calls `load_conversation_artifacts(run_dir)` → returns `(existing_artifacts, skip_ids)`.
3. Passes `skip_ids` to `run_simulation`; runner skips any conversation_id in the set.
4. After run, merges `existing_artifacts` with new ones before passing to `write_run`.

## Interface Signatures

```python
# evals/runner.py
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
) -> RunArtifacts: ...
```

```python
# evals/report.py
def write_initial_manifest(
    run_dir: Path,
    run_id: str,
    git_sha: str | None,
    config: dict,
    personas: list[PersonaSpec],
    started_at: str,
) -> None: ...

def write_conversation(
    run_dir: Path,
    artifact: ConversationArtifact,
    persona: PersonaSpec | None,
) -> None: ...

def load_conversation_artifacts(
    run_dir: Path,
) -> tuple[list[ConversationArtifact], set[str]]: ...

# write_run: unchanged signature; skips conversation files that already exist on disk
```

```
# evals/cli.py — new argument
--resume <run_id>   Resume a crashed run from output_dir/run_id; skips already-written conversations
```

## Resume Reconstruction

`load_conversation_artifacts` reads each `conversations/*.json` and reconstructs:
- `Transcript` via `Transcript.model_validate({...})` — fields `conversation_id`, `endpoint`, `turns`, `termination` are stored directly; `persona_id` comes from `payload["persona"]["id"]`; `replicate_index` is parsed from the `__repNN` suffix (1-indexed → 0-indexed).
- `JudgeOutput` via `JudgeOutput.model_validate(payload["judge"])` when present.
- Corrupt/unreadable files are silently skipped (logged to stderr).

## `skip_ids` Check in Runner

Inside `one(persona, rep)`:
```python
conversation_id = f"{persona.id}__rep{rep + 1:02d}"
if skip_ids and conversation_id in skip_ids:
    return None
```
This mirrors the formula in `run_conversation` exactly.

## Behaviour Changes

- `write_run` gains one guard: `if path.exists(): continue` before writing each conversation file. This is backward-compatible; in a fresh run no files exist yet.
- The initial `manifest.json` contains `status: "in_progress"`; the final one does not include a `status` field (matching the current schema).
- `on_artifact` is called with `asyncio.to_thread` if the write is slow — but given small JSON files, a sync write inside the callback is acceptable and simpler.

## Testing

| Test | Location |
|---|---|
| `test_on_artifact_called_per_completed_artifact` | `test_runner.py` |
| `test_skip_ids_skips_matching_conversations` | `test_runner.py` |
| `test_write_initial_manifest_creates_in_progress` | `test_report.py` |
| `test_write_conversation_creates_file` | `test_report.py` |
| `test_load_conversation_artifacts_roundtrip` | `test_report.py` |
| `test_write_run_skips_existing_conversation_files` | `test_report.py` |
| CLI resume integration (`--resume` loads + merges) | `test_cli.py` |

## Out of Scope

- Viewer integration (pairing with R5 is noted in the ticket but not part of this change).
- Atomic writes (write to `.tmp` then rename) — not needed at this scale.
