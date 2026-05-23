"""Quick summarizer for an eval run — surfaces failures, low scores, and notable moments."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

RUN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("evals/runs/baseline_v2_2026_05_23")

summary = json.loads((RUN_DIR / "summary.json").read_text(encoding="utf-8"))

print(f"=== RUN: {summary['run_id']} ===")
print(f"started: {summary['started_at']}")
print(f"ended:   {summary['ended_at']}")
print(f"convs:   {summary['conversation_count']}")
print(f"overall verdicts: {summary['overall']['verdict_counts']}")
print(f"overall incomplete (transport errors): {summary['overall'].get('incomplete_count', 0)}")
print(
    f"overall mean scores: "
    f"groundedness={summary['overall'].get('mean_groundedness')}, "
    f"factfulness={summary['overall'].get('mean_factfulness')}, "
    f"goal={summary['overall'].get('mean_goal_completion')}, "
    f"persona={summary['overall'].get('mean_persona_handling')}, "
    f"coherence={summary['overall'].get('mean_coherence')}"
)
print()

print("=== BY CATEGORY ===")
for cat, s in summary["by_category"].items():
    print(
        f"  {cat:8s}: verdicts={s['verdict_counts']} mean_persona_handling={s.get('mean_persona_handling')}"
    )
print()

print("=== BY ENDPOINT ===")
for ep, s in summary["by_endpoint"].items():
    print(
        f"  {ep:14s}: count={s['count']} verdicts={s['verdict_counts']} mean_latency_ms={s.get('mean_latency_ms', 0):.0f}"
    )
print()

print("=== PER-PERSONA RESULTS ===")
for pid, s in sorted(summary["by_persona"].items()):
    verdict = s.get("verdict_counts", {})
    print(
        f"  {pid:40s} {verdict.get('pass', 0)}p/{verdict.get('partial', 0)}P/{verdict.get('fail', 0)}f "
        f"turns={s.get('mean_turns', 0):.1f} "
        f"ground={s.get('mean_groundedness', 0):.1f} "
        f"goal={s.get('mean_goal_completion', 0):.1f} "
        f"persona={s.get('mean_persona_handling', 0):.1f}"
    )
print()

print("=== TERMINATION REASONS ===")
termination_counts: dict[str, int] = {}
errors_by_persona: list[tuple[str, str]] = []
for f in sorted((RUN_DIR / "conversations").glob("*.json")):
    convo = json.loads(f.read_text(encoding="utf-8"))
    reason = convo["termination"]["reason"]
    err = convo["termination"].get("error_type") or ""
    key = f"{reason}/{err}" if err else reason
    termination_counts[key] = termination_counts.get(key, 0) + 1
    if reason == "error":
        errors_by_persona.append((convo["persona"]["id"], err))
for k, v in sorted(termination_counts.items(), key=lambda x: -x[1]):
    print(f"  {k:30s} {v}")
print()
if errors_by_persona:
    print("=== ERRORS ===")
    for pid, etype in errors_by_persona:
        print(f"  {pid:40s} {etype}")
    print()

print("=== NOTABLE MOMENTS (judge-flagged) ===")
for f in sorted((RUN_DIR / "conversations").glob("*.json")):
    convo = json.loads(f.read_text(encoding="utf-8"))
    if isinstance(convo.get("judge"), dict) and "notable_moments" in convo["judge"]:
        for nm in convo["judge"]["notable_moments"]:
            print(f"  {convo['persona']['id'][:30]:30s} turn={nm['turn']} type={nm['type']} — {nm['note'][:100]}")
print()

print("=== LATENCY PERCENTILES (completed conversations only) ===")
latencies: list[float] = []
for f in sorted((RUN_DIR / "conversations").glob("*.json")):
    convo = json.loads(f.read_text(encoding="utf-8"))
    for t in convo["turns"]:
        if t.get("latency_ms"):
            latencies.append(float(t["latency_ms"]))
if latencies:
    sorted_l = sorted(latencies)
    print(f"  N={len(sorted_l)} min={sorted_l[0]:.0f}ms")
    print(f"  p50={sorted_l[len(sorted_l) // 2]:.0f}ms")
    print(f"  p95={sorted_l[int(len(sorted_l) * 0.95)]:.0f}ms")
    print(f"  max={sorted_l[-1]:.0f}ms")
else:
    print("  (no completed turns)")
