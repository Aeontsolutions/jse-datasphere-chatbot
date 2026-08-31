"""Render an eval run's summary.json as GitHub step-summary markdown.

The release pipeline puts this in front of the human approver -- it is the
report they read before deciding whether a tag ships to prod, and it is what
makes that approval a real review rather than a click on a green check.

Scores come from `by_category.positive`, matching scripts/check_eval_gate.py:
`overall` mixes in refusal personas that score goal_completion=1.0 by design.
"""

from __future__ import annotations

from typing import Any

# Exactly the dimensions scripts/check_eval_gate.py gates on. _persona_stats
# also emits tool_use_appropriateness, which the gate ignores.
DIMENSIONS = (
    "groundedness",
    "factfulness",
    "goal_completion",
    "persona_handling",
    "coherence",
)

GET_ENDPOINTS = (
    ("/docs", "Swagger UI -- interactive, drive the API by hand from here"),
    ("/openapi.json", "OpenAPI schema, for pointing your own client at it"),
    ("/version", "commit this build was made from"),
    ("/health", "component status (S3, BigQuery, Gemini)"),
    ("/financial/metadata", "available companies and statement coverage"),
)

POST_ENDPOINTS = (
    ("/chat/stream", "main agent endpoint -- what the eval personas exercise"),
    ("/fast_chat_v2", "financial-data fast path"),
    ("/chat", "basic non-streaming chat"),
)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _delta(current: float | None, base: float | None) -> str:
    if current is None or base is None:
        return "n/a"
    diff = current - base
    marker = "🔻" if diff < 0 else "▲" if diff > 0 else "—"
    return f"{diff:+.2f} {marker}"


def render_summary_markdown(
    summary: dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    tag: str,
    commit: str,
    base_url: str,
    category: str = "positive",
) -> str:
    """Build the release report. Never raises on a partial summary -- a run
    that failed halfway must still render something the approver can read."""
    stats = summary.get("by_category", {}).get(category, {})
    base_means = (baseline or {}).get("means", {})

    lines: list[str] = [
        f"# Release {tag}",
        "",
        f"Commit `{commit}`, confirmed as the build dev is serving.",
        "",
        f"Eval run `{summary.get('run_id', 'unknown')}` — category `{category}`, "
        f"{stats.get('judged_count', 0)} of {stats.get('count', 0)} conversations judged.",
        "",
    ]

    if baseline is None:
        lines += ["> **No baseline found.** Scores are shown without comparison.", ""]

    lines += [
        "| Dimension | Baseline | This run | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for dim in DIMENSIONS:
        current = stats.get(f"mean_{dim}")
        base = base_means.get(dim)
        lines.append(f"| {dim} | {_fmt(base)} | {_fmt(current)} | {_delta(current, base)} |")

    verdicts = stats.get("verdict_counts", {})
    fail_now = verdicts.get("fail", 0)
    fail_base = (baseline or {}).get("fail_verdicts")
    fail_base_cell = "n/a" if fail_base is None else str(fail_base)
    fail_delta = "n/a" if fail_base is None else f"{fail_now - fail_base:+d}"
    lines += [
        f"| fail verdicts | {fail_base_cell} | {fail_now} | {fail_delta} |",
        "",
        f"Verdicts: **{verdicts.get('pass', 0)} pass**, "
        f"{verdicts.get('partial', 0)} partial, {verdicts.get('fail', 0)} fail.",
        "",
        "## Verify it yourself",
        "",
        f"This release was evaluated on dev at `{base_url}`. Exercise it with your own "
        "client before approving — the scores above are a floor, not the whole picture.",
        "",
        # The approval gate can sit for days, and every merge to main redeploys
        # dev in the meantime. Without this check the approver hand-tests a
        # later build and approves on it, so /version is made load-bearing
        # rather than just another link in the list below.
        f"**Check [`GET {base_url}/version`]({base_url}/version) first — it must return "
        f"`{commit}`.** If it returns anything else, dev has moved on since this report "
        "was written and the URLs below no longer exercise this release; what you try by "
        "hand would be a different build.",
        "",
    ]
    for path, note in GET_ENDPOINTS:
        lines.append(f"- [`GET {path}`]({base_url}{path}) — {note}")
    for path, note in POST_ENDPOINTS:
        lines.append(f"- `POST {base_url}{path}` — {note}")

    lines += [
        "",
        "Full transcripts are in the run artifact attached to this workflow run. "
        "Download it and browse with `python evals/serve.py`.",
        "",
    ]
    return "\n".join(lines)
