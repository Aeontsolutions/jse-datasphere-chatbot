# JSE Chatbot Eval Suite

Persona-driven, multi-turn simulation suite for the JSE DataSphere chatbot.
See the [design spec](../docs/superpowers/specs/2026-05-23-simulation-eval-design.md)
for rationale.

## Install

```bash
cd evals
pip install -e ".[dev]"
```

Set `GOOGLE_API_KEY` (or whatever `gemini_api_key_env` is set to in
`config/default.yaml`) to a Gemini API key.

## Run a simulation

The FastAPI chatbot must be reachable at `--base-url`. Example:

```bash
# 1. Start the chatbot in another terminal
cd ../fastapi_app && uvicorn main:app --port 8000

# 2. Run the suite
cd evals
python -m evals.cli --replicates 3
# → writes evals/runs/<timestamp>/
```

Common variations:

```bash
# Run a single persona to iterate quickly
python -m evals.cli --persona senior_analyst_ncb_financials --replicates 1

# Restrict to negative personas
python -m evals.cli --category negative

# Restrict to one endpoint
python -m evals.cli --endpoint fast_chat_v2

# Override defaults
python -m evals.cli --concurrency 6 --max-cost-usd 2.0 --run-id smoke

# Use a custom config file
python -m evals.cli --config path/to/my-config.yaml
```

## View results

Two ways to open the viewer:

**(a) Drag-and-drop** — open `evals/viewer/index.html` in your browser
(Chrome / Edge work best). Drag the `evals/runs/<run_id>/` folder onto the
dropzone. To compare runs, drop two or more folders.

**(b) Local server** — run a tiny static server:

```bash
python -m evals.serve --port 8765
```

Then open:
- `http://localhost:8765/viewer/?run=<run_id>` to auto-load a single run
- `http://localhost:8765/viewer/?runs=<id1>,<id2>` to load multiple runs

## Authoring personas

Each persona is a YAML file in `evals/personas/`. See
[the design spec](../docs/superpowers/specs/2026-05-23-simulation-eval-design.md#4-persona-schema)
for the full schema. The repo ships four examples to start from:

- `senior_analyst_ncb_financials.yaml` — analyst, `/fast_chat_v2`
- `student_what_is_stock_market.yaml` — novice, `/chat/stream`
- `investor_compare_ncb_vs_jmmb.yaml` — retail investor, head-to-head
- `negative_chitchat_offtopic.yaml` — negative; bot should decline

## Configuration

`evals/config/default.yaml` controls models, concurrency, and cost caps.
`evals/config/judge_rubric.yaml` controls the prompt the judge sees;
edit dimension descriptions to tune scoring without touching code.

`judge_rubric.yaml`'s `verdict_weights` block is not just documentation —
`report.py` uses it to compute an independent, deterministic
`computed_verdict` per conversation from the six dimension scores, and flags
`verdict_agreement` (whether it matches the judge LLM's own holistic
`pass`/`fail`/`partial` call). Check `summary.json`'s `verdict_agreement_rate`
(under `overall`, and per category) after a run; a low rate means either the
judge is drifting from the documented rubric or the rubric no longer reflects
what actually matters — worth investigating either way, not just noise.

**The deploy gate counts failures from `computed_verdict`, not the judge's
holistic verdict.** The two disagree on roughly one conversation in six —
measured at 84% and 87% agreement over two 93-conversation runs — and the
judge is consistently the harsher of the pair. Gating on it put the judge's
scoring drift in the release path. Across twelve CI-shaped slices (one
replicate over all personas) drawn from four runs:

| verdict source | fail counts | spread | stdev |
|---|---|---|---|
| judge holistic | 8,9,4,4,5,5,6,6,4,6,7,6 | 5 | 1.52 |
| `computed_verdict` | 5,3,3,3,3,3,2,3,3,3,3,5 | 3 | 0.83 |

`--max-fail-increase` defaults to 2, so the judge metric swung wider than the
gate's entire tolerance. Both counts appear in `summary.json`
(`verdict_counts` and `computed_verdict_counts`) and both are printed by
`check_eval_gate.py`; only the computed one gates.

Baselines record `verdict_source: "computed"`. A baseline seeded before this
change counted judge verdicts, so the gate **refuses** to compare against it
rather than silently mixing two metrics — re-seed with `--update-baseline`
over three or more reviewed runs.

## Cost accounting

`max_cost_usd_per_run` and `max_cost_usd_per_conversation` (in
`config/default.yaml`) bound **all** Gemini spend the eval suite generates:
the chatbot-under-test's own reported cost, the persona actor's calls
(one per turn), and the judge's calls (one per conversation) — not just the
chatbot's. `summary.json`'s `total_cost_usd` is the true total for the same
reason; per-conversation JSON breaks it out under `totals` as
`chat_and_persona_cost_usd` + `judge_cost_usd`.

## Judge calibration

Unit tests (`test_judge.py`) only check that the judge produces well-formed
output against mocked responses — nothing validates that its *scoring* is
accurate, since there's no ground truth to compare against in a
no-network test run. `tests/test_judge_calibration.py` is that ground
truth: it runs the real judge against a hand-labeled obviously-grounded
transcript and a hand-labeled obviously-hallucinated one, and asserts it
lands in the expected bucket. It's skipped unless `GOOGLE_API_KEY` is set
(costs a couple of real Gemini calls); run it explicitly after bumping
`judge_model` or editing `judge_rubric.yaml`:

```bash
GOOGLE_API_KEY=... pytest tests/test_judge_calibration.py -v
```

## Endpoint coverage caveat

24 of the 31 shipped personas hit `chat_stream`; only 7 hit `fast_chat_v2`.
Running with `--endpoint chat_stream` for quick iteration therefore
under-exercises the financial DB tool relative to a full, unfiltered run —
the CLI prints a `NOTE:` line when a filtered persona selection is light on
financial-tool coverage (see `financial_tool_coverage()` in `cli.py`). To
directly verify financial tool calling on `/chat/stream`, send a metric
query (e.g. "What was NCB revenue in 2023?") with `enable_financial_data:
true` rather than relying on a `chat_stream`-filtered eval run.

## Provenance caveat

Nothing in this suite verifies which chatbot code is actually running at
`--base-url` — that's on you. Before trusting a run's results, confirm the
server process was started from the commit you think it was
(`git -C <chatbot-checkout-dir> rev-parse HEAD`), and that there are no
uncommitted local edits. `manifest.json`'s `git_sha` records the eval
suite's own commit, not the chatbot server's.

## Tests

```bash
pytest                                  # all unit tests (no network, run in CI)
```

The CLI's `--persona` mode is the de-facto live integration test — it
hits the real Gemini API and the real chatbot. Keep `--replicates 1`
for quick iteration. `tests/test_judge_calibration.py` is a second,
narrower live check — see "Judge calibration" above.

## Layout

```
evals/
├── persona.py / persona_actor.py / judge.py / runner.py / report.py / cli.py
├── client/{base,financial,agent_stream}.py
├── config/{default,judge_rubric}.yaml
├── personas/*.yaml
├── tests/
├── viewer/{index.html, viewer.js, styles.css}
└── runs/<run_id>/{manifest,summary}.json + conversations/*.json   # gitignored
```
