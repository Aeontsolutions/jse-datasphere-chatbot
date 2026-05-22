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

## Tests

```bash
pytest                                  # all unit tests (no network)
```

The CLI's `--persona` mode is the de-facto live integration test — it
hits the real Gemini API and the real chatbot. Keep `--replicates 1`
for quick iteration.

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
