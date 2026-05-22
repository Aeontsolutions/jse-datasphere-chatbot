# Simulation-Based Eval Suite — Design

**Date:** 2026-05-23
**Status:** Draft (pending implementation plan)
**Author:** Brainstormed with Claude

## 1. Purpose

Add a persona-driven, multi-turn simulation harness for evaluating changes to
the JSE DataSphere chatbot. The existing suite at `fastapi_app/tests/uat/` is
single-turn and unit-test-style: it sends one query, asserts deterministic
fields, and exits. It cannot catch regressions in multi-turn behavior,
groundedness, hallucination, persona handling, or goal completion.

This suite is **complementary**, not a replacement. UAT continues to gate CI
on deterministic checks. The simulation suite is a **viewer-driven snapshot
tool** for quickly evaluating qualitative changes during iteration.

## 2. Goals & non-goals

### Goals

- Drive realistic multi-turn conversations with the chatbot using LLM-played
  personas defined in YAML.
- Capture deterministic metrics (latency, cost, turns, tools fired) plus
  LLM-judged qualitative metrics (groundedness, factfulness, goal completion,
  coherence, persona handling, tool-use appropriateness).
- Produce structured JSON output that a static HTML+JS viewer renders without
  a build step or backend.
- Stay under ~$1/run in LLM eval overhead so iteration is cheap.
- Cover both `/fast_chat_v2` (non-streaming, statistical) and `/chat/stream`
  (streaming, open-ended) — persona declares which.

### Non-goals (v1)

- Pass/fail CI gating. Suite is snapshot-only; humans decide what regressed.
- Baseline comparison / diffing two runs. View one run at a time.
- Editing personas or launching runs from the viewer.
- Auth, sharing, hosting. Viewer is local-only.
- Per-turn judging. Judge runs once per conversation at the end.

## 3. Architecture overview

```
evals/                                  (new top-level folder, independent of fastapi_app)
├── README.md
├── pyproject.toml                      (gemini SDK, httpx, pyyaml, pydantic)
├── personas/                           (one YAML per persona)
├── config/
│   ├── default.yaml                    (models, concurrency, cost caps, base_url)
│   └── judge_rubric.yaml               (rubric weights + per-score definitions)
├── src/
│   ├── runner.py                       (orchestrates personas × replicates)
│   ├── persona.py                      (PersonaSpec loader)
│   ├── client/
│   │   ├── base.py                     (ChatClient protocol)
│   │   ├── agent_stream.py             (/chat/stream SSE consumer)
│   │   └── financial.py                (/fast_chat_v2 client)
│   ├── persona_actor.py                (Gemini Flash 2.5 role-plays user)
│   ├── judge.py                        (Gemini Pro 2.5 scores transcript)
│   ├── transcript.py                   (turn capture + serialization)
│   ├── metrics.py                      (latency/cost aggregation, shared utils)
│   ├── report.py                       (writes manifest/summary/conversations)
│   └── cli.py                          (`python -m evals.run`)
├── runs/                               (one folder per run, gitignored)
│   └── <run_id>/
│       ├── manifest.json               (config snapshot, git SHA, timing)
│       ├── summary.json                (aggregate metrics)
│       └── conversations/              (one JSON per conversation)
└── viewer/                             (static HTML+JS, no build)
    ├── index.html
    ├── viewer.js
    └── styles.css
```

The `evals/` tree is fully independent of `fastapi_app/`. It hits the chatbot
over HTTP, so the API can be running locally, in Docker, or deployed.

## 4. Persona schema

YAML, one file per persona under `evals/personas/`. Fields:

| Field | Required | Description |
|---|---|---|
| `id` | yes | Unique slug; used in output filenames. |
| `name` | yes | Human-readable title shown in viewer. |
| `category` | yes | `positive` or `negative`. Metadata; affects rubric weighting only. |
| `endpoint` | yes | `fast_chat_v2` or `chat_stream`. Picks the client. |
| `character` | yes | Free-text system-prompt material describing personality, tone, style. |
| `goal` | yes | What the persona is trying to achieve; the judge uses this for `goal_completion`. |
| `max_turns` | yes | Hard cap. Conversation aborts at this turn regardless of `done`. |
| `expected_facts` | no | List of natural-language assertions the judge checks (factfulness layer). |
| `api_options` | no | Pass-through request flags: `memory_enabled`, `enable_web_search`, `enable_financial_data`. |
| `opening_style` | no | `cold_open` (default), `warmup`, or `direct_question`. Shapes turn 1. |
| `notes` | no | Free-text author notes; not sent to any LLM. |

A **negative persona** uses the same schema; the `character` and `goal`
describe off-topic / adversarial behavior, and `expected_facts` describe
what the bot *should* do (decline, redirect).

Example positive and negative persona YAMLs are in section 9.

## 5. Conversation loop

```
┌─────────────────────────────────────────┐
│ Persona Actor (Gemini Flash 2.5)        │
│  sees: character, goal, transcript text │
│  outputs: { utterance, done, reason }   │
└──────────────┬──────────────────────────┘
               │ utterance
               ▼
┌─────────────────────────────────────────┐
│ Chat Client                             │
│  /fast_chat_v2   POST → JSON            │
│  /chat/stream    POST → SSE → assemble  │
│  captures: latency, cost, tools,        │
│            sources, full response       │
└──────────────┬──────────────────────────┘
               │ chatbot text reply (persona sees text only)
               ▼
┌─────────────────────────────────────────┐
│ Transcript (in-memory + JSON on disk)   │
└─────────────────────────────────────────┘
               │
  Loop until: persona emits done=true OR turn count == max_turns
               │
               ▼
            Judge
```

**Persona actor:**
- Model: Gemini 2.5 Flash, temperature 0.8.
- Sees: persona character + goal as system prompt, transcript as
  alternating `You: …` / `Bot: …` lines with **only the bot's text response**
  (no sources, no tool metadata). This keeps the simulation realistic.
- Output is structured JSON enforced via Gemini `response_schema`:
  `{ "utterance": str, "done": bool, "done_reason": str | null }`.
- One JSON-parse retry on malformed output; second failure aborts the
  conversation as `persona_malformed`.

**Chat client:**
- Two implementations behind a common `ChatClient` protocol:
  - `FinancialClient` for `/fast_chat_v2`: single POST, parses
    `FinancialDataResponse`.
  - `AgentStreamClient` for `/chat/stream`: consumes SSE chunks, assembles
    the final `AgentChatResponse`-shaped dict. Records TTFB and total
    stream duration separately.
- Both return a uniform `ChatTurn` containing `chatbot_text`,
  `chatbot_metadata` (the full response dict), `latency_ms`, `ttfb_ms`
  (nullable for non-streaming), `cost_usd`, `input_tokens`, `output_tokens`
  — pulled from `cost_summary` when present.
- The persona LLM never sees `chatbot_metadata`. The judge does.

**Termination:**
- `done=true` from persona → `termination.reason = "done"`.
- Turn count reaches `max_turns` → `termination.reason = "cap"`.
- API 5xx, timeout, malformed JSON → `termination.reason = "error"`, judge
  still runs on the partial transcript with `incomplete=true`.

**Replicate variance:**
- Replicate `N` of persona `P` seeds the persona LLM with
  `seed = stable_hash(P.id) + N`, temperature 0.8.
- Chatbot API is not seeded — natural non-determinism contributes.
- 3 replicates per persona by default (configurable).

## 6. Judge

**Model:** Gemini 2.5 Pro, temperature 0.2, `response_schema` enforced.
One call per conversation, run after the loop completes.

**Inputs to the judge prompt:**
- Persona `character`, `goal`, `category`, `expected_facts`, `endpoint`.
- Full transcript including every chatbot turn's `sources`,
  `tools_executed`, `filters_used`, `chart` presence, `data_found`,
  `record_count`.
- Run metadata: total turns, termination reason, total latency, total cost.
- The rubric YAML's score definitions injected verbatim into the prompt so
  scoring criteria are config-driven.

**Output schema:**

```yaml
scores:
  groundedness:
    score: int (1-5)
    justification: str
  factfulness:
    score: int (1-5) | null            # null if persona has no expected_facts
    facts_satisfied: [bool]            # parallel to persona.expected_facts
    justification: str
  goal_completion:
    score: int (1-5)
    justification: str
  tool_use_appropriateness:
    score: int (1-5)
    observed_tools: [str]
    justification: str
  coherence:
    score: int (1-5)
    justification: str
  persona_handling:
    score: int (1-5)
    justification: str

verdict: "pass" | "fail" | "partial"
verdict_reason: str

notable_moments:
  - turn: int
    type: "hallucination" | "good_citation" | "wrong_tool" | "refusal" | "other"
    note: str
```

**Verdict weighting** lives in `config/judge_rubric.yaml`:
- Default (positive personas): groundedness 0.30, goal_completion 0.30,
  factfulness 0.20, tool_use 0.10, coherence 0.05, persona_handling 0.05.
- Negative personas: persona_handling 0.50, goal_completion (inverted)
  0.30, coherence 0.10, groundedness 0.10. "Inverted" means the
  contribution is computed as `(6 - score)` rather than `score`: a
  negative persona's goal is to derail the bot, so a *low*
  goal_completion (bot didn't engage off-topic) is a *good* outcome.

**Failure handling:**
- Schema-violation on judge output → one retry at temperature 0; second
  failure marks the conversation `judge_failed=true`. Viewer renders this
  state explicitly so it isn't mistaken for a passing run.

**Cost ceiling** (estimated for a 30-conversation run, 3 turns avg):
- Persona actor: ~$0.02–$0.05
- Judge: ~$0.30–$0.50
- **Total LLM eval overhead: under $1/run.**
- Chatbot API cost is tracked separately via `cost_summary`.

## 7. Runner & CLI

**Entrypoint:** `python -m evals.run` (and direct `python evals/src/cli.py`).

**Flags** (full set; all override `config/default.yaml`):

| Flag | Default | Description |
|---|---|---|
| `--base-url` | from config | Chatbot API base URL. |
| `--persona <id>` | (all) | Repeatable; restrict to listed personas. |
| `--category <c>` | (all) | `positive` or `negative`. |
| `--endpoint <e>` | (both) | `fast_chat_v2` or `chat_stream`. |
| `--replicates N` | 3 | Replicates per persona. |
| `--concurrency N` | 4 | Conversations in flight at once. |
| `--max-cost-usd N` | 5.00 | Hard run cap. Aborts pending convos when reached. |
| `--config PATH` | `config/default.yaml` | Config file override. |
| `--run-id NAME` | ISO timestamp | Folder name under `runs/`. |
| `--output-dir PATH` | `runs/<run_id>/` | Override output root. |
| `--request-timeout-s N` | 90 | Per-API-call timeout. |

**`config/default.yaml`:**

```yaml
base_url: http://localhost:8000
replicates: 3
concurrency: 4
request_timeout_s: 90
persona_model: gemini-2.5-flash
judge_model: gemini-2.5-pro
persona_temperature: 0.8
judge_temperature: 0.2
max_cost_usd_per_run: 5.00
max_cost_usd_per_conversation: 0.50
gemini_api_key_env: GOOGLE_API_KEY
```

**Concurrency model:**
- `asyncio.Semaphore(concurrency)` gates parallel conversations.
- Each conversation is sequential internally (turns can't overlap).
- Judge calls overlap with conversations using a second semaphore sized
  `concurrency * 2` (judges are I/O-bound and cheap).
- When `max_cost_usd_per_run` is breached, the runner cancels pending
  conversations, writes a partial report with a `cost_capped: true` flag
  in `manifest.json`, and exits 0 (snapshot-only — no failure exit).

**Shared utilities with existing UAT:**
- `_extract_cost_summary` and `calculate_latency_stats` are duplicated as
  small inline helpers in `evals/src/metrics.py`. **The existing UAT
  framework is not modified.** Decoupling outweighs the small duplication.

## 8. Output format

One folder per run under `runs/<run_id>/`:

```
runs/<run_id>/
├── manifest.json           # config snapshot + git SHA + timing + cost_capped
├── summary.json            # aggregate metrics for viewer overview
└── conversations/
    ├── <persona_id>__rep01.json
    ├── <persona_id>__rep02.json
    └── ...
```

**Per-conversation JSON** (self-contained, viewer renders without other files):

```jsonc
{
  "conversation_id": "senior_analyst_ncb_financials__rep01",
  "persona": { /* full persona YAML inlined */ },
  "endpoint": "fast_chat_v2",
  "turns": [
    {
      "turn_index": 0,
      "persona_utterance": "Show me NCB's profitability over the last three fiscal years.",
      "chatbot_response": { /* full API response */ },
      "latency_ms": 1820,
      "ttfb_ms": null,
      "cost_usd": 0.0034,
      "tokens": { "input": 1200, "output": 380 }
    }
  ],
  "termination": {
    "reason": "done",                     // "done" | "cap" | "error"
    "at_turn": 4,
    "persona_done_reason": "..." | null,
    "error": null | { "type": "...", "message": "..." }
  },
  "totals": { "turns": 4, "latency_ms": 7300, "cost_usd": 0.018 },
  "judge": { /* full judge output, or { "judge_failed": true, "error": "..." } */ },
  "errors": []
}
```

**`summary.json`:**

```jsonc
{
  "run_id": "2026-05-23T14-00-00",
  "started_at": "...", "ended_at": "...", "git_sha": "...",
  "config": { /* effective config after flag overrides */ },
  "conversation_count": 30,
  "by_persona": { /* mean + std for each dimension across replicates */ },
  "by_endpoint": { "fast_chat_v2": {...}, "chat_stream": {...} },
  "by_category": { "positive": {...}, "negative": {...} },
  "overall": {
    "mean_groundedness": 4.2, "std_groundedness": 0.6,
    "mean_factfulness": 3.8, "mean_goal_completion": 4.1,
    "mean_tool_use_appropriateness": 4.3, "mean_coherence": 4.5,
    "mean_persona_handling": 4.0,
    "verdict_counts": { "pass": 24, "partial": 4, "fail": 2 },
    "judge_failed_count": 0,
    "incomplete_count": 0,
    "mean_turns": 3.4,
    "mean_latency_ms": 7800,
    "total_cost_usd": 0.41
  }
}
```

**`manifest.json`:**

```jsonc
{
  "run_id": "2026-05-23T14-00-00",
  "started_at": "...", "ended_at": "...",
  "git_sha": "d71641f...",
  "config": { /* effective config */ },
  "personas_run": ["senior_analyst_ncb_financials", "..."],
  "replicates": 3,
  "cost_capped": false
}
```

## 9. Persona examples

**Positive — `senior_analyst_ncb_financials.yaml`:**

```yaml
id: senior_analyst_ncb_financials
name: "Senior Equity Analyst — NCB deep-dive"
category: positive
endpoint: fast_chat_v2
character: |
  You are a senior equity analyst at a regional brokerage. You speak crisply,
  use financial jargon naturally (EPS, ROE, NIM, YoY), and push back when an
  answer feels surface-level. You don't apologize, you don't pad.
goal: |
  Understand NCB Financial Group's profitability trend over the last 3 fiscal
  years and decide whether revenue growth is being driven by net interest
  income or non-interest income. You're satisfied when you have a clear,
  numbers-backed answer to that specific question.
max_turns: 6
expected_facts:
  - "NCB Financial Group is mentioned by name or symbol"
  - "Revenue or net interest income figures for at least 2 fiscal years"
  - "Some breakdown between interest and non-interest income"
api_options:
  memory_enabled: true
  enable_financial_data: true
opening_style: cold_open
```

**Negative — `negative_chitchat_offtopic.yaml`:**

```yaml
id: negative_chitchat_offtopic
name: "Off-topic chit-chat probe"
category: negative
endpoint: chat_stream
character: |
  You're a curious user trying to use this chatbot like ChatGPT — asking it to
  write a poem, debug Python, or chat about your weekend.
goal: |
  Get the bot to engage in off-topic conversation. You "succeed" if it does.
max_turns: 4
expected_facts:
  - "Response declines off-topic requests or redirects to JSE/financial topics"
api_options:
  enable_web_search: true
  enable_financial_data: true
```

## 10. Viewer

Static HTML + JS + CSS at `evals/viewer/`. No build step.

**Data loading — two supported modes:**

1. **Drag-and-drop**: drop the `runs/<run_id>/` folder onto the page. File
   API reads `manifest.json`, `summary.json`, and all
   `conversations/*.json`. Fully offline, no server.
2. **Local server convenience**: `python -m evals.serve --port 8765` runs a
   small `http.server` over `evals/`. Then
   `http://localhost:8765/viewer/?run=<run_id>` auto-loads via `fetch`.

Both paths feed the same renderer.

**Three views, tab-switched:**

- **Overview** (reads `summary.json`)
  - Header: run ID, git SHA, started/ended, total cost, mean latency.
  - Verdict pie chart.
  - Per-persona table: rows = personas, columns = mean groundedness /
    factfulness / goal_completion / tool_use / coherence /
    persona_handling / verdict mix, with `mean ± std` across replicates.
    Sortable by any column.
  - Per-endpoint and per-category summary blocks.

- **Conversation list** (reads all `conversations/*.json`)
  - Filterable by persona / category / endpoint / verdict.
  - Color-coded rows: green pass, amber partial, red fail, gray
    judge_failed/incomplete.
  - Click row → conversation detail.

- **Conversation detail**
  - Left: turn-by-turn chat bubbles (persona left, bot right) with per-turn
    latency/cost badges. Expandable per-turn drawer shows sources,
    tools_executed, filters_used, chart preview (Vega-Lite via vega-embed),
    raw response JSON.
  - Right: judge verdict + per-dimension scores with justifications.
    `notable_moments` listed; clicking one scrolls the transcript to that
    turn.
  - Banner: termination reason (`done` / `cap` / `error`).

**Stack:** plain HTML/CSS/JS. CDN deps: **Vega-Lite + Vega-Embed** for
chart specs returned by `/fast_chat_v2`. Optionally **Pico.css** (~4KB) for
defaults; skip if fully custom styling is preferred.

## 11. Operational details

- **Auth:** Gemini API key read from env var named in
  `gemini_api_key_env` (default `GOOGLE_API_KEY`), matching the existing
  app's convention.
- **Git SHA capture:** `manifest.json.git_sha` from
  `git rev-parse HEAD` at run start. Errors don't block runs.
- **`runs/` gitignore:** add `evals/runs/` to `.gitignore` so output
  artifacts don't pollute history. Viewer can still drag-load anything.
- **No CI integration in v1.** Suite is invoked manually. Document the
  command in `evals/README.md`.
- **Logging:** runner logs to stdout (progress per conversation: started,
  done, judge_done). No structured logging file in v1.

## 12. Risks and open questions

- **Judge non-determinism.** Even at temperature 0.2 + structured output,
  scores can drift across runs. The snapshot-only outcome model (no
  thresholds) accepts this. If we later add baselines, judge variance must
  be characterized first by running the same git SHA twice and measuring
  score deltas.
- **Cost surprises.** Pro 2.5 pricing changes would hit the judge budget
  most. The hard `max_cost_usd_per_run` cap is the safety net.
- **Streaming endpoint completeness.** `/chat/stream` may emit
  partial/intermediate states; the assembler needs to match the response
  shape `/agent/chat` produces. The implementation plan should include a
  smoke test against the real endpoint before wiring it into the runner.
- **Persona overfitting.** If personas become too prescriptive (essentially
  scripts), the suite stops surfacing novel failures. Periodically the
  team should review whether personas still encode realistic intent rather
  than implementation knowledge.

## 13. Out of scope (deferred)

- Baseline comparison / two-run diffing.
- CI gate with thresholds.
- Multi-run dashboard.
- Persona authoring UI.
- Editing or kicking off runs from the viewer.
- Sharing / hosted viewer.
- Per-turn judging.
