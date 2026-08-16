# ADR 0002: Bound Gemini thinking budgets, and fail closed on an unusable router verdict

## Status

Accepted. Fixed in `agent_v2.py` (`/chat/stream`), with supporting changes in
`utils/cost_tracking.py`, `models.py`, `interaction_log.py`, `main.py`, and
`scripts/create_interactions_table.py`.

## Context

Interaction `7dba1a26bc014fdebfdb2b9567012c52` (request
`beb31830-8854-4c24-aa82-17b972737ea4`, `chat_stream`, 2026-08-15 02:00:46)
returned a response that stopped mid-sentence:

> Recent geopolitical events can certainly have a ripple effect on companies

The row logged `success: true`, `output_tokens: 13`, `cost_usd: 0.0000693`,
`error_message: null`. Nothing in the record indicated a failure.

Investigation found three distinct defects, one of which was hidden by another.

### 1. Thinking tokens consumed the entire output budget

The query routed `REFUSE`, so `_fast_path()` generated the refusal with
`gemini-2.5-flash` at `max_output_tokens=256`.

On Gemini 2.5 models `max_output_tokens` bounds **thinking and visible text
together**, and thinking is enabled by default with a *dynamic* budget.
Reproduced against the live API, 3/3 runs — run 1 matches the logged response
word for word:

| run | finish_reason | thinking | visible | text |
|-----|---------------|----------|---------|------|
| 1 | `MAX_TOKENS` | 241 | 11 | "…ripple effect on companies" |
| 2 | `MAX_TOKENS` | 244 | 8  | "…have ripple effects" |
| 3 | `MAX_TOKENS` | 242 | 10 | "…ripple effects on companies" |

Reasoning consumed ~94% of the ceiling; generation was cut off after ~10
tokens of prose.

Critically, **raising `max_output_tokens` does not fix this**. The dynamic
budget expands to fill whatever ceiling it is given — at
`max_output_tokens=1024` the model thought for 908 tokens and truncated again.
Only an explicit `thinking_config.thinking_budget` bounds it.

This also explains why the only other refusal in the table survived: "Should I
buy Tesla stock?" is trivially out of scope (60 thinking tokens), while the
geopolitical query is a borderline judgement call (241). **The harder the
refusal is to reason about, the more likely it was to truncate** — precisely
backwards from what the system needs.

### 2. The refusal prompt did not actually compel a refusal

Testing candidate fixes surfaced a second, independent defect that the
truncation had been masking. In *every* configuration that ran to completion,
Flash answered the question rather than refusing it — recommending JSE
"defensive stocks": utilities, telecommunications, consumer staples, local
food producers. That is exactly the personalised investment advice the router
had flagged.

The cause was the wording of `REFUSAL_FLASH_PROMPT`. It described only *how*
to refuse **if** a request was out of scope; it never stated that this request
*was* out of scope. The router's `REFUSE` verdict was never passed into the
call, so Flash re-litigated the decision and complied.

The user-visible consequence is worth stating plainly: fixing the token budget
alone would have converted a truncated sentence into fluent, non-compliant
investment advice. The bug we were asked to fix was suppressing a worse one.

### 3. The router failed open

`_fast_path()` treated any verdict not containing `REFUSE` as `ALLOW`. An
empty string satisfies that condition, so a truncated or empty router response
would have silently permitted an unclassified query through to Pro.

The router call shared the same 256-token config, and measured 96–120 thinking
tokens against a 2-token verdict. It did not truncate in practice, but the
margin was set by luck rather than design, and the failure mode was silent.

### Why none of this was visible

- `TokenUsage.from_response()` read `candidates_token_count`, which **excludes**
  reasoning tokens. The 241 thinking tokens never reached BigQuery, so the row
  looked like a cheap, tiny, successful call.
- `finish_reason` was never read anywhere in the live application, so
  `MAX_TOKENS` was indistinguishable from `STOP`.
- Cost was computed from visible output only. Thinking bills at the output
  rate, so every reasoning-heavy call was under-reported — this interaction by
  roughly 4×.

## Decision

**Bound thinking explicitly on every Flash call, and keep headroom for the
answer.** `agent_v2.py` now defines the budgets as named constants
(`ROUTER_MAX_OUTPUT_TOKENS=512`/`ROUTER_THINKING_BUDGET=256`,
`REFUSAL_MAX_OUTPUT_TOKENS=1024`/`REFUSAL_THINKING_BUDGET=256`) with a
`MIN_VISIBLE_OUTPUT_HEADROOM=256` invariant asserted by a unit test. Sizing is
empirical: observed reasoning peaks at ~120 tokens for routing and ~240 for
refusals.

**State the verdict as settled fact in the refusal prompt.**
`REFUSAL_FLASH_PROMPT` now tells the model the request has already been
determined out of scope, that the decision is not its to revisit, and that it
must not answer in whole or in part — including hedged or caveated guidance.

**Fail closed when the router yields no usable verdict.** Routing moved into
`_route_verdict()`, which returns `ALLOW`, `REFUSE`, or `REFUSE_UNVERIFIED`. A
blank verdict earns one retry with `thinking_budget=0` — guaranteeing the full
ceiling is available for the single word required — and if that also comes back
empty the request is refused using canned text, without a further call to the
same misbehaving model.

**Make truncation observable.** `TokenUsage` now carries `thinking_tokens`;
`get_finish_reason()`/`was_truncated()` expose the finish reason; `PhaseCost`
and `CostSummary` carry `finish_reason`/`truncated`; and the `interactions`
table gains `thinking_tokens`, `finish_reason`, and `truncated` columns. A
truncated phase also emits a `logger.warning`.

**Bill thinking tokens as output**, matching how Google actually charges.

### Two deliberate non-changes

**Exceptions still fall through to the web path.** ADR 0001's graceful
degradation is unchanged, and `test_router_exception_still_falls_through_to_web`
locks it in. A transport error means the check never ran, and Pro's own system
prompt still constrains scope; a *successful* call returning nothing is a
different, more suspicious signal. Only the latter now fails closed.

**The Pro generation call's token config is untouched.** It runs at
`max_output_tokens=8192` with unbounded dynamic thinking, so it is
theoretically subject to the same defect. We have no evidence of it occurring,
and capping Pro's reasoning is a quality trade-off that should be made on data
rather than speculation. It is now instrumented, so if it does truncate the
logs will say so.

## Consequences

- Refusals complete. Verified end-to-end against the live API on the original
  query: `finish_reason=STOP`, 50 visible tokens, 231 thinking tokens, not
  truncated.
- Refusals refuse. The same run declines without naming a sector or ticker.
- Reported cost rises for reasoning-heavy calls — the previous figures were
  wrong, not lower. The verification run costs $0.00030 against the $0.00007
  originally logged for the broken one.
- Capping router reasoning at 256 tokens could in principle degrade
  classification on an unusually subtle query. Observed usage peaks at 120, so
  the cap is not currently binding.
- `interactions` gains three nullable columns. Rows written before this change
  have `NULL` for all three; `truncated IS NULL` means "not recorded", not
  "not truncated".

### Migration

`scripts/create_interactions_table.py` previously no-op'd when the table
existed, which would have silently dropped the new fields on every insert. It
now reconciles an existing table with `INTERACTIONS_SCHEMA`, adding missing
columns and refusing to auto-apply any non-additive change. **It must be run
once per deployed environment before this change ships**, or inserts will fail
on the unknown columns:

```bash
python scripts/create_interactions_table.py --project jse-datasphere --dataset jse_raw_financial_data_dev_elroy
```

## Detecting a recurrence

```sql
SELECT timestamp, endpoint, finish_reason, thinking_tokens, output_tokens, query
FROM `jse-datasphere.jse_raw_financial_data_dev_elroy.interactions`
WHERE truncated
ORDER BY timestamp DESC
```

A healthy row has `thinking_tokens` comfortably below the phase's budget. The
warning sign this ADR exists to catch is `thinking_tokens` approaching the
budget while `output_tokens` stays small.
