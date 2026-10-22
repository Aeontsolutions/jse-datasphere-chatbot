# Which Gemini model really answers `/chat/stream`?

Date: 2026-10-22. Stage 1, change 2 of `chatbot-latency-improvement-plan.md`.

## Answer

**`gemini-3.7-flash` is served by `gemini-3.8-flash`.** This is confirmed by direct calls, not inferred from docs. Both the router/refusal calls and the grounded generation call use that name, so all three are answered by 3.8 Flash.

Nothing in our code or logs showed this. The app never read `model_version`, so `interactions.model` and `phase_costs_json[].model` hold the name we **asked for**.

## Evidence

### 1. Direct calls (2026-10-22, google-genai SDK, key from local `fastapi_app/.env`, same model strings and thinking config as the app)

| Requested | `response.model_version` | Notes |
|---|---|---|
| `gemini-3.7-flash` (router config: `thinking_level=LOW`, 256 max tokens) | `gemini-3.8-flash` | 51 thinking tokens, 2.2s |
| `gemini-3.7-flash` (Search-grounded) | `gemini-3.8-flash` | |
| `gemini-3.8-flash` | `gemini-3.8-flash` | control |
| `gemini-3.6-flash` | `gemini-3.6-flash` | control: the field is not simply always "3.8" |
| `gemini-3.5-flash` | **`gemini-3.6-flash`** | second alias found |
| `gemini-3.7-flash-001` | 404 | no dated or pinned 3.7 name exists |

`models.list()` shows `gemini-3.5-flash`, `3.6-flash`, `3.7-flash`, `3.8-flash`. No separate 3.7 can be requested. **Pinning "real 3.7" is not possible.**

### 2. Same-sized A/B of the grounded synthesis call (3 prompts per model, default thinking, no prompt cache; indicative only, n=3)

| Requested | Served by | Latency (s) | Output tokens | Thinking tokens |
|---|---|---|---|---|
| `gemini-3.7-flash` | 3.8 | 17.2 / 19.3 / 7.3 | 682 / 634 / 154 | 1034 / 1207 / 472 |
| `gemini-3.8-flash` | 3.8 | 22.7 / 16.8 / 11.3 | 980 / 578 / 162 | 965 / 967 / 632 |
| `gemini-3.6-flash` | 3.6 | 13.1 / 12.0 / 6.8 | 820 / 632 / 154 | 1128 / 1221 / 625 |

3.6 was about 25% faster per call with similar token counts. With n=3 and Google-side variance this is a hint, not a result.

### 3. Our own interactions table (`chat_stream`, BigQuery)

Same kind of dev traffic (eval personas, similar answer length of about 1,700 chars), before and after:

| Slice | n | total p50 / p95 | avg output tokens | avg thinking tokens |
|---|---|---|---|---|
| dev, 2026-09-02 and 09-05 | 64 / 758 | 4.2s / 13.6s, 5.0s / 11.0s | 431 / 428 | 444 / 445 |
| dev, 2026-10-13 | 78 | 13.7s / 30.6s | 445 | 678 |
| dev, 2026-10-22 | 80 | 14.0s / 31.2s | 454 | 645 |

- Median latency is about 3x higher. Thinking tokens are up about 50%. **Visible output length did not change.**
- Prod by week (`output_tokens`, `thinking_tokens`): Sept stays at 530-670 output and 500-750 thinking. The week of 10-12 reads 1364 and 1471 on only 9 requests.
- Per-phase `latency_ms` (dev, 2026-10-22, PR 112): routing p50 2.0s / p95 3.4s, symbol extraction p50 3.1s, **generation p50 13.7s / p95 28.0s**. Generation is nearly the whole request. Router `thinking_tokens` average 63, so the LOW floor is honoured on the routed model.

## Does it explain the October slowdown?

**Plausible, not proven.** For: 3.7 is confirmed to be served by 3.8 now; 3.8 is known to be slower and to think more (benchmarks note); our thinking tokens rose about 50% and latency rose about 3x on comparable traffic; the generation phase is where the time goes. Against or unknown:

- **We cannot tell when the re-route started.** Google's page was updated 2026-10-13, and our data has no served-model field before this change. Our earliest per-phase rows are 2026-10-13.
- The Sept and Oct dev slices differ in date, load and query mix. The comparison is suggestive, not controlled.
- Tokens alone do not explain a 3x slowdown (thinking +50%, output flat). The rest would be lower tokens per second or longer time to first token. That fits what the benchmarks say about 3.8, but we did not measure it separately. Provider-side variance and capacity stay open as other causes.
- Prod October n is tiny (about 20 requests).

## Recommendation

1. **Ship the logging change** (PR attached to this work): `model_version` is now stored next to `model` in every phase record, and costs nothing. This closes the blind spot, and shows when the served model changes.
2. **Do not "pin 3.7"**; it does not exist. The real choice is:
   - Keep `gemini-3.7-flash` and accept 3.8 (status quo; the name hides what runs and can change again without notice).
   - Name it honestly as `gemini-3.8-flash`. Same behaviour, no surprise later. Note: `resolve_no_thinking` matches by prefix, so add a `"gemini-3.8"` entry (`LOW`; tested directly on `gemini-3.8-flash`: LOW accepted, MINIMAL returns 400, same as 3.7) at the same time. Without it, 3.8 would fall back to `thinking_budget=0`, which the table says 3.7 silently ignores.
   - **Trial `gemini-3.6-flash`** on dev via `ROUTER_MODEL_NAME` and `SYNTHESIS_MODEL_NAME` (no code change; it is the only model with a separately verified identity that was faster in our probe). Router floor `MINIMAL` is already in the table. Judge quality with the eval suite before any prod move; the prompt caches are bound to the model name, so both env vars must change together.
3. Recommended next step: after the PR is deployed to dev, run the eval suite once with the default and once with the 3.6 override, and compare per-phase `latency_ms` and `thinking_tokens`. That is the controlled test this finding could not do.

## Not confirmed

- The date routing began.
- That routing, rather than load or provider variance, caused the October prod p50/p95 change.
- Whether prod's key behaves like the dev/local key (same alias in the public Gemini API; not tested with the prod key, which this task did not touch).
- The model behind `symbol_extraction` (`extract_companies_from_query_async` in `document_selector`); its phase record has an empty model and no `model_version`.
- CloudWatch logs were not used: the app never logged the served model, so they could not answer this.
