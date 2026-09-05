# Is internet grounding enough to disambiguate companies?

**Date:** 2026-09-05
**Status:** Evidence gathered; decision open
**Branch:** `claude/ticker-disambiguation-grounding-bb419c` (stacked on PR #91)

## Question

Prod refused a question about Quantas Advantage (JSE: QAINC, listed 2026) as
non-JSE, reading it as Qantas the airline. The identity list that was supposed
to prevent this is derived from the financial-statements table in BigQuery, so
a company that has not filed annual results yet cannot be in it, and keeping it
current is manual.

That raised the question this experiment answers: **does the curated company
knowledge still earn its keep, or would Google Search grounding alone identify
the right company?**

## Method

AgentV2 carries company identity in two independent places, so removing one
while leaving the other would measure very little. Two flags, run as separate
arms against the same build (see `fastapi_app/app/agent_v2.py`):

| Arm | `DISABLE_REGISTRY_GROUNDING` | `DISABLE_PROMPT_COMPANY_LIST` | Removes |
|---|---|---|---|
| baseline | — | — | nothing (control) |
| A | true | — | BigQuery symbol extraction + AEO-25 verified-metadata block |
| B | true | true | the above, plus the ~30-ticker sector list in `SYSTEM_PROMPT` |

Instrument: the five disambiguation personas from PR #91, 3 replicates each,
against a local server (`127.0.0.1:8000`) so all three arms share one
environment. A local baseline was run rather than reusing the dev numbers from
PR #91 — otherwise any delta could be localhost vs ECS rather than the flags.

Flags were verified live, not just in unit tests:

- Baseline log: 40 `Company extraction completed` events. Arm A: **0**.
- Arm B process: `Key JSE Stocks by Sector` and `NCBFG` absent from
  `SYSTEM_PROMPT`; `SAFETY RULES` and `SCOPE RULES` still present.

## Results

| Persona | baseline | arm A | arm B |
|---|---|---|---|
| `ticker_shared_with_foreign_issuer` | pass ×3 | pass ×3 | pass ×3 |
| `jse_is_jamaica_not_johannesburg` | pass ×3 | pass ×3 | pass ×3 |
| `same_brand_different_listing` | pass ×3 | pass ×3 | pass ×3 |
| `negative_foreign_namesake_via_jse_ticker` | pass ×3 | pass ×3 | pass ×3 |
| `recent_listing_not_in_database` | fail ×3 | fail ×2, partial | fail ×3 |

**Removing every piece of curated company knowledge changed nothing measurable.**

Arm B transcript, with no registry and no ticker list:

> "Here is an update on **Carreras Limited (JSE: CAR)** based on recent corporate
> actions and Jamaica Stock Exchange (JSE) filings…"
> "Here is the latest update on **Key Insurance Company Limited (JSE: KEY)**…"

The MDS case that motivated AEO-25 — prod once claimed the ticker was shared
between Medical Disposables & Supplies and Mailpac Group — was probed directly
on arm B, three times. All three resolved to Medical Disposables & Supplies
Limited; none mentioned Mailpac.

## What this does and does not show

**Does show.** For companies that exist on the public web, Google Search
grounding resolves JSE ticker identity as well as the BigQuery registry does,
including against larger foreign namesakes (Avis Budget, KeyCorp, JSE Ltd
Johannesburg, Sagicor Financial Company). On this evidence the registry is not
what makes disambiguation work today.

**Does not show.**

- The recent-listing failure is unchanged in every arm, because it is a
  *router* defect, not an identity-resolution one. The router refuses before
  any identity machinery runs. Removing the registry neither helps nor hurts it.
- Four collision pairs were tested, all long-established companies with strong
  web presence. Nothing here speaks to a thinly-covered junior-market listing.
- The AEO-25 hallucination was originally observed on gemini-2.5-pro; these runs
  are gemini-3.7-flash. The registry may be redundant *for this model* rather
  than redundant in principle. A model migration should re-run these arms
  before assuming the result carries over.
- The registry is still load-bearing for `/fast_chat_v2`, which queries
  BigQuery for actual financial data. These flags only touch identity grounding
  in AgentV2.

## Implication

The registry is not paying for itself as a disambiguation mechanism on the
current model, and it is the thing that goes stale. The recency defect it
causes is real and reproducible; the collision protection it provides is not
currently distinguishable from what search does for free.

That points away from "keep the list current by hand" and toward fixing the
router: resolve entity identity before the scope decision, and fail open on an
unrecognised proper noun rather than refusing it.

## Reproducing

```bash
cd fastapi_app
DISABLE_REGISTRY_GROUNDING=true DISABLE_PROMPT_COMPANY_LIST=true \
  PYTHONPATH=. python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```bash
PYTHONPATH="$PWD" python scripts/run_eval.py --base-url http://127.0.0.1:8000 \
  --personas-dir "$PWD/evals/personas" --output-dir "$PWD/evals/runs" \
  --replicates 3 --run-id local_arm_b --persona disambiguation_ticker_shared_with_foreign_issuer
```

`PYTHONPATH` and `--personas-dir` are both required: `run_eval.py` otherwise
resolves the `evals` package through the editable install, which points at
whichever worktree installed it last — personas *and* harness code.
