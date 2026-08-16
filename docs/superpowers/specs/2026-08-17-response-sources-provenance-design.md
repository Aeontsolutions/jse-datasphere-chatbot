# Response Sources & Provenance

**Date:** 2026-08-17
**Status:** Design — approved approach, pending spec review

## Context

A request came in to return the sources each endpoint used to answer a query.
Users need this to fact-check answers; the UI needs it to render a citations
panel.

The feature is roughly a quarter built today, and only on one endpoint.

| Endpoint | Answers from | Provenance returned today |
|---|---|---|
| `/chat/stream` (AgentV2) | Gemini 2.5 Pro + Google Search grounding | `sources: [{type, title, url}]` and `web_search_results` |
| `/fast_chat_v2` | BigQuery financial table | None — only `data_preview` and `filters_used` |
| `/chat` | PDFs in S3 | Filenames only (`documents_loaded`), no URLs, no pages |

### Facts about the current code

- `AgentChatResponse.sources` is typed `Optional[List[Dict[str, Any]]]`
  ([`models.py:253`](../../../fastapi_app/app/models.py)). There is no schema, so
  the UI cannot rely on any key being present.
- The only producer is `AgentV2._extract_grounding_metadata`
  ([`agent_v2.py:268`](../../../fastapi_app/app/agent_v2.py)), which emits
  `{type: "web", title, url}` per grounding chunk. It does not dedupe, and
  Gemini repeats the same chunk across grounding supports — so the list
  currently contains duplicates.
- Gemini's grounding URIs are `vertexaisearch.cloud.google.com` redirects that
  expire in roughly 30 days.
- `grounding_supports` (Gemini's per-claim segment→chunk mapping) is available on
  the response and is currently discarded. It is **not** used in this design; see
  "Rejected alternatives".
- `FinancialDataResponse` and `ChatResponse` have no `sources` field at all.
- Documents are private `s3://bucket/key` paths. There is no presigned-URL
  support anywhere in the codebase
  ([`s3_client.py:58`](../../../fastapi_app/app/s3_client.py)).
- `auto_load_relevant_documents`
  ([`document_selector.py:479`](../../../fastapi_app/app/document_selector.py))
  has `doc_info` with `document_link` and `reason` in scope but returns only
  filenames.
- Both `/fast_chat_v2` and `/chat/stream` cache responses in Redis.
  `/chat/stream` already persists `sources` in its cache payload
  ([`main.py:1013`](../../../fastapi_app/app/main.py)); `/fast_chat_v2` has no
  sources to persist yet.
- [`mock_client/chat_assistant_test_client.html:944`](../../../mock_client/chat_assistant_test_client.html)
  renders sources by reading `s.description` — a key the API has never emitted.
  The test client's source display has therefore always been blank.

## Decisions

Settled during brainstorming:

1. **All three endpoints** return sources, not just `/chat/stream`.
2. **Flat source list** per response, not inline per-claim citations.
3. **Opaque document id plus a resolver endpoint**, not inline presigned URLs.
4. **One polymorphic `sources` array** with a `type` discriminator, not separate
   per-type arrays and not a strict discriminated union.

## Design

### The `Source` model

Added to [`models.py`](../../../fastapi_app/app/models.py):

```python
class SourceType(str, Enum):
    WEB = "web"            # Google Search grounding chunk
    DOCUMENT = "document"  # PDF in S3
    DATASET = "dataset"    # BigQuery financial table


class Source(BaseModel):
    type: SourceType
    title: str                            # always present, always human-readable
    detail: Optional[str] = None          # why this source was used
    retrieved_at: Optional[str] = None    # ISO 8601 timestamp

    # web
    url: Optional[str] = None             # Gemini redirect URI — expires (~30d)
    domain: Optional[str] = None          # e.g. "jamstockex.com"

    # document
    document_id: Optional[str] = None     # resolve via GET /documents/{id}
    company: Optional[str] = None
    year: Optional[str] = None

    # dataset
    table: Optional[str] = None           # "project.dataset.table"
    filters: Optional[Dict[str, Any]] = None
    record_count: Optional[int] = None
```

`type` and `title` are the only required fields. A client that understands
nothing else can always render a meaningful citation line.

`retrieved_at` is the time the underlying retrieval ran, not the time the
response was serialized: the Gemini call for web sources, the BigQuery query for
dataset sources, the S3 fetch for document sources. On a cache hit it is
replayed from the cached payload, so it correctly reports when the data was
actually fetched rather than when it was re-served.

For dataset sources, `filters` carries the four query-shaping fields from
`FinancialDataFilters` — `companies`, `symbols`, `years`, `standard_items`. The
conversational fields (`interpretation`, `is_follow_up`, `context_used`,
`data_availability_note`, `unrecognized_items`) are excluded: they describe how
the query was understood, not which data was read, and `filters_used` already
carries the full object for clients that want them.

### Response model changes

- `AgentChatResponse.sources` changes type from `Optional[List[Dict[str, Any]]]`
  to `Optional[List[Source]]`. The serialized JSON for a web source is
  unchanged.
- `FinancialDataResponse` gains `sources: Optional[List[Source]] = None`.
- `ChatResponse` gains `sources: Optional[List[Source]] = None`.

Two choices that are load-bearing:

- **`domain` is not redundant with `url`.** Grounding URIs expire. When the link
  dies, `domain` plus `title` still tells the user who said it — the citation
  degrades to text rather than to nothing. The alternative, resolving each
  redirect to its publisher URL, costs an HTTP round-trip per source on the hot
  path and is not worth it.
- **Dataset sources cite the query, not a row.** Honest provenance for
  `/fast_chat_v2` is "this came from `<table>`, filtered to NCBFG / 2023 /
  net_profit, 4 rows matched". Both `filters` and `record_count` are already
  computed; this only surfaces them.

### Backward compatibility

Today's web sources emit `{type, title, url}`. All three fields survive
unchanged, and everything else is additive, so existing consumers keep working.
`documents_loaded` on `ChatResponse` is retained alongside the new `sources`
field for the same reason.

### Per-endpoint wiring

**`/chat/stream`** — `_extract_grounding_metadata` builds `Source` objects
instead of raw dicts. Two fixes land with it:

- Dedupe chunks by URL.
- Populate `domain` from `web.domain`, falling back to `web.title` (the SDK puts
  the site domain in `title` for most chunks).

The refusal fast path continues to return `sources=None`.

**`/fast_chat_v2`** — add `sources` to `FinancialDataResponse`. After
`query_data`, build a single dataset `Source` from the table identity on
`FinancialDataManager` (its `project_id` / `dataset` / `table` env config), the
applied `filters`, and `len(results)`. Add `sources` to the Redis cache payload
at [`main.py:707`](../../../fastapi_app/app/main.py) — cache-safe, because
nothing in a dataset source expires.

**`/chat`** — add `sources` to `ChatResponse`, keeping `documents_loaded`.

`auto_load_relevant_documents` must return the document metadata it currently
discards. It already returns a 3-tuple and now needs a 4th value, so its return
type changes to a small `DocumentLoadResult` dataclass with fields `texts`,
`message`, `loaded_docs`, and `sources`. Growing the tuple instead would break
the same call sites (tuple unpacking fails on any arity change), so the readable
option wins at equal cost.

Seven call sites are updated. Five unpack the tuple directly
([`main.py:477`](../../../fastapi_app/app/main.py),
`_archive/streaming_chat.py:166`, `tests/test_async_s3_downloads.py:407` and
`:455`, `test_async_integration.py:149`). Two assert its arity explicitly —
`tests/unit/test_document_selector.py:53` and `:64` both check
`len(result) == 3` — and become assertions on the dataclass fields instead.

The same change is applied to `auto_load_relevant_documents_async`. That
function is used only by archived code and tests today, but letting the pair
diverge is how it rots.

### Document resolver endpoint

```
GET /documents/{document_id}  →  307 redirect to a presigned S3 URL
```

- Presigned URL TTL: 5 minutes.
- Backed by a `{document_id: s3_path}` index built once when `metadata.json`
  loads.
- `document_id = sha256(s3_path)[:16]`.
- Unknown id → `404`.

**The id must not be a reversible encoding of the S3 path.** A base64'd
`s3://bucket/key` would let a client edit the id and read arbitrary objects from
the bucket through our own resolver. Because the id is a lookup key rather than
a decodable path, there is no input a client can craft to reach an object
outside `metadata.json`.

This indirection is also what makes sources cacheable: a presigned URL baked
into a Redis-cached response would be dead by the time it was served.

### Test client fix

Update [`chat_assistant_test_client.html`](../../../mock_client/chat_assistant_test_client.html)
to render the real schema (`type`, `title`, `url`/`document_id`) instead of the
non-existent `description` key.

## Testing

- Unit tests per source builder: grounding chunk → `Source`, filters → `Source`,
  `doc_info` → `Source`.
- Grounding dedupe: repeated chunks for one URL yield one source.
- Resolver: a tampered or unknown `document_id` returns `404` and never a
  presigned URL.
- Cache round-trip: sources survive Redis on both `/chat/stream` and
  `/fast_chat_v2`.
- Backward compatibility: a web source still serializes with `type`, `title`,
  and `url`.

Tests follow the existing layout under `fastapi_app/tests/unit/`.

## Known limitation: financial figures cannot cite a statement page

The BigQuery table has no column referencing the PDF a figure was extracted
from. `query_data` selects `Company, Symbol, Year, standard_item, item,
unit_multiplier, item_type, item_name`
([`financial_utils.py:897`](../../../fastapi_app/app/financial_utils.py)) and
that is all the table holds.

So a `/fast_chat_v2` answer can cite the table and the filters, but it cannot
hand a user the statement page a number came from — which is the strongest form
of fact-checking for financial figures. Closing this requires adding a
source-document reference upstream in the extraction pipeline, which is separate
work and out of scope here.

This is recorded rather than papered over, so the dataset citation does not
imply more provenance than it has.

## Rejected alternatives

**Separate typed arrays** (`web_sources`, `document_sources`, `data_sources`) —
cleaner typing with no optional-field soup, but the UI must merge three lists to
render one citations panel, and it orphans the existing `sources` field.

**Strict discriminated union** (Pydantic `Field(discriminator=...)`) — best
OpenAPI output (`oneOf`), but adding a fourth source type later becomes a
breaking change for strictly-validating clients.

**Inline per-claim citations** using `grounding_supports` — genuine
fact-checking, but the char offsets Gemini returns are fragile, and neither the
financial nor the document path has an equivalent, so those would have to be
synthesized. The `Source` schema leaves room to add per-claim spans additively
later.

**Resolving grounding redirects to publisher URLs** — fixes link expiry, but
costs an HTTP round-trip per source on the hot path. `domain` covers the
degraded case at no latency cost.
