# Response Sources & Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every chat endpoint returns a typed `sources` array so users can fact-check answers and the UI can render a citations panel.

**Architecture:** One polymorphic `Source` model with a `type` discriminator (`web` / `document` / `dataset`) is returned by all three chat endpoints. Web sources come from Gemini grounding chunks, dataset sources from the BigQuery filters already computed, and document sources carry a non-forgeable `document_id` resolved on demand by a new `GET /documents/{id}` endpoint that issues short-lived presigned S3 URLs.

**Tech Stack:** Python 3.10–3.12, FastAPI, Pydantic v2, boto3, google-genai, pytest, Redis (response cache), BigQuery.

**Spec:** [`docs/superpowers/specs/2026-08-17-response-sources-provenance-design.md`](../specs/2026-08-17-response-sources-provenance-design.md)

## Global Constraints

- **All API changes are additive.** `type`, `title`, and `url` must keep serializing identically for web sources; `documents_loaded` stays on `ChatResponse`. A backward-compatibility test guards this.
- **Line length 100.** Formatted with `black`, linted with `ruff` (config in `fastapi_app/pyproject.toml`).
- **Python 3.10–3.12**, Pydantic v2 syntax (`model_dump`, not `dict`).
- **`document_id` must never be a reversible encoding of an S3 path.** It is `sha256(s3_path)[:16]`, resolved only by lookup against `metadata.json`.
- **Presigned URL TTL is 300 seconds.** Presigned URLs are never written to the Redis response cache.
- **All commands run from `fastapi_app/`** (pytest is configured with `testpaths = ["tests"]`, `pythonpath = [".", "app"]`, `asyncio_mode = "auto"`, `--strict-markers`).
- **Unit tests carry `@pytest.mark.unit`** and live in `fastapi_app/tests/unit/`.

---

### Task 0: Capture the regression baseline

**Files:**
- Create: `fastapi_app/baseline-tests.txt` (untracked scratch output; not committed)

**Interfaces:**
- Consumes: nothing.
- Produces: `fastapi_app/baseline-tests.txt` — the suite's pass/fail state **before any change**, read by Task 7.

This must run **first**, before a single line is edited. "No regressions" is only meaningful as a comparison against a known starting point, and this repo has pre-existing failures that would otherwise get blamed on this work. Capturing it later does not work: every task commits, so by Task 7 there are no uncommitted changes for `git stash` to set aside and the "baseline" would silently include the whole feature.

- [ ] **Step 1: Confirm the working tree is clean and on the feature branch**

```bash
git status --short && git rev-parse --abbrev-ref HEAD
```

Expected: no output from `git status --short`. If there are uncommitted changes, stop and resolve them — the baseline would be meaningless.

- [ ] **Step 2: Run the full suite and save the result**

From `fastapi_app/`:

```bash
python -m pytest tests -q --no-cov > baseline-tests.txt 2>&1; tail -20 baseline-tests.txt
```

The path is deliberately repo-relative rather than `/tmp/...`: this is a Windows checkout, `/tmp` resolves differently between Git Bash and PowerShell, and native Python resolves it to `C:\tmp` rather than the shell's temp directory.

- [ ] **Step 3: Record the counts**

Note the final summary line (e.g. `12 failed, 340 passed, 5 skipped`) and the names of any failing tests. These are **pre-existing** and are not caused by this work. Task 7 compares against exactly this list.

- [ ] **Step 4: Keep the file out of version control**

`baseline-tests.txt` is scratch output. Do not `git add` it. Confirm it stays untracked:

```bash
git status --short fastapi_app/baseline-tests.txt
```

Expected: `?? fastapi_app/baseline-tests.txt` (untracked), never staged.

---

### Task 1: The `Source` model and response wiring

**Files:**
- Modify: `fastapi_app/app/models.py` (add `SourceType`, `Source`; wire into `ChatResponse:51`, `FinancialDataResponse:131`, `AgentChatResponse:253`)
- Modify: `fastapi_app/app/main.py:1013` (wrap cached sources in `_jsonable`)
- Test: `fastapi_app/tests/unit/test_source_model.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `SourceType` (str enum: `WEB="web"`, `DOCUMENT="document"`, `DATASET="dataset"`) and `Source` (Pydantic model; required `type: SourceType`, `title: str`; all other fields `Optional` defaulting to `None`). Every later task builds `Source` objects.

- [ ] **Step 1: Write the failing test**

Create `fastapi_app/tests/unit/test_source_model.py`:

```python
"""Unit tests for the Source provenance model.

Every chat endpoint returns a `sources` array so users can fact-check an
answer. Only `type` and `title` are required, so a client that understands
nothing else can still render a meaningful citation line.
"""

import pytest
from pydantic import ValidationError

from app.models import (
    AgentChatResponse,
    ChatResponse,
    FinancialDataFilters,
    FinancialDataResponse,
    Source,
    SourceType,
)


@pytest.mark.unit
class TestSourceModel:
    def test_web_source_keeps_legacy_shape(self):
        """Existing clients read type/title/url. Those must serialize verbatim."""
        source = Source(
            type=SourceType.WEB,
            title="Jamaica Stock Exchange",
            url="https://vertexaisearch.example/redirect/abc",
        )
        dumped = source.model_dump()
        assert dumped["type"] == "web"
        assert dumped["title"] == "Jamaica Stock Exchange"
        assert dumped["url"] == "https://vertexaisearch.example/redirect/abc"

    def test_type_and_title_are_sufficient(self):
        source = Source(type=SourceType.DATASET, title="JSE financial statements")
        assert source.url is None
        assert source.document_id is None
        assert source.record_count is None

    def test_title_is_required(self):
        with pytest.raises(ValidationError):
            Source(type=SourceType.WEB)

    def test_agent_response_coerces_source_dicts(self):
        """Redis returns plain dicts; the response model must coerce them."""
        built = AgentChatResponse(
            response="r",
            data_found=True,
            record_count=0,
            sources=[{"type": "web", "title": "T", "url": "https://x"}],
        )
        assert built.sources[0].type == SourceType.WEB
        assert built.sources[0].title == "T"

    def test_financial_response_accepts_sources(self):
        built = FinancialDataResponse(
            response="r",
            data_found=True,
            record_count=2,
            filters_used=FinancialDataFilters(),
            sources=[{"type": "dataset", "title": "JSE financials", "record_count": 2}],
        )
        assert built.sources[0].type == SourceType.DATASET
        assert built.sources[0].record_count == 2

    def test_chat_response_accepts_sources_alongside_documents_loaded(self):
        """documents_loaded is retained for backward compatibility."""
        built = ChatResponse(
            response="r",
            documents_loaded=["ncb_annual_2023.pdf"],
            sources=[{"type": "document", "title": "ncb_annual_2023.pdf", "document_id": "abc123"}],
        )
        assert built.documents_loaded == ["ncb_annual_2023.pdf"]
        assert built.sources[0].document_id == "abc123"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_source_model.py -v --no-cov`
Expected: FAIL with `ImportError: cannot import name 'Source' from 'app.models'`

- [ ] **Step 3: Add the model to `app/models.py`**

Insert after the `ChartSpec` class (currently ends at line 128), before `FinancialDataResponse`:

```python
class SourceType(str, Enum):
    """Kind of evidence backing an answer."""

    WEB = "web"  # Google Search grounding chunk
    DOCUMENT = "document"  # PDF in S3
    DATASET = "dataset"  # BigQuery financial table


class Source(BaseModel):
    """A single piece of provenance for an answer.

    Only `type` and `title` are required — a client that understands nothing
    else can still render a citation line. Per-type fields are optional so new
    source kinds can be added without breaking existing consumers.
    """

    type: SourceType = Field(..., description="Kind of source backing the answer")
    title: str = Field(..., description="Human-readable source name")
    detail: Optional[str] = Field(default=None, description="Why this source was used")
    retrieved_at: Optional[str] = Field(
        default=None, description="ISO 8601 time the underlying retrieval ran"
    )

    # --- web ---
    url: Optional[str] = Field(
        default=None, description="Source URL. Gemini grounding URIs expire (~30 days)."
    )
    domain: Optional[str] = Field(
        default=None, description="Publisher domain. Outlives an expired url."
    )

    # --- document ---
    document_id: Optional[str] = Field(
        default=None, description="Opaque id. Resolve via GET /documents/{document_id}"
    )
    company: Optional[str] = Field(default=None, description="Company the document belongs to")
    year: Optional[str] = Field(default=None, description="Reporting year of the document")

    # --- dataset ---
    table: Optional[str] = Field(default=None, description="Fully-qualified BigQuery table")
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Filters applied when reading the table"
    )
    record_count: Optional[int] = Field(default=None, description="Rows matched by the query")
```

- [ ] **Step 4: Wire `sources` into the three response models**

In `ChatResponse` (line 51), add after `conversation_history`:

```python
    sources: Optional[List[Source]] = Field(
        default=None, description="Provenance for the answer (document sources)"
    )
```

In `FinancialDataResponse` (line 131), add after `chart`:

```python
    sources: Optional[List[Source]] = Field(
        default=None, description="Provenance for the answer (dataset sources)"
    )
```

In `AgentChatResponse` (line 253), replace the existing untyped `sources` field:

```python
    sources: Optional[List[Source]] = Field(
        default=None, description="Provenance for the answer (web sources)"
    )
```

- [ ] **Step 5: Fix the Redis serialization of sources**

`main.py:1013` currently writes `"sources": response.sources` into the cache payload. That works only while sources are plain dicts — once they are `Source` objects, the Redis JSON write fails. Change that line to match its neighbours:

```python
                    "sources": _jsonable(response.sources),
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_source_model.py -v --no-cov`
Expected: PASS (6 tests)

- [ ] **Step 7: Verify no regression in existing tests**

Run: `python -m pytest tests/unit tests/test_chat_stream_endpoint.py -v --no-cov`
Expected: PASS, same count as before this task

- [ ] **Step 8: Commit**

```bash
git add fastapi_app/app/models.py fastapi_app/app/main.py fastapi_app/tests/unit/test_source_model.py
git commit -m "feat(models): add typed Source model for answer provenance"
```

---

### Task 2: Web sources from Gemini grounding

**Files:**
- Modify: `fastapi_app/app/agent_v2.py:268-327` (`_extract_grounding_metadata`)
- Test: `fastapi_app/tests/unit/test_agent_v2_sources.py`

**Interfaces:**
- Consumes: `Source`, `SourceType` from Task 1.
- Produces: `AgentV2._extract_grounding_metadata(response) -> Dict[str, Any]` with key `"sources"` now holding `List[Source]` (was `List[dict]`) and `"search_results"` unchanged.

Two bugs are fixed alongside the model change: Gemini repeats the same grounding chunk across grounding supports, so the current list contains **duplicates**; and `domain` must be populated so a citation survives its URL expiring.

- [ ] **Step 1: Write the failing test**

Create `fastapi_app/tests/unit/test_agent_v2_sources.py`:

```python
"""Unit tests for web source extraction from Gemini grounding metadata.

Gemini returns one grounding chunk per grounding support, so the same URL
appears several times in a single response — the extracted source list must
be deduped. Grounding URIs are vertexaisearch redirects that expire after
about 30 days, so `domain` is captured separately: when the link dies, the
citation degrades to readable text instead of to nothing.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.agent_v2 import AgentV2
from app.models import SourceType


def _web_chunk(uri, title, domain=None):
    """Build a grounding chunk.

    `spec` is required: a bare MagicMock auto-creates any attribute, so
    `getattr(web, "domain", None)` would return a Mock instead of None and the
    fallback path would never be exercised.
    """
    spec = ["uri", "title"] + (["domain"] if domain is not None else [])
    web = MagicMock(spec=spec)
    web.uri = uri
    web.title = title
    if domain is not None:
        web.domain = domain
    chunk = MagicMock(spec=["web"])
    chunk.web = web
    return chunk


def _grounded_response(chunks):
    grounding = MagicMock(spec=["grounding_chunks", "web_search_queries", "search_entry_point"])
    grounding.grounding_chunks = chunks
    grounding.web_search_queries = ["ncb net profit 2023"]
    entry_point = MagicMock(spec=["rendered_content"])
    entry_point.rendered_content = "<div></div>"
    grounding.search_entry_point = entry_point
    candidate = MagicMock(spec=["grounding_metadata"])
    candidate.grounding_metadata = grounding
    response = MagicMock(spec=["candidates"])
    response.candidates = [candidate]
    return response


@pytest.fixture
def agent():
    with patch("app.agent_v2.get_genai_client"):
        yield AgentV2()


@pytest.mark.unit
class TestWebSourceExtraction:
    def test_duplicate_urls_are_deduped(self, agent):
        response = _grounded_response(
            [
                _web_chunk("https://jse.com/a", "jamstockex.com"),
                _web_chunk("https://jse.com/a", "jamstockex.com"),
                _web_chunk("https://jse.com/b", "jamstockex.com"),
            ]
        )
        sources = agent._extract_grounding_metadata(response)["sources"]
        assert len(sources) == 2
        assert [s.url for s in sources] == ["https://jse.com/a", "https://jse.com/b"]

    def test_sources_are_typed_web(self, agent):
        response = _grounded_response([_web_chunk("https://jse.com/a", "jamstockex.com")])
        source = agent._extract_grounding_metadata(response)["sources"][0]
        assert source.type == SourceType.WEB
        assert source.title == "jamstockex.com"

    def test_domain_field_is_preferred_when_present(self, agent):
        response = _grounded_response(
            [_web_chunk("https://x/a", "JSE Annual Review", domain="jamstockex.com")]
        )
        source = agent._extract_grounding_metadata(response)["sources"][0]
        assert source.domain == "jamstockex.com"
        assert source.title == "JSE Annual Review"

    def test_domain_falls_back_to_title(self, agent):
        response = _grounded_response([_web_chunk("https://x/a", "jamstockex.com")])
        source = agent._extract_grounding_metadata(response)["sources"][0]
        assert source.domain == "jamstockex.com"

    def test_retrieved_at_is_populated(self, agent):
        response = _grounded_response([_web_chunk("https://x/a", "jamstockex.com")])
        source = agent._extract_grounding_metadata(response)["sources"][0]
        assert source.retrieved_at is not None
        assert "T" in source.retrieved_at  # ISO 8601

    def test_chunks_without_uri_are_skipped(self, agent):
        response = _grounded_response([_web_chunk("", "jamstockex.com")])
        assert agent._extract_grounding_metadata(response)["sources"] == []

    def test_no_candidates_yields_no_sources(self, agent):
        response = MagicMock(spec=["candidates"])
        response.candidates = []
        assert agent._extract_grounding_metadata(response)["sources"] == []

    def test_search_results_still_populated(self, agent):
        """web_search_results is a separate field and must not change shape."""
        response = _grounded_response([_web_chunk("https://x/a", "jamstockex.com")])
        results = agent._extract_grounding_metadata(response)["search_results"]
        assert results["queries"] == ["ncb net profit 2023"]
        assert results["grounding_chunks"] == [{"title": "jamstockex.com", "uri": "https://x/a"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_agent_v2_sources.py -v --no-cov`
Expected: FAIL — `test_duplicate_urls_are_deduped` gets 3 sources instead of 2, and `AttributeError: 'dict' object has no attribute 'url'` on the typed assertions.

- [ ] **Step 3: Rewrite the chunk loop in `_extract_grounding_metadata`**

In `app/agent_v2.py`, replace lines 296–320 (from `# Extract grounding chunks (web sources)` through the `if grounding_chunks:` block) with:

```python
        # Extract grounding chunks (web sources).
        # Gemini emits one chunk per grounding support, so the same URL can
        # appear many times — dedupe while preserving first-seen order.
        chunks = getattr(grounding, "grounding_chunks", []) or []
        grounding_chunks = []
        retrieved_at = datetime.now(timezone.utc).isoformat()
        seen_urls = set()

        for chunk in chunks:
            if not (hasattr(chunk, "web") and chunk.web):
                continue
            web = chunk.web
            uri = getattr(web, "uri", "") or ""
            title = getattr(web, "title", "") or ""
            grounding_chunks.append({"title": title, "uri": uri})

            if not uri or uri in seen_urls:
                continue
            seen_urls.add(uri)

            # Newer SDKs expose web.domain; older ones put the site domain in
            # web.title. Either way the citation stays readable once the
            # redirect URI expires.
            domain = getattr(web, "domain", None) or title or None
            sources.append(
                Source(
                    type=SourceType.WEB,
                    title=title or domain or uri,
                    url=uri,
                    domain=domain,
                    retrieved_at=retrieved_at,
                )
            )

        if grounding_chunks:
            search_results["grounding_chunks"] = grounding_chunks
```

- [ ] **Step 4: Add the imports**

At the top of `app/agent_v2.py`, add to the existing `from app.models import ...` line (line 23) and add the datetime import:

```python
from datetime import datetime, timezone
```

```python
from app.models import CostSummary, PhaseCost, Source, SourceType
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_agent_v2_sources.py -v --no-cov`
Expected: PASS (8 tests)

- [ ] **Step 6: Verify the agent's existing tests still pass**

Run: `python -m pytest tests/test_agent_v2_grounding.py tests/test_agent_v2_router.py tests/test_chat_stream_endpoint.py -v --no-cov`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add fastapi_app/app/agent_v2.py fastapi_app/tests/unit/test_agent_v2_sources.py
git commit -m "feat(agent): emit typed, deduped web sources from grounding metadata"
```

---

### Task 3: Document registry and the resolver endpoint

**Files:**
- Create: `fastapi_app/app/document_registry.py`
- Modify: `fastapi_app/app/main.py` (build index in `lifespan` after metadata load ~line 68; add `get_document_index` dependency near line 216; add `GET /documents/{document_id}` endpoint)
- Test: `fastapi_app/tests/unit/test_document_registry.py`, `fastapi_app/tests/test_documents_endpoint.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `make_document_id(s3_path: str) -> str` — 16-char hex.
  - `build_document_index(metadata: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]` — maps `document_id` to `{"s3_path", "filename", "company", "year"}`.
  - `parse_s3_path(s3_path: str) -> Tuple[str, str]` — `(bucket, key)`; raises `ValueError` on a malformed path.
  - `presign_document(s3_client, s3_path: str, expires_in: int = 300) -> str`.
  - `app.state.document_index` and the `get_document_index()` FastAPI dependency.

S3 metadata is shaped `{company_name: [document, ...]}` where each document carries `document_link` (an `s3://bucket/key` string) and `filename`.

- [ ] **Step 1: Write the failing registry test**

Create `fastapi_app/tests/unit/test_document_registry.py`:

```python
"""Unit tests for the document registry.

`document_id` is deliberately a one-way hash rather than an encoded S3 path.
If it were reversible (e.g. base64), a client could edit the id and read
arbitrary objects out of the bucket through our own resolver endpoint.
Because resolution is a dict lookup built from metadata.json, no crafted
input can reach an object we did not index.
"""

import pytest

from app.document_registry import (
    build_document_index,
    make_document_id,
    parse_s3_path,
)

METADATA = {
    "NCB Financial Group": [
        {
            "filename": "ncb_annual_report_2023.pdf",
            "document_link": "s3://jse-renamed-docs-copy-2/organized/ncb/annual_2023.pdf",
            "year": "2023",
        },
        {
            "filename": "ncb_q4_2023.pdf",
            "document_link": "s3://jse-renamed-docs-copy-2/organized/ncb/q4_2023.pdf",
            "period": "2023",
        },
    ],
    "GraceKennedy": [
        {
            "filename": "gk_annual_report_2024.pdf",
            "document_link": "s3://jse-renamed-docs-copy-2/organized/gk/annual_2024.pdf",
            "year": "2024",
        }
    ],
}


@pytest.mark.unit
class TestMakeDocumentId:
    def test_is_stable(self):
        path = "s3://bucket/key.pdf"
        assert make_document_id(path) == make_document_id(path)

    def test_is_sixteen_hex_chars(self):
        doc_id = make_document_id("s3://bucket/key.pdf")
        assert len(doc_id) == 16
        assert all(c in "0123456789abcdef" for c in doc_id)

    def test_differs_per_path(self):
        assert make_document_id("s3://bucket/a.pdf") != make_document_id("s3://bucket/b.pdf")

    def test_leaks_no_path_material(self):
        """The id must not reveal bucket or key — it is a lookup key, not a codec."""
        doc_id = make_document_id("s3://secret-bucket/organized/private.pdf")
        assert "secret-bucket" not in doc_id
        assert "private" not in doc_id
        assert "organized" not in doc_id


@pytest.mark.unit
class TestBuildDocumentIndex:
    def test_indexes_every_document(self):
        index = build_document_index(METADATA)
        assert len(index) == 3

    def test_entry_carries_resolution_fields(self):
        index = build_document_index(METADATA)
        doc_id = make_document_id("s3://jse-renamed-docs-copy-2/organized/gk/annual_2024.pdf")
        entry = index[doc_id]
        assert entry["s3_path"] == "s3://jse-renamed-docs-copy-2/organized/gk/annual_2024.pdf"
        assert entry["filename"] == "gk_annual_report_2024.pdf"
        assert entry["company"] == "GraceKennedy"
        assert entry["year"] == "2024"

    def test_period_is_used_when_year_is_absent(self):
        index = build_document_index(METADATA)
        doc_id = make_document_id("s3://jse-renamed-docs-copy-2/organized/ncb/q4_2023.pdf")
        assert index[doc_id]["year"] == "2023"

    def test_documents_without_a_link_are_skipped(self):
        index = build_document_index({"ACME": [{"filename": "no_link.pdf"}]})
        assert index == {}

    def test_none_metadata_yields_empty_index(self):
        assert build_document_index(None) == {}

    def test_malformed_metadata_is_tolerated(self):
        """Metadata comes from S3 — a bad shape must not crash startup."""
        assert build_document_index({"ACME": "not-a-list"}) == {}
        assert build_document_index({"ACME": ["not-a-dict"]}) == {}


@pytest.mark.unit
class TestParseS3Path:
    def test_splits_bucket_and_key(self):
        assert parse_s3_path("s3://my-bucket/a/b/c.pdf") == ("my-bucket", "a/b/c.pdf")

    def test_rejects_non_s3_scheme(self):
        with pytest.raises(ValueError):
            parse_s3_path("https://my-bucket/a.pdf")

    def test_rejects_bucket_without_key(self):
        with pytest.raises(ValueError):
            parse_s3_path("s3://my-bucket")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_document_registry.py -v --no-cov`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.document_registry'`

- [ ] **Step 3: Create `app/document_registry.py`**

```python
"""Stable, non-forgeable identifiers for the S3 documents we cite.

A cited document is exposed to clients as an opaque `document_id`, never as
an S3 path. The id is a truncated SHA-256 of the path and is resolved by a
dict lookup built from metadata.json, so a client cannot craft an id that
reaches an object we did not index. Presigned URLs are minted per request
and never cached, so a cached response can never hand out a dead link.
"""

import hashlib
from typing import Any, Dict, Optional, Tuple

from app.logging_config import get_logger

logger = get_logger(__name__)

DOCUMENT_ID_LENGTH = 16
PRESIGNED_URL_TTL_SECONDS = 300


def make_document_id(s3_path: str) -> str:
    """Derive a stable, one-way id for an S3 path."""
    return hashlib.sha256(s3_path.encode("utf-8")).hexdigest()[:DOCUMENT_ID_LENGTH]


def parse_s3_path(s3_path: str) -> Tuple[str, str]:
    """Split an ``s3://bucket/key`` path into ``(bucket, key)``."""
    if not s3_path or not s3_path.startswith("s3://"):
        raise ValueError(f"Not an s3:// path: {s3_path!r}")
    remainder = s3_path[len("s3://") :]
    bucket, _, key = remainder.partition("/")
    if not bucket or not key:
        raise ValueError(f"S3 path is missing a bucket or key: {s3_path!r}")
    return bucket, key


def build_document_index(metadata: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Map document_id -> resolution info for every document in S3 metadata.

    Metadata is shaped ``{company_name: [document, ...]}``. It is loaded from
    S3 at startup, so a malformed shape is tolerated rather than fatal.
    """
    index: Dict[str, Dict[str, Any]] = {}
    if not metadata or not isinstance(metadata, dict):
        return index

    for company, documents in metadata.items():
        if not isinstance(documents, list):
            continue
        for doc in documents:
            if not isinstance(doc, dict):
                continue
            s3_path = doc.get("document_link")
            if not s3_path:
                continue
            index[make_document_id(s3_path)] = {
                "s3_path": s3_path,
                "filename": doc.get("filename") or s3_path.rsplit("/", 1)[-1],
                "company": company,
                "year": doc.get("year") or doc.get("period"),
            }

    logger.info(f"Document index built: {len(index)} documents")
    return index


def presign_document(
    s3_client: Any, s3_path: str, expires_in: int = PRESIGNED_URL_TTL_SECONDS
) -> str:
    """Mint a short-lived presigned GET URL for a document."""
    bucket, key = parse_s3_path(s3_path)
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=expires_in,
    )
```

- [ ] **Step 4: Run registry tests to verify they pass**

Run: `python -m pytest tests/unit/test_document_registry.py -v --no-cov`
Expected: PASS (13 tests)

- [ ] **Step 5: Write the failing endpoint test**

Create `fastapi_app/tests/test_documents_endpoint.py`:

```python
"""Tests for GET /documents/{document_id}.

The resolver is the only path from a citation to the underlying PDF. It must
resolve exactly the documents present in metadata.json and nothing else — an
unknown or tampered id gets a 404, never a presigned URL.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.document_registry import build_document_index, make_document_id
from app.main import app

S3_PATH = "s3://jse-renamed-docs-copy-2/organized/ncb/annual_2023.pdf"
METADATA = {
    "NCB Financial Group": [
        {
            "filename": "ncb_annual_report_2023.pdf",
            "document_link": S3_PATH,
            "year": "2023",
        }
    ]
}


@pytest.fixture
def client():
    app.state.metadata = METADATA
    app.state.document_index = build_document_index(METADATA)
    mock_s3 = MagicMock()
    mock_s3.generate_presigned_url.return_value = "https://s3.example/presigned?sig=abc"
    app.state.s3_client = mock_s3
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.integration
class TestDocumentResolver:
    def test_known_id_redirects_to_presigned_url(self, client):
        doc_id = make_document_id(S3_PATH)
        response = client.get(f"/documents/{doc_id}", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "https://s3.example/presigned?sig=abc"

    def test_presigned_url_is_short_lived(self, client):
        doc_id = make_document_id(S3_PATH)
        client.get(f"/documents/{doc_id}", follow_redirects=False)
        _, kwargs = app.state.s3_client.generate_presigned_url.call_args
        assert kwargs["ExpiresIn"] == 300
        assert kwargs["Params"]["Bucket"] == "jse-renamed-docs-copy-2"
        assert kwargs["Params"]["Key"] == "organized/ncb/annual_2023.pdf"

    def test_unknown_id_returns_404(self, client):
        response = client.get("/documents/0000000000000000", follow_redirects=False)
        assert response.status_code == 404
        app.state.s3_client.generate_presigned_url.assert_not_called()

    def test_tampered_id_returns_404(self, client):
        """Flipping one character must not resolve to anything."""
        doc_id = make_document_id(S3_PATH)
        tampered = ("0" if doc_id[0] != "0" else "1") + doc_id[1:]
        response = client.get(f"/documents/{tampered}", follow_redirects=False)
        assert response.status_code == 404
        app.state.s3_client.generate_presigned_url.assert_not_called()

    def test_path_traversal_attempt_does_not_resolve(self, client):
        response = client.get("/documents/..%2F..%2Fetc%2Fpasswd", follow_redirects=False)
        assert response.status_code == 404
        app.state.s3_client.generate_presigned_url.assert_not_called()
```

- [ ] **Step 6: Run endpoint test to verify it fails**

Run: `python -m pytest tests/test_documents_endpoint.py -v --no-cov`
Expected: FAIL with 404 on the known-id case (route does not exist yet)

- [ ] **Step 7: Wire the index into app startup**

In `app/main.py`, add the import beside the other app imports:

```python
from app.document_registry import build_document_index, presign_document
```

and `RedirectResponse` to the existing `fastapi.responses` import:

```python
from fastapi.responses import StreamingResponse, JSONResponse, Response, RedirectResponse
```

In `lifespan`, immediately after the metadata `try/except` block that sets `app.state.metadata` (around line 79), add:

```python
        # Index documents for citation resolution (GET /documents/{id}).
        app.state.document_index = build_document_index(app.state.metadata)
```

- [ ] **Step 8: Add the dependency and the endpoint**

Beside `get_metadata()` (line 216) add:

```python
def get_document_index():
    return getattr(app.state, "document_index", {}) or {}
```

Add the endpoint after the `/financial/metadata` endpoint:

```python
@app.get("/documents/{document_id}")
async def resolve_document(
    document_id: str,
    s3_client: Any = Depends(get_s3_client),
    document_index: Dict = Depends(get_document_index),
):
    """Resolve a cited document to a short-lived presigned S3 URL.

    `document_id` is a one-way hash, so this is a lookup against the documents
    present in metadata.json — an id we did not mint resolves to nothing.
    """
    entry = document_index.get(document_id)
    if not entry:
        logger.info(f"document_resolve_miss id={document_id[:32]}")
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        url = presign_document(s3_client, entry["s3_path"])
    except ValueError as e:
        logger.error(f"document_resolve_bad_path id={document_id}: {e}")
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"document_resolve_failed id={document_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not resolve document")

    return RedirectResponse(url=url, status_code=307)
```

- [ ] **Step 9: Run endpoint tests to verify they pass**

Run: `python -m pytest tests/test_documents_endpoint.py -v --no-cov`
Expected: PASS (5 tests)

- [ ] **Step 10: Verify no regression**

Run: `python -m pytest tests/unit tests/test_api.py -v --no-cov`
Expected: PASS

- [ ] **Step 11: Commit**

```bash
git add fastapi_app/app/document_registry.py fastapi_app/app/main.py fastapi_app/tests/unit/test_document_registry.py fastapi_app/tests/test_documents_endpoint.py
git commit -m "feat(documents): add document registry and GET /documents/{id} resolver"
```

---

### Task 4: Document sources on `/chat`

**Files:**
- Modify: `fastapi_app/app/document_selector.py` (add `DocumentLoadResult`; change returns of `auto_load_relevant_documents:479` and `auto_load_relevant_documents_async:564`)
- Modify: `fastapi_app/app/main.py:477-520` (`/chat` endpoint)
- Modify: `fastapi_app/app/_archive/streaming_chat.py:165-173`
- Modify: `fastapi_app/tests/unit/test_document_selector.py:49,58`; `fastapi_app/tests/test_async_s3_downloads.py:407,455`; `fastapi_app/test_async_integration.py:149`
- Test: `fastapi_app/tests/unit/test_document_sources.py`

**Interfaces:**
- Consumes: `Source`, `SourceType` (Task 1); `make_document_id` (Task 3).
- Produces: `DocumentLoadResult` dataclass with fields `texts: Dict[str, str]`, `message: str`, `loaded_docs: List[str]`, `sources: List[Source]`. Both loader functions return it instead of a 3-tuple.

The loaders already have `document_link` and `reason` in scope inside the load loop and discard them. Seven call sites unpack or assert the tuple; all are updated in this task.

- [ ] **Step 1: Write the failing test**

Create `fastapi_app/tests/unit/test_document_sources.py`:

```python
"""Unit tests for document provenance returned by the /chat loader.

The loader has always had document_link and the selection reason in scope
and thrown them away, returning bare filenames. Users fact-checking an
answer need to reach the actual PDF, so the loader now returns Source
objects carrying an opaque document_id the resolver endpoint can redeem.
"""

from unittest.mock import Mock, patch

import pytest

from app.document_registry import make_document_id
from app.document_selector import DocumentLoadResult, auto_load_relevant_documents
from app.models import SourceType

S3_PATH = "s3://jse-renamed-docs-copy-2/organized/ncb/annual_2023.pdf"

RECOMMENDATION = {
    "companies_mentioned": ["NCB Financial Group"],
    "documents_to_load": [
        {
            "company": "NCB Financial Group",
            "document_link": S3_PATH,
            "filename": "ncb_annual_report_2023.pdf",
            "reason": "Contains FY2023 net profit",
        }
    ],
}


@pytest.fixture
def loaded():
    with patch("app.document_selector.semantic_document_selection") as mock_select, patch(
        "app.document_selector.download_and_extract_from_s3"
    ) as mock_download:
        mock_select.return_value = RECOMMENDATION
        mock_download.return_value = "extracted pdf text"
        yield auto_load_relevant_documents(
            s3_client=Mock(),
            query="NCB net profit 2023",
            metadata={"NCB Financial Group": []},
            current_document_texts={},
        )


@pytest.mark.unit
class TestDocumentSources:
    def test_returns_a_document_load_result(self, loaded):
        assert isinstance(loaded, DocumentLoadResult)

    def test_loaded_docs_still_lists_filenames(self, loaded):
        """Backward compatibility: ChatResponse.documents_loaded is unchanged."""
        assert loaded.loaded_docs == ["ncb_annual_report_2023.pdf"]

    def test_texts_are_keyed_by_filename(self, loaded):
        assert loaded.texts["ncb_annual_report_2023.pdf"] == "extracted pdf text"

    def test_source_is_typed_document(self, loaded):
        assert loaded.sources[0].type == SourceType.DOCUMENT

    def test_source_carries_resolvable_document_id(self, loaded):
        assert loaded.sources[0].document_id == make_document_id(S3_PATH)

    def test_source_never_exposes_the_s3_path(self, loaded):
        """The bucket layout must not reach the browser."""
        dumped = loaded.sources[0].model_dump()
        assert S3_PATH not in str(dumped)
        assert "jse-renamed-docs-copy-2" not in str(dumped)

    def test_source_carries_selection_reason_and_company(self, loaded):
        source = loaded.sources[0]
        assert source.title == "ncb_annual_report_2023.pdf"
        assert source.detail == "Contains FY2023 net profit"
        assert source.company == "NCB Financial Group"
        assert source.retrieved_at is not None

    def test_failed_download_produces_no_source(self):
        """A document we could not read must not be cited as evidence."""
        with patch("app.document_selector.semantic_document_selection") as mock_select, patch(
            "app.document_selector.download_and_extract_from_s3"
        ) as mock_download:
            mock_select.return_value = RECOMMENDATION
            mock_download.return_value = None
            result = auto_load_relevant_documents(
                s3_client=Mock(),
                query="NCB net profit 2023",
                metadata={"NCB Financial Group": []},
                current_document_texts={},
            )
        assert result.loaded_docs == []
        assert result.sources == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_document_sources.py -v --no-cov`
Expected: FAIL with `ImportError: cannot import name 'DocumentLoadResult'`

- [ ] **Step 3: Add the dataclass and imports to `document_selector.py`**

At the top of `app/document_selector.py`, add:

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app.document_registry import make_document_id
from app.models import Source, SourceType
```

Add the dataclass after the module's imports:

```python
@dataclass
class DocumentLoadResult:
    """What a document-loading pass produced.

    Replaces the old 3-tuple: callers needed a fourth value (`sources`), and
    growing the tuple would have broken every unpacking site anyway.
    """

    texts: Dict[str, str]
    message: str
    loaded_docs: List[str] = field(default_factory=list)
    sources: List[Source] = field(default_factory=list)
```

- [ ] **Step 4: Build sources in `auto_load_relevant_documents`**

Replace the body of `auto_load_relevant_documents` (lines 498–556) with:

```python
    document_texts = current_document_texts.copy() if current_document_texts else {}
    loaded_docs = []
    sources: List[Source] = []

    # Use two-stage LLM approach to determine which documents to load
    recommendation = semantic_document_selection(
        query, metadata, conversation_history, associations
    )

    if not recommendation or "documents_to_load" not in recommendation:
        return DocumentLoadResult(
            texts=document_texts,
            message="No relevant documents were identified for your query.",
        )

    docs_to_load = recommendation["documents_to_load"]

    # Load the recommended documents (limited to 3)
    for doc_info in docs_to_load[:3]:
        doc_link = doc_info["document_link"]
        doc_name = doc_info["filename"]

        # Only load if not already loaded
        if doc_name in document_texts:
            continue

        load_start = time.time()
        try:
            text = download_and_extract_from_s3(s3_client, doc_link)
            load_duration = time.time() - load_start
            if text:
                document_texts[doc_name] = text
                loaded_docs.append(doc_name)
                # Only cite a document we actually read.
                sources.append(
                    Source(
                        type=SourceType.DOCUMENT,
                        title=doc_name,
                        detail=doc_info.get("reason"),
                        document_id=make_document_id(doc_link),
                        company=doc_info.get("company"),
                        year=doc_info.get("year") or doc_info.get("period"),
                        retrieved_at=datetime.now(timezone.utc).isoformat(),
                    )
                )
                record_document_load(source="s3", duration=load_duration, success=True)
            else:
                record_document_load(source="error", duration=load_duration, success=False)
        except Exception as e:
            load_duration = time.time() - load_start
            record_document_load(source="error", duration=load_duration, success=False)
            logger.error(
                "document_load_failed",
                extra={
                    "document": doc_name,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )

    if not loaded_docs:
        return DocumentLoadResult(
            texts=document_texts,
            message=(
                "No documents were loaded. Please check S3 access permissions "
                "or document availability."
            ),
        )

    message = f"Semantically selected {len(loaded_docs)} documents based on your query:\n"
    for doc_name in loaded_docs:
        matching_doc = next((d for d in docs_to_load if d["filename"] == doc_name), None)
        if matching_doc:
            message += f"• {doc_name} - {matching_doc.get('reason', '')}\n"

    return DocumentLoadResult(
        texts=document_texts,
        message=message,
        loaded_docs=loaded_docs,
        sources=sources,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_document_sources.py -v --no-cov`
Expected: PASS (8 tests)

- [ ] **Step 6: Apply the same change to the async loader**

In `auto_load_relevant_documents_async` (line 564), build a `sources` list the same way — append a `Source` for each document whose download succeeded (the loop that appends to `loaded_docs` around line 670) — and return `DocumentLoadResult(texts=..., message=..., loaded_docs=..., sources=...)` at each of its return points (lines ~697–712 and the early-return path).

The async loader is used only by archived code and tests today, but letting the pair diverge is how it rots.

- [ ] **Step 7: Update the `/chat` endpoint**

In `app/main.py`, replace lines 477–484:

```python
            load_result = auto_load_relevant_documents(
                s3_client,
                request.query,
                metadata,
                {},  # Start with empty document_texts since this is stateless
                request.conversation_history,
                associations,
            )
            document_texts = load_result.texts
            document_selection_message = load_result.message
            loaded_docs = load_result.loaded_docs
            doc_sources = load_result.sources
```

Initialise `doc_sources = []` beside `loaded_docs = []` in the variable setup (line 468), and add the field to the returned `ChatResponse` (line 515):

```python
        return ChatResponse(
            response=response_text,
            documents_loaded=loaded_docs if loaded_docs else None,
            document_selection_message=document_selection_message,
            conversation_history=updated_conversation_history,
            sources=doc_sources if doc_sources else None,
        )
```

- [ ] **Step 8: Update the remaining call sites**

- `fastapi_app/app/_archive/streaming_chat.py:165-173` — replace tuple unpacking with `load_result = await auto_load_relevant_documents_async(...)` then read `.texts`, `.message`, `.loaded_docs`.
- `fastapi_app/tests/test_async_s3_downloads.py:407,455` — replace `document_texts, message, loaded_docs = await ...` with `result = await ...` and read the attributes.
- `fastapi_app/test_async_integration.py:149` — same change.
- `fastapi_app/tests/unit/test_document_selector.py:53,64` — these assert `isinstance(result, tuple)` and `len(result) == 3`. Replace both with:

```python
            assert isinstance(result, DocumentLoadResult)
            assert isinstance(result.texts, dict)
            assert isinstance(result.loaded_docs, list)
            assert isinstance(result.sources, list)
```

and add `DocumentLoadResult` to that file's imports from `app.document_selector`.

- [ ] **Step 9: Run the full affected suite**

Run: `python -m pytest tests/unit tests/test_async_s3_downloads.py tests/test_api.py -v --no-cov`
Expected: PASS

- [ ] **Step 10: Commit**

```bash
git add fastapi_app/app/document_selector.py fastapi_app/app/main.py fastapi_app/app/_archive/streaming_chat.py fastapi_app/tests fastapi_app/test_async_integration.py
git commit -m "feat(chat): return document sources with resolvable ids from /chat"
```

---

### Task 5: Dataset sources on `/fast_chat_v2`

**Files:**
- Modify: `fastapi_app/app/financial_utils.py` (add `FinancialDataManager.describe_source`)
- Modify: `fastapi_app/app/main.py:707-744` (`/fast_chat_v2` cache write and response) and `main.py:631-641` (cache-hit response)
- Test: `fastapi_app/tests/unit/test_dataset_source.py`

**Interfaces:**
- Consumes: `Source`, `SourceType` (Task 1).
- Produces: `FinancialDataManager.describe_source(filters: FinancialDataFilters, record_count: int) -> Source`.

Per the spec, `filters` carries only the four query-shaping fields — `companies`, `symbols`, `years`, `standard_items`. The conversational fields describe how the query was understood, not which data was read, and `filters_used` already carries the full object.

- [ ] **Step 1: Write the failing test**

Create `fastapi_app/tests/unit/test_dataset_source.py`:

```python
"""Unit tests for dataset provenance on /fast_chat_v2.

A financial figure's honest provenance is the table it came from plus the
filters applied — not a fabricated per-number URL. See the known limitation
in the design doc: the table has no column pointing back to the statement
PDF, so a dataset source deliberately stops at the query.
"""

from unittest.mock import patch

import pytest

from app.financial_utils import FinancialDataManager
from app.models import FinancialDataFilters, SourceType


@pytest.fixture
def manager():
    with patch.object(FinancialDataManager, "__init__", lambda self: None):
        mgr = FinancialDataManager()
    mgr.project_id = "jse-proj"
    mgr.dataset = "financials"
    mgr.table = "statements"
    return mgr


@pytest.mark.unit
class TestDescribeSource:
    def test_source_is_typed_dataset(self, manager):
        source = manager.describe_source(FinancialDataFilters(), 0)
        assert source.type == SourceType.DATASET

    def test_table_is_fully_qualified(self, manager):
        source = manager.describe_source(FinancialDataFilters(), 3)
        assert source.table == "jse-proj.financials.statements"

    def test_record_count_is_carried(self, manager):
        source = manager.describe_source(FinancialDataFilters(), 42)
        assert source.record_count == 42

    def test_only_query_shaping_filters_are_included(self, manager):
        """Conversational fields describe interpretation, not what was read."""
        filters = FinancialDataFilters(
            companies=["NCB Financial Group"],
            symbols=["NCBFG"],
            years=["2023"],
            standard_items=["net_profit"],
            interpretation="user wants NCB net profit",
            is_follow_up=True,
            context_used="previous turn",
            data_availability_note="note",
            unrecognized_items=["ebitda"],
        )
        source = manager.describe_source(filters, 4)
        assert source.filters == {
            "companies": ["NCB Financial Group"],
            "symbols": ["NCBFG"],
            "years": ["2023"],
            "standard_items": ["net_profit"],
        }

    def test_title_names_the_dataset_readably(self, manager):
        source = manager.describe_source(FinancialDataFilters(), 1)
        assert source.title
        assert "jse-proj" not in source.title  # a title, not a table id

    def test_retrieved_at_is_populated(self, manager):
        source = manager.describe_source(FinancialDataFilters(), 1)
        assert source.retrieved_at is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_dataset_source.py -v --no-cov`
Expected: FAIL with `AttributeError: 'FinancialDataManager' object has no attribute 'describe_source'`

- [ ] **Step 3: Add `describe_source` to `FinancialDataManager`**

In `app/financial_utils.py`, add the import and the method (place it next to `query_data`, around line 891):

```python
from datetime import datetime, timezone

from app.models import Source, SourceType
```

```python
    def describe_source(self, filters: FinancialDataFilters, record_count: int) -> Source:
        """Describe the dataset read for this answer.

        Honest provenance for a financial figure is the table plus the filters
        applied. The table carries no reference to the statement PDF a figure
        was extracted from, so this deliberately stops at the query rather than
        implying a document-level citation it cannot support.
        """
        return Source(
            type=SourceType.DATASET,
            title="JSE financial statements dataset",
            detail="Figures read directly from the JSE financial statements table",
            table=f"{self.project_id}.{self.dataset}.{self.table}",
            filters={
                "companies": list(filters.companies or []),
                "symbols": list(filters.symbols or []),
                "years": list(filters.years or []),
                "standard_items": list(filters.standard_items or []),
            },
            record_count=record_count,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_dataset_source.py -v --no-cov`
Expected: PASS (6 tests)

- [ ] **Step 5: Wire sources into `/fast_chat_v2`**

In `app/main.py`, after the chart generation block (line ~698) add:

```python
            sources = (
                [financial_manager.describe_source(filters, len(results))] if results else None
            )
```

Add to the cache payload written at line 707:

```python
                    "sources": _jsonable(sources),
```

Add to the returned `FinancialDataResponse` at line 734:

```python
            sources=sources,
```

And in the cache-hit branch (line 631), add:

```python
                sources=cached.get("sources"),
```

- [ ] **Step 6: Write the cache round-trip test**

Append to `fastapi_app/tests/unit/test_dataset_source.py`:

```python
@pytest.mark.unit
class TestSourceCacheRoundTrip:
    def test_source_survives_json_round_trip(self, manager):
        """Sources go through Redis as JSON; nothing in them may be unserializable."""
        import json

        from app.main import _jsonable
        from app.models import Source

        source = manager.describe_source(
            FinancialDataFilters(companies=["NCB"], years=["2023"]), 5
        )
        payload = json.loads(json.dumps(_jsonable([source])))
        restored = [Source(**item) for item in payload]
        assert restored[0].table == source.table
        assert restored[0].filters == source.filters
        assert restored[0].record_count == 5
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_dataset_source.py -v --no-cov`
Expected: PASS (7 tests)

- [ ] **Step 8: Verify no regression**

Run: `python -m pytest tests/unit tests/test_financial_utils.py tests/test_api.py -v --no-cov`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add fastapi_app/app/financial_utils.py fastapi_app/app/main.py fastapi_app/tests/unit/test_dataset_source.py
git commit -m "feat(financial): return dataset provenance from /fast_chat_v2"
```

---

### Task 6: Fix the test client's source rendering

**Files:**
- Modify: `mock_client/chat_assistant_test_client.html:943-945`

**Interfaces:**
- Consumes: the `Source` JSON shape from Tasks 1, 2, 4, 5.
- Produces: nothing consumed by later tasks.

The client reads `s.description` — a key the API has never emitted — so its source display has always been blank. There is no automated test for the mock client; verification is by eye in Task 7.

- [ ] **Step 1: Replace the renderer**

Replace lines 943–945:

```javascript
      if (data.sources?.length) {
        const sourcesList = data.sources.map(s => {
          const label = s.title || s.domain || s.table || 'source';
          if (s.type === 'web' && s.url) return `<a href="${s.url}" target="_blank" rel="noopener noreferrer">${label}</a>`;
          if (s.type === 'document' && s.document_id) return `<a href="${baseUrl}/documents/${s.document_id}" target="_blank" rel="noopener noreferrer">${label}</a>`;
          if (s.type === 'dataset') return `${label}${s.record_count != null ? ` (${s.record_count} rows)` : ''}`;
          return label;
        }).join(', ');
        extraDetails.push(`Sources: ${sourcesList}`);
      }
```

- [ ] **Step 2: Commit**

```bash
git add mock_client/chat_assistant_test_client.html
git commit -m "fix(mock-client): render the real Source schema instead of a nonexistent key"
```

---

### Task 7: Full local verification — no-regression gate

**Files:** none modified. This task produces evidence, not code.

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces: a verified, regression-free branch.

**Do not report this work as complete until every step below has been run and its actual output inspected.** A step that fails is a bug to fix, not a note to append.

- [ ] **Step 1: Re-read the baseline captured in Task 0**

```bash
tail -20 fastapi_app/baseline-tests.txt
```

This is the pre-change pass/fail state. If the file is missing, Task 0 was skipped — go back and run it against `origin/main` in a scratch worktree, because the current branch can no longer produce an honest baseline.

- [ ] **Step 2: Run the full test suite**

Run: `python -m pytest tests -v --no-cov`
Expected: PASS. Compare the failure set against Task 0's baseline — it must be identical or smaller. **Any test that passed in the baseline and fails now blocks this task**, regardless of whether it looks related to this work.

- [ ] **Step 3: Run linting and formatting**

```bash
python -m ruff check app tests && python -m black --check app tests
```

Expected: clean. Fix anything reported, then re-run.

- [ ] **Step 4: Confirm the OpenAPI schema builds**

A malformed response model breaks schema generation without failing any unit test:

```bash
python -c "from app.main import app; s = app.openapi(); assert 'Source' in s['components']['schemas']; print('Source schema OK'); print(sorted(s['paths']))"
```

Expected: `Source schema OK` and `/documents/{document_id}` present in the path list.

- [ ] **Step 5: Smoke-test the running app**

REQUIRED SUB-SKILL: use the `run-local-smoke-test` skill to launch the app and drive it. It covers the `.env`-in-worktree gotcha and the Windows shutdown steps.

Verify by inspecting actual response bodies, not just status codes:
- `GET /health` returns 200.
- `POST /chat/stream` with `{"query": "What was NCB's net profit in 2023?"}` returns a non-empty `sources` array whose entries have `type: "web"`, a `title`, and a `url`.
- `POST /fast_chat_v2` with the same query returns a `sources` array containing one `dataset` entry with a fully-qualified `table` and a `record_count` matching `record_count` at the top level.
- `POST /chat` with a document-shaped query returns `sources` entries of `type: "document"` each carrying a `document_id`, and `documents_loaded` still populated.
- `GET /documents/{document_id}` using an id from that response returns a 307 whose `Location` is an S3 presigned URL.
- `GET /documents/0000000000000000` returns 404.

- [ ] **Step 6: Verify the cache path specifically**

Cached responses are the most likely place for this feature to break, because sources must survive a JSON round-trip through Redis. With the app running, issue the **same** `/fast_chat_v2` query twice with no conversation history. Confirm the second response is a cache hit (check the logs for `cache hit`) and that its `sources` array is identical to the first. Repeat for `/chat/stream`.

- [ ] **Step 7: Verify the test client by eye**

Open `mock_client/chat_assistant_test_client.html` against the local server and confirm the Sources line renders real, clickable entries in all three modes — not blank, and not `[object Object]`.

- [ ] **Step 8: Report results honestly**

State the actual test counts, any pre-existing failures carried over from the Step 1 baseline, and anything not verified. If a step could not be run (no AWS credentials, no Redis), say so explicitly rather than implying it passed.

- [ ] **Step 9: Commit any fixes**

Stage explicitly — `git add -A` would sweep in `baseline-tests.txt`:

```bash
git add fastapi_app/app fastapi_app/tests mock_client
git commit -m "test: verify sources feature end-to-end with no regressions"
```

- [ ] **Step 10: Clean up the scratch baseline**

```bash
rm -f fastapi_app/baseline-tests.txt && git status --short
```

Expected: a clean tree.

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| `Source` model + `SourceType` | 1 |
| Response model changes (3 models) | 1 |
| `retrieved_at` semantics | 2, 4, 5 (set at retrieval time in each producer) |
| Backward compatibility | 1 (test), 4 (`documents_loaded` retained) |
| `/chat/stream` wiring + dedupe + `domain` | 2 |
| `/fast_chat_v2` wiring + cache payload + filter subset | 5 |
| `/chat` wiring + `DocumentLoadResult` + 7 call sites | 4 |
| Document resolver + non-forgeable id + 5-min TTL | 3 |
| Test client fix | 6 |
| Testing section (builders, dedupe, resolver 404, cache round-trip, back-compat) | 1, 2, 3, 4, 5 |
| Known limitation (no statement-page citation) | 5 (documented in `describe_source` docstring) |

**Not in the spec, added here:**
- The `_jsonable` fix at `main.py:1013` (Task 1, Step 5). Without it, changing `sources` to a typed model breaks the `/chat/stream` Redis write — a regression the spec did not anticipate.
- Task 0 (baseline capture) and Task 7 (verification gate), per the requirement that no regression be introduced locally before this is called done.

**Known-broken code deliberately left alone:** `POST /cache/refresh` calls `load_metadata_from_s3()` with no arguments while the function requires `s3_client` (`main.py:806`), so it raises `TypeError` on every call. Pre-existing and unrelated; tracked separately. It is the reason the document index is built in `lifespan` (Task 3, Step 7) rather than hung off that endpoint — but note that until it is fixed, a metadata refresh will not rebuild the document index.

**Type consistency:** `Source` / `SourceType` field names are identical across Tasks 1–5. `DocumentLoadResult` uses `.texts` / `.message` / `.loaded_docs` / `.sources` in Task 4's definition and at every call site. `make_document_id` has one signature, used in Tasks 3 and 4. `describe_source(filters, record_count)` matches its call in Task 5, Step 5.
