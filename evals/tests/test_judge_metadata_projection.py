"""The judge must be able to see `sources` before it scores groundedness.

Raw API metadata was dumped into the judge prompt and cut at 4000 chars.
Measured across 617 recorded turns, `data_preview` averaged 40802 chars and
`conversation_history` 4313, while `sources` -- the actual evidence --
averaged 1283. The result: on 90% of turns that HAD sources, the judge never
reached them, and it fell back on `data_found` to guess. That produced
groundedness 5 on answers with zero sources, and groundedness 1 on answers
that were properly sourced but truncated out of view.

`_project_metadata` reduces each turn to the grounding-relevant fields, puts
`sources` first, summarises `data_preview`, and trims the opaque redirect
URLs. Replayed over the same 617 turns: `sources` unreachable on 0%, average
metadata shown down from 128376 chars to 3624.
"""

import json

import pytest

from evals.judge import (
    _DATA_PREVIEW_ROWS,
    _METADATA_CHAR_LIMIT,
    _SOURCE_URL_CHARS,
    _project_metadata,
)

REDIRECT = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/" + "A" * 200


def _metadata(**over):
    md = {
        "sources": [
            {
                "type": "web",
                "title": "JSE Filing",
                "url": REDIRECT,
                "domain": "jamstockex.com",
            }
        ],
        "tools_executed": ["google_search"],
        "record_count": 0,
        "data_found": True,
        "response": "a duplicate of BOT TEXT " * 50,
        "conversation_history": [{"role": "user", "content": "x" * 4000}],
        "data_preview": [{"Company": f"C{i}", "Year": 2023, "Value": i} for i in range(500)],
        "web_search_results": {
            "queries": ["ncb revenue 2023"],
            "grounding_chunks": [{"web": {"uri": REDIRECT}} for _ in range(40)],
        },
        "cost_summary": {"total_cost_usd": 0.01, "phases": [{"x": 1}] * 20},
        "chart": {"data": list(range(500))},
    }
    md.update(over)
    return md


def test_sources_are_kept_and_come_first():
    """Ordering is the whole point: if the block is still cut, the cut must
    fall after the evidence, not before it."""
    out = _project_metadata(_metadata())
    assert "sources" in out
    assert next(iter(out)) == "sources"


@pytest.mark.parametrize("field", ["conversation_history", "response", "cost_summary", "chart"])
def test_non_grounding_fields_are_dropped(field):
    assert field not in _project_metadata(_metadata())


def test_data_preview_is_summarised_not_dumped():
    out = _project_metadata(_metadata())
    assert out["data_preview_row_count"] == 500
    assert len(out["data_preview_sample"]) == _DATA_PREVIEW_ROWS
    assert "data_preview" not in out


def test_search_queries_kept_but_chunk_bulk_dropped():
    out = _project_metadata(_metadata())
    assert out["web_search_queries"] == ["ncb revenue 2023"]
    assert "web_search_results" not in out


def test_redirect_urls_are_trimmed_but_title_and_domain_survive():
    """Grounding URIs are opaque and expire in ~30 days; title and domain are
    what let the judge check a claim against its source."""
    src = _project_metadata(_metadata())["sources"][0]
    assert len(src["url"]) < len(REDIRECT)
    assert src["url"].startswith(REDIRECT[:_SOURCE_URL_CHARS])
    assert src["title"] == "JSE Filing"
    assert src["domain"] == "jamstockex.com"


def test_short_urls_are_left_alone():
    md = _metadata(sources=[{"type": "web", "title": "T", "url": "https://a.test/x"}])
    assert _project_metadata(md)["sources"][0]["url"] == "https://a.test/x"


def test_projection_fits_the_budget_for_a_realistic_turn():
    """The regression that matters: this metadata is 100k+ chars raw."""
    md = _metadata()
    assert len(json.dumps(md, indent=2)) > _METADATA_CHAR_LIMIT
    assert len(json.dumps(_project_metadata(md), indent=2)) <= _METADATA_CHAR_LIMIT


def test_data_found_is_still_shown():
    """Kept deliberately. It is legitimate evidence for tool_use and for a bot
    that honestly reports no data; the rubric is what forbids reading it as
    proof of grounding."""
    assert _project_metadata(_metadata())["data_found"] is True


def test_empty_and_missing_metadata_are_safe():
    assert _project_metadata({}) == {}
    assert _project_metadata(None) == {}


def test_non_list_data_preview_is_passed_through():
    assert _project_metadata(_metadata(data_preview={"odd": 1}))["data_preview"] == {"odd": 1}
