"""`data_found` must mean data was found, not that the answer was long.

On /chat/stream it was `bool(response_text and len(response_text) > 50)` --
true for any answer over one sentence, including one the model produced from
its priors with no search at all. /fast_chat_v2 sets it honestly from
`bool(results)` (app/main.py), so the same field meant two different things
depending on the endpoint.

That is not only cosmetic. The field is returned on AgentChatResponse and
written to the BigQuery interactions table, and the eval judge reads it as
evidence that a data tool returned something -- scoring answers with zero
sources as well grounded on the strength of a length check.

It now reflects retrieval: true when the answer is backed by sources the
grounding step actually returned.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent_v2 import AgentV2


def _plain(text):
    r = MagicMock(spec=["text", "candidates", "usage_metadata"])
    r.text = text
    r.candidates = []
    r.usage_metadata = None
    return r


def _grounded(text, n_sources):
    """A synthesis response carrying `n_sources` distinct grounding chunks."""
    chunks = []
    for i in range(n_sources):
        web = MagicMock(spec=["uri", "title", "domain"])
        web.uri, web.title, web.domain = f"https://x.test/{i}", f"Source {i}", "x.test"
        c = MagicMock(spec=["web"])
        c.web = web
        chunks.append(c)
    grounding = MagicMock(spec=["grounding_chunks", "web_search_queries", "search_entry_point"])
    grounding.grounding_chunks = chunks
    grounding.web_search_queries = ["q"]
    grounding.search_entry_point = None
    cand = MagicMock(spec=["grounding_metadata", "finish_reason"])
    cand.grounding_metadata = grounding
    cand.finish_reason = None
    r = MagicMock(spec=["text", "candidates", "usage_metadata"])
    r.text = text
    r.candidates = [cand]
    r.usage_metadata = None
    return r


@pytest.fixture
def client():
    with patch("app.agent_v2.get_genai_client") as get_client:
        c = MagicMock()
        c.aio.models.generate_content = AsyncMock()
        get_client.return_value = c
        yield c


async def _run(client, synthesis):
    client.aio.models.generate_content.side_effect = [_plain("ALLOW"), synthesis]
    with patch("app.agent_v2.extract_companies_from_query_async", new=AsyncMock()) as ex:
        ex.return_value = {"companies": [], "symbols": []}
        return await AgentV2().run(query="what was NCB revenue in 2023")


LONG = "NCB Financial Group reported revenue growth across its banking segment in 2023." * 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_data_found_true_when_sources_back_the_answer(client):
    result = await _run(client, _grounded(LONG, n_sources=3))
    assert result["sources"], "fixture should produce sources"
    assert result["data_found"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_data_found_false_for_long_answer_with_no_sources(client):
    """The defect, stated directly. This answer is 200+ characters and came
    from the model's priors -- the old length check called it data."""
    result = await _run(client, _grounded(LONG, n_sources=0))
    assert len(result["response"]) > 50
    assert result["sources"] is None
    assert result["data_found"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_data_found_not_driven_by_response_length(client):
    """A short grounded answer beats a long ungrounded one."""
    short = await _run(client, _grounded("Yes.", n_sources=2))
    long_ungrounded = await _run(client, _grounded(LONG, n_sources=0))
    assert len(short["response"]) < 50
    assert short["data_found"] is True
    assert long_ungrounded["data_found"] is False
