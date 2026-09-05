"""The router classifies request FORM, never subject matter.

Background. The router is a plain-text classifier with no company list, no
metadata and no search. It used to also refuse "non-JSE markets" -- the one
refusal category that requires knowing what an entity IS. Measured against the
live router on 2026-09-05 at temperature 0.0, eight calls per phrasing:

    REFUSE 8/8   "What is the latest news on quantas advantage"
    ALLOW  7/8   "What is the latest news on Quantas Advantage"
    ALLOW  8/8   "Tell me about CAR" / "What does KEY do"

Same company (Quantas Advantage Inc, JSE: QAINC, listed 2026), flipped by
capitalisation alone, because the model matched the lowercase string to Qantas
the airline. Prod refused a real JSE listing as a foreign airline.

The fix removes subject-matter judgement from the router. Market scope now lives
only in SYSTEM_PROMPT section 2, applied by the grounded synthesis call that can
actually resolve the entity.

These tests pin the policy in the prompt text. They cannot prove the model obeys
it -- that needs live calls, which the eval suite's disambiguation and negative
personas cover. What they do prevent is the market-scope category being quietly
reintroduced into the router, which would restore the defect with no test going
red.
"""

import pytest

from app.agent_v2 import QUERY_ROUTER_PROMPT, SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# What the router must still refuse -- none of these need entity knowledge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "category",
    [
        "personalised investment advice",
        "predicting a future price",
        "directional forecast",
        "poems",
        "persona",
    ],
)
def test_router_still_refuses_form_based_categories(category):
    assert category in QUERY_ROUTER_PROMPT, f"router no longer refuses {category!r}"


# ---------------------------------------------------------------------------
# What the router must NOT decide
# ---------------------------------------------------------------------------


def test_router_does_not_refuse_on_market_scope():
    """The regression that caused the defect. If a future edit reintroduces a
    market/geography category into the REFUSE list, this fails."""
    refuse_block = QUERY_ROUTER_PROMPT.split("REFUSE —")[1].split("ALLOW —")[0]
    for banned in ("non-JSE", "US equities", "crypto", "forex", "foreign exchange"):
        assert banned not in refuse_block, (
            f"{banned!r} is back in the router's REFUSE list -- the router cannot "
            "resolve entities, so it must not judge market scope"
        )


def test_router_told_an_unrecognised_name_is_not_evidence():
    lowered = QUERY_ROUTER_PROMPT.lower()
    assert "never heard of is allow" in lowered
    assert "sounds foreign is allow" in lowered


def test_router_biases_to_allow():
    assert "When in doubt, choose ALLOW" in QUERY_ROUTER_PROMPT


def test_router_still_emits_exactly_two_labels():
    assert QUERY_ROUTER_PROMPT.rstrip().endswith("Output only: REFUSE or ALLOW")


# ---------------------------------------------------------------------------
# Market scope must survive downstream -- the whole fix depends on it
# ---------------------------------------------------------------------------


def test_market_scope_still_enforced_in_system_prompt():
    """Removing market scope from the router is only safe because the grounded
    synthesis call still enforces it. If this rule ever leaves SYSTEM_PROMPT the
    assistant loses its scope boundary entirely, not just its cheap refusal."""
    assert "Out-of-scope markets" in SYSTEM_PROMPT
    for market in ("US equities", "crypto", "forex", "S&P 500", "NASDAQ"):
        assert market in SYSTEM_PROMPT, f"{market!r} no longer named in SYSTEM_PROMPT scope rules"


def test_system_prompt_still_forbids_analysis_of_out_of_scope_markets():
    assert "Do not provide any analysis, data, or commentary for these markets" in SYSTEM_PROMPT
