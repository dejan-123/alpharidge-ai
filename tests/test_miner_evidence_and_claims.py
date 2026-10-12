"""Miner-side self-verification: evidence spans the floor accepts, and claims
grounded before submission.

A bare surface form ("NVDA") as an evidence span fails floor.span_supported —
it lacks the >=12 chars or >=3 context words — which lands the asset in
unevidenced_assets, drops it from keeper judgment fields, and lowers
floor_quality. An ungroundable numeric claim can never be in the grader's gold
set (the reference's claims pass the same parser), so it can only cost
precision and calibration.
"""

import types

from alpharidge_ai.analyzer.article_intelligence_analyzer import (
    ArticleIntelligenceAnalyzer)
from alpharidge_ai.oracle import floor, runner

BODY = ("Nvidia shares jumped 12.5% after the chipmaker reported quarterly "
        "revenue of $44.1 billion, beating estimates. Analysts at Morgan "
        "Stanley raised their price target for NVDA to $210 following the "
        "print.")


def _analyzer():
    # _build_assets / _build_numeric_claims / _evidence_spans touch no instance
    # state beyond class attributes — skip the heavy model loads.
    return ArticleIntelligenceAnalyzer.__new__(ArticleIntelligenceAnalyzer)


def _entity(text="nvda", ticker="NVDA", name="Nvidia", forms=None):
    return types.SimpleNamespace(
        text=text, ticker=ticker, canonical_name=name,
        surface_forms=[text] if forms is None else forms,
        confidence=0.9, source="keyword", asset_class="stock",
        is_primary_subject=True)


def _ner(*entities):
    return types.SimpleNamespace(resolved_assets=list(entities))


def _claim(value, unit, metric="margin", confidence=0.5):
    return {"metric_name": metric, "value": value, "unit": unit,
            "confidence": confidence}


def _sub(assets):
    return types.SimpleNamespace(numeric_claims=[], quotes=[],
                                 assets=list(assets), entities=[])


# ---- evidence spans ---------------------------------------------------------------

def test_asset_evidence_is_a_verbatim_context_span():
    assets = _analyzer()._build_assets(_ner(_entity()), [], BODY)
    assert len(assets) == 1
    spans = assets[0].evidence_spans
    assert spans, "expected at least one evidence span"
    art = floor.normalize(BODY)
    for s in spans:
        assert floor.span_supported(art, s, "NVDA"), s


def test_sentiment_mention_sentence_is_used_when_verbatim():
    sentence = BODY[:BODY.index(",")]
    assets = _analyzer()._build_assets(
        _ner(_entity()),
        [{"ticker": "NVDA", "evidence_spans": [sentence]}], BODY)
    spans = assets[0].evidence_spans
    assert sentence in spans
    art = floor.normalize(BODY)
    for s in spans:
        assert floor.span_supported(art, s, "NVDA"), s


def test_unverifiable_span_falls_back_to_body_window():
    """A sentiment span that is not verbatim in the body (e.g. from a cleaned
    text path) is replaced by a slice of the body itself."""
    assets = _analyzer()._build_assets(
        _ner(_entity()),
        [{"ticker": "NVDA", "evidence_spans": ["text that is not in the body"]}],
        BODY)
    art = floor.normalize(BODY)
    assert assets[0].evidence_spans
    for s in assets[0].evidence_spans:
        assert floor.span_supported(art, s, "NVDA"), s


def test_unfindable_asset_emits_no_spans_rather_than_a_failing_one():
    """An empty list is skipped by the floor; a failing span is punished."""
    assets = _analyzer()._build_assets(
        _ner(_entity(text="zzzcorp", ticker="ZZZ", name="Zzz Corp",
                     forms=["zzzcorp"])),
        [{"ticker": "ZZZ", "evidence_spans": ["not in the body either"]}], BODY)
    assert assets[0].evidence_spans == []
    result = floor.evaluate(_sub(assets), BODY)
    assert result.span_failures == set()
    assert result.unevidenced_assets == set()
    # ... and the asset still reaches keeper judgment fields.
    assert runner._judgment_fields(_sub(assets), result)["assets"] == ["ZZZ"]


def test_spans_never_fail_the_floor_end_to_end():
    assets = _analyzer()._build_assets(_ner(_entity()), [], BODY)
    result = floor.evaluate(_sub(assets), BODY)
    assert result.floor_pass
    assert result.span_failures == set()
    assert result.unevidenced_assets == set()
    assert runner._judgment_fields(_sub(assets), result)["assets"] == ["NVDA"]


# ---- claim grounding + calibration -------------------------------------------------

def test_ungroundable_claim_is_dropped():
    claims = _analyzer()._build_numeric_claims([_claim(999.9, "%")], BODY)
    assert claims == []


def test_literal_claim_is_kept_with_high_confidence():
    claims = _analyzer()._build_numeric_claims(
        [_claim(12.5, "%", confidence=0.5)], BODY)
    assert len(claims) == 1
    assert claims[0].confidence >= 0.95


def test_scaled_claim_in_plain_unit_grounds_literally():
    claims = _analyzer()._build_numeric_claims(
        [_claim(44.1e9, "usd", metric="revenue", confidence=0.9)], BODY)
    assert len(claims) == 1
    assert claims[0].confidence >= 0.95


def test_scale_ambiguous_claim_is_inferred_and_capped():
    """44.1 'billion usd' matches only under a scale reading — adjudicated,
    never auto-granted — so confidence stays honest."""
    claims = _analyzer()._build_numeric_claims(
        [_claim(44.1, "billion usd", metric="revenue", confidence=0.9)], BODY)
    assert len(claims) == 1
    assert claims[0].confidence <= 0.7


def test_filtering_happens_before_the_cap():
    raw = [_claim(999.9, "%")] * 45 + [_claim(12.5, "%")]
    claims = _analyzer()._build_numeric_claims(raw, BODY)
    assert len(claims) == 1


def test_no_content_keeps_claims_unfiltered():
    """Without a body to check against, dropping everything would be worse."""
    claims = _analyzer()._build_numeric_claims([_claim(999.9, "%")], "")
    assert len(claims) == 1
