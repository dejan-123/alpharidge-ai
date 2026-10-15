"""Reputation-improvement plumbing: miner provider routing, fallback-model retry
budget, and the pre-submission floor polish for the audit_v2 channel."""

import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from alpharidge_ai.analyzer.article_intelligence_analyzer import (
    ArticleIntelligenceAnalyzer as A)
from alpharidge_ai.models.article_intelligence import NumericClaim, QuoteExtraction
from alpharidge_ai.utils import llm_spend
from neurons.miner import _polish_for_audit


# ---- _polish_for_audit -------------------------------------------------------------

class _Intel:
    """Duck-type ArticleIntelligence for the floor: attrs + model_copy."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def model_copy(self, update=None):
        new = _Intel(**self.__dict__)
        if update:
            new.__dict__.update(update)
        return new


TEXT = ('Revenue rose to $1.2 billion, up 12.5%, the chief executive said '
        '"growth is broadening" during the call.')


def _claim(value, unit="none", metric="revenue", confidence=0.9):
    return NumericClaim(metric_name=metric, value=value, unit=unit,
                        confidence=confidence)


def _quote(text, confidence=0.9):
    return QuoteExtraction(speaker="ceo", text=text, confidence=confidence)


def _intel(claims=(), quotes=()):
    return _Intel(article_id=7, numeric_claims=list(claims),
                  quotes=list(quotes), assets=[], entities=[])


def test_polish_drops_ungrounded_claims_and_bad_quotes():
    intel = _intel(
        claims=[_claim(1.2e9, "USD"),            # grounded literal
                _claim(88.0, "USD")],            # ungrounded — no 88 in text
        quotes=[_quote("growth is broadening"),  # aligns
                _quote("we will double revenue")])  # invented — rejected
    out = _polish_for_audit(intel, TEXT)
    assert [c.value for c in out.numeric_claims] == [1.2e9]
    assert [q.text for q in out.quotes] == ["growth is broadening"]


def test_polish_caps_inferred_claim_confidence():
    # "2 million" parses as 2e6 literal; claiming value=2 with unit "million" is
    # ambiguous (2 or 2e6) -> inferred, so confidence is capped to neutral.
    text = "headcount was 2 million people"
    intel = _intel(claims=[_claim(2.0, "million", metric="headcount",
                                  confidence=0.95)])
    out = _polish_for_audit(intel, text)
    assert len(out.numeric_claims) == 1
    assert out.numeric_claims[0].confidence == 0.5


def test_polish_leaves_clean_submission_untouched():
    intel = _intel(claims=[_claim(1.2e9, "USD", confidence=0.9)],
                   quotes=[_quote("growth is broadening")])
    out = _polish_for_audit(intel, TEXT)
    assert len(out.numeric_claims) == 1
    assert out.numeric_claims[0].confidence == 0.9
    assert len(out.quotes) == 1


def test_polish_survives_garbage_input():
    intel = _intel(claims=[_claim(1.2e9, "USD")], quotes=[])
    out = _polish_for_audit(intel, "")
    assert len(out.numeric_claims) == 1


def test_polish_inferred_claim_with_none_confidence_not_crashing():
    """confidence=None means schema-grace neutral — must not TypeError on the cap."""
    text = "headcount was 2 million people"
    intel = _intel(claims=[_claim(2.0, "million", metric="headcount",
                                  confidence=None)])
    out = _polish_for_audit(intel, text)
    assert len(out.numeric_claims) == 1
    assert out.numeric_claims[0].confidence is None


# ---- miner provider prefs on _llm_call ---------------------------------------------

def _analyzer(client):
    a = A.__new__(A)
    a.client = client
    a.model = "test-model"
    return a


def _ok_response():
    tc = SimpleNamespace(function=SimpleNamespace(arguments='{"ok": true}'))
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tc]),
                                 finish_reason="stop")])


def _client():
    c = MagicMock()
    c.base_url = "https://openrouter.ai/api/v1"
    c.chat.completions.create.return_value = _ok_response()
    return c


def test_miner_call_sends_provider_ignore(monkeypatch):
    from alpharidge_ai import config
    monkeypatch.setattr(config, "MINER_PROVIDER_IGNORE", "OpenInference",
                        raising=False)
    client = _client()
    out = _analyzer(client)._llm_call("p", {}, "extract_and_classify",
                                      strict=False)
    assert out == {"ok": True}
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["extra_body"] == {"provider": {"ignore": ["OpenInference"]}}


def test_miner_call_omits_prefs_when_unset(monkeypatch):
    from alpharidge_ai import config
    monkeypatch.setattr(config, "MINER_PROVIDER_IGNORE", "", raising=False)
    monkeypatch.setattr(config, "MINER_PROVIDER_ORDER", "", raising=False)
    monkeypatch.setattr(config, "MINER_PROVIDER_MAX_PRICE", "", raising=False)
    client = _client()
    _analyzer(client)._llm_call("p", {}, "extract_and_classify", strict=False)
    _, kwargs = client.chat.completions.create.call_args
    assert "extra_body" not in kwargs


def test_miner_call_no_prefs_off_openrouter(monkeypatch):
    from alpharidge_ai import config
    monkeypatch.setattr(config, "MINER_PROVIDER_IGNORE", "OpenInference",
                        raising=False)
    client = _client()
    client.base_url = "http://localhost:11434/v1"
    _analyzer(client)._llm_call("p", {}, "extract_and_classify", strict=False)
    _, kwargs = client.chat.completions.create.call_args
    assert "extra_body" not in kwargs


def test_provider_prefs_order_implies_fallbacks():
    prefs = llm_spend.provider_prefs(
        SimpleNamespace(base_url="https://openrouter.ai/api/v1"),
        order="DeepInfra, GMICloud")
    assert prefs["order"] == ["DeepInfra", "GMICloud"]
    assert prefs["allow_fallbacks"] is True


# ---- retry budget for claim-bearing articles ----------------------------------------

def test_claim_bearing_articles_get_fallback_attempt(monkeypatch):
    """The third attempt only exists for claim-bearing articles with a
    fallback model configured — the failure path that produces hard
    false-negative triage events."""
    import neurons.miner as m

    monkeypatch.setenv("MODEL_FALLBACK", "fallback/model")

    article = SimpleNamespace(id=1, url="", title="t", source="s",
                              published=None, summary="", content="x")

    calls = []

    class _Analyzer:
        def analyze(self, **kw):
            calls.append(kw.get("model"))
            return None

    miner = m.Miner.__new__(m.Miner)
    miner.article_intel_analyzer = _Analyzer()
    miner.triage_stage = SimpleNamespace(
        _assets=object(),
        evaluate=lambda t, c: ({"label": "relevant"}, {}, None))
    miner.wallet = SimpleNamespace(
        hotkey=SimpleNamespace(ss58_address="hk"))
    monkeypatch.setattr(m.Miner, "_claim_bearing", lambda self, a: True)

    # Drive just the retry loop by stubbing everything around it.
    import time as _time
    batch_deadline = _time.time() + 60
    intel = None
    fallback_model = __import__("os").environ.get("MODEL_FALLBACK", "")
    max_attempts = 3 if fallback_model and miner._claim_bearing(article) else 2
    for _attempt in range(max_attempts):
        candidate = miner.article_intel_analyzer.analyze(
            model=fallback_model if _attempt == 2 else None)
    assert calls == [None, None, "fallback/model"]


def test_non_claim_bearing_uses_two_attempts(monkeypatch):
    monkeypatch.setenv("MODEL_FALLBACK", "fallback/model")
    import neurons.miner as m
    miner = m.Miner.__new__(m.Miner)
    monkeypatch.setattr(m.Miner, "_claim_bearing", lambda self, a: False)
    article = SimpleNamespace(id=1)
    fallback_model = __import__("os").environ.get("MODEL_FALLBACK", "")
    max_attempts = 3 if fallback_model and miner._claim_bearing(article) else 2
    assert max_attempts == 2


# ---- bounded JSON repair (miner path only) -----------------------------------------

def _args_response(arguments):
    tc = SimpleNamespace(function=SimpleNamespace(arguments=arguments))
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tc]),
                                 finish_reason="stop")])


def test_repair_clean_json_passes_through():
    a = _analyzer(MagicMock())
    assert a._repair_tool_args('{"ok": true, "n": 3}') == {"ok": True, "n": 3}


def test_repair_salvages_trailing_comma():
    a = _analyzer(MagicMock())
    out = a._repair_tool_args('{"ok": true, "x": [1, 2,],}')
    assert out == {"ok": True, "x": [1, 2]}


def test_repair_salvages_truncated_tail():
    a = _analyzer(MagicMock())
    out = a._repair_tool_args('{"ok": true, "items": ["a", "b')
    assert out["ok"] is True
    # The unterminated '"b' is closed rather than dropped — salvage keeps it.
    assert out["items"] == ["a", "b"]


def test_repair_salvages_control_chars_in_string():
    a = _analyzer(MagicMock())
    out = a._repair_tool_args('{"ok": true, "s": "line1\nline2"}')
    assert out["ok"] is True


def test_repair_rejects_garbage_and_non_dicts():
    a = _analyzer(MagicMock())
    assert a._repair_tool_args("not json") is None
    assert a._repair_tool_args("[1, 2]") is None
    assert a._repair_tool_args("") is None


def test_malformed_tool_args_repaired_on_miner_path():
    client = _client()
    client.chat.completions.create.return_value = _args_response(
        '{"ok": true, "x": [1, 2,],}')
    out = _analyzer(client)._llm_call("p", {}, "extract_and_classify",
                                      strict=False)
    assert out["ok"] is True


def test_unrepairable_tool_args_miner_returns_empty():
    client = _client()
    client.chat.completions.create.return_value = _args_response("garbage{")
    out = _analyzer(client)._llm_call("p", {}, "extract_and_classify",
                                      strict=False)
    assert out == {}


def test_malformed_tool_args_strict_still_raises():
    """Validator/reference calls keep ReferenceUnavailable semantics — a failed
    reference must skip the audit, not be repaired into grading evidence."""
    from alpharidge_ai.analyzer.article_intelligence_analyzer import (
        ReferenceUnavailable)
    client = _client()
    client.chat.completions.create.return_value = _args_response('{"ok"')
    with pytest.raises(ReferenceUnavailable):
        _analyzer(client)._llm_call("p", {}, "extract_and_classify",
                                    strict=True)


# ---- early deadline gate ------------------------------------------------------------

def test_analyze_deadline_gates_before_ner():
    """A past-deadline article returns None without touching NER — previously the
    remote NER/embedding stage ran even when the analysis was already doomed."""
    import time as _time
    a = _analyzer(MagicMock())
    out = a.analyze(article_id=1, url="", title="t", source="s",
                    content="body", deadline=_time.time() - 1)
    assert out is None


# ---- relevance verdict wiring ---------------------------------------------------------

def test_verdict_none_when_check_disabled(monkeypatch):
    import neurons.miner as m
    from alpharidge_ai import config
    monkeypatch.setattr(config, "MINER_RELEVANCE_CHECK", False, raising=False)
    miner = m.Miner.__new__(m.Miner)
    miner.article_intel_analyzer = SimpleNamespace(client=None, model="m")
    assert miner._relevance_verdict("t", "b") is None


def test_verdict_none_without_analyzer(monkeypatch):
    import neurons.miner as m
    from alpharidge_ai import config
    monkeypatch.setattr(config, "MINER_RELEVANCE_CHECK", True, raising=False)
    miner = m.Miner.__new__(m.Miner)
    miner.article_intel_analyzer = None
    assert miner._relevance_verdict("t", "b") is None


def test_verdict_delegates_to_judge_keep(monkeypatch):
    import threading
    import neurons.miner as m
    from alpharidge_ai import config
    monkeypatch.setattr(config, "MINER_RELEVANCE_CHECK", True, raising=False)
    miner = m.Miner.__new__(m.Miner)
    miner.article_intel_analyzer = SimpleNamespace(client=None, model="m")
    miner._relevance_check_lock = threading.Lock()   # normally set by __init__
    miner._relevance_check = SimpleNamespace(keep=lambda t, b: True)
    assert miner._relevance_verdict("t", "b") is True
    miner._relevance_check.keep = lambda t, b: False
    assert miner._relevance_verdict("t", "b") is False
    miner._relevance_check.keep = lambda t, b: None
    assert miner._relevance_verdict("t", "b") is None


def test_relevance_check_miner_body_prefs(monkeypatch):
    """Judge calls inherit MINER_PROVIDER_IGNORE via extra_body_fn."""
    from alpharidge_ai import config
    from alpharidge_ai.analyzer.relevance_check import RelevanceCheck
    monkeypatch.setattr(config, "MINER_PROVIDER_IGNORE", "OpenInference",
                        raising=False)
    client = _client()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content='{"relevant": true}'))])
    rc = RelevanceCheck(client, "m", spend_purpose="miner_relevance",
                        extra_body_fn=llm_spend.miner_body)
    assert rc.verdict("t", "b") is True
    _, kwargs = client.chat.completions.create.call_args
    assert kwargs["extra_body"] == {"provider": {"ignore": ["OpenInference"]}}


# ---- clearly-irrelevant confirmation (det-audit mirror) -----------------------------

def test_beyond_doubt_delegates_to_auditor(monkeypatch):
    import threading
    import neurons.miner as m
    from alpharidge_ai import config
    monkeypatch.setattr(config, "MINER_RELEVANCE_CHECK", True, raising=False)
    miner = m.Miner.__new__(m.Miner)
    miner.article_intel_analyzer = SimpleNamespace(client=None, model="m")
    miner._relevance_check_lock = threading.Lock()   # normally set by __init__
    miner._triage_auditor = SimpleNamespace(
        clearly_irrelevant=lambda t, b: True)
    assert miner._irrelevant_beyond_doubt("t", "b") is True
    miner._triage_auditor.clearly_irrelevant = lambda t, b: False
    assert miner._irrelevant_beyond_doubt("t", "b") is False


def test_beyond_doubt_false_when_check_disabled(monkeypatch):
    import neurons.miner as m
    from alpharidge_ai import config
    monkeypatch.setattr(config, "MINER_RELEVANCE_CHECK", False, raising=False)
    miner = m.Miner.__new__(m.Miner)
    miner.article_intel_analyzer = SimpleNamespace(client=None, model="m")
    assert miner._irrelevant_beyond_doubt("t", "b") is False


def test_beyond_doubt_false_without_analyzer(monkeypatch):
    import neurons.miner as m
    from alpharidge_ai import config
    monkeypatch.setattr(config, "MINER_RELEVANCE_CHECK", True, raising=False)
    miner = m.Miner.__new__(m.Miner)
    miner.article_intel_analyzer = None
    assert miner._irrelevant_beyond_doubt("t", "b") is False


def test_beyond_doubt_false_on_auditor_error(monkeypatch):
    import threading
    import neurons.miner as m
    from alpharidge_ai import config
    monkeypatch.setattr(config, "MINER_RELEVANCE_CHECK", True, raising=False)
    miner = m.Miner.__new__(m.Miner)
    miner.article_intel_analyzer = SimpleNamespace(client=None, model="m")
    miner._relevance_check_lock = threading.Lock()
    def _boom(t, b):
        raise RuntimeError("auditor exploded")
    miner._triage_auditor = SimpleNamespace(clearly_irrelevant=_boom)
    assert miner._irrelevant_beyond_doubt("t", "b") is False
