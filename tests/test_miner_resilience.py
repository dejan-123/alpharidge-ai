"""Resilience fixes: bounded LLM retries and per-validator send serialization."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from alpharidge_ai.analyzer.article_intelligence_analyzer import (
    ArticleIntelligenceAnalyzer as A)
from neurons.miner import Miner, _analysis_looks_degraded


def _analyzer(client):
    a = A.__new__(A)
    a.client = client
    a.model = "test-model"
    return a


def _miner():
    m = Miner.__new__(Miner)
    m._send_locks = {}
    m._send_locks_guard = threading.Lock()
    return m


def _response(tool_calls):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=tool_calls))])


def _ok_response():
    tc = SimpleNamespace(function=SimpleNamespace(arguments='{"ok": true}'))
    return _response([tc])


# ---- _llm_call retry ---------------------------------------------------------------

def test_llm_call_retries_once_on_empty_tool_calls():
    client = MagicMock()
    client.chat.completions.create.side_effect = [_response(None), _ok_response()]
    out = _analyzer(client)._llm_call("p", {}, "extract_and_classify")
    assert out == {"ok": True}
    assert client.chat.completions.create.call_count == 2


def test_llm_call_retries_once_on_exception():
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        RuntimeError("rate limited"), _ok_response()]
    out = _analyzer(client)._llm_call("p", {}, "extract_and_classify")
    assert out == {"ok": True}
    assert client.chat.completions.create.call_count == 2


def test_llm_call_gives_up_after_two_failures():
    client = MagicMock()
    client.chat.completions.create.return_value = _response(None)
    out = _analyzer(client)._llm_call("p", {}, "extract_and_classify")
    assert out == {}
    assert client.chat.completions.create.call_count == 2


def test_llm_call_does_not_retry_a_success():
    client = MagicMock()
    client.chat.completions.create.return_value = _ok_response()
    out = _analyzer(client)._llm_call("p", {}, "extract_and_classify")
    assert out == {"ok": True}
    assert client.chat.completions.create.call_count == 1


# ---- degraded-analysis detection -----------------------------------------------------

_LONG_CONTENT = "word " * 200


def _intel(entities=1, assets=0, title_embedding=None, body_embedding=None):
    return SimpleNamespace(
        entities=[object()] * entities,
        assets=[object()] * assets,
        title_embedding=title_embedding,
        body_embedding=body_embedding,
        narrative_keywords=["k"],
        economic_data=[],
    )


def test_missing_embeddings_are_degraded():
    intel = _intel(entities=5, title_embedding=None)
    assert _analysis_looks_degraded(intel, _LONG_CONTENT)


def test_empty_entities_on_long_article_is_degraded():
    intel = _intel(entities=0, assets=4,
                   title_embedding=[0.1], body_embedding=[0.1])
    assert _analysis_looks_degraded(intel, _LONG_CONTENT)


def test_empty_entities_on_short_article_is_not_degraded():
    intel = _intel(entities=0, title_embedding=[0.1], body_embedding=[0.1])
    assert not _analysis_looks_degraded(intel, "short stub")


def test_full_analysis_is_not_degraded():
    intel = _intel(entities=3, assets=2,
                   title_embedding=[0.1], body_embedding=[0.1])
    assert not _analysis_looks_degraded(intel, _LONG_CONTENT)


def test_none_content_does_not_crash():
    intel = _intel(entities=0, title_embedding=[0.1], body_embedding=[0.1])
    assert not _analysis_looks_degraded(intel, None)


# ---- per-validator send lock --------------------------------------------------------

def test_send_lock_is_per_validator_hotkey():
    m = _miner()
    lock_a1 = Miner._validator_send_lock(m, "5AAA")
    lock_a2 = Miner._validator_send_lock(m, "5AAA")
    lock_b = Miner._validator_send_lock(m, "5BBB")
    assert lock_a1 is lock_a2
    assert lock_a1 is not lock_b


def test_sends_to_the_same_validator_cannot_overlap():
    m = _miner()
    lock = Miner._validator_send_lock(m, "5AAA")
    active = {"count": 0, "max": 0}

    def send():
        with lock:
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            time.sleep(0.05)
            active["count"] -= 1

    threads = [threading.Thread(target=send) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert active["max"] == 1


def test_sends_to_different_validators_run_concurrently():
    m = _miner()
    lock_a = Miner._validator_send_lock(m, "5AAA")
    lock_b = Miner._validator_send_lock(m, "5BBB")
    entered_b = threading.Event()

    def send_a():
        with lock_a:
            # While A's send is held, B's send to another validator must proceed.
            assert entered_b.wait(timeout=2.0)

    def send_b():
        time.sleep(0.02)
        with lock_b:
            entered_b.set()

    ta, tb = threading.Thread(target=send_a), threading.Thread(target=send_b)
    ta.start()
    tb.start()
    ta.join()
    tb.join()
