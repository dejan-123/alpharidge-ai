"""Resilience fixes: bounded LLM retries and per-validator send serialization."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
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


# ---- per-item isolation: no batch is lost to one bad item ----------------------

def _tweet(id, text):
    return SimpleNamespace(id=id, text=text, analysis=None)


def _classification():
    ns = SimpleNamespace
    return ns(sentiment=ns(value="neutral"), asset_id=1, asset_symbol="BTC",
              content_type=ns(value="other"), technical_quality=ns(value="none"),
              market_analysis=ns(value="other"), impact_potential=ns(value="NONE"))


def _tweet_miner():
    m = _miner()
    m.analyzer = MagicMock()
    m.wallet = MagicMock()
    m.metagraph = MagicMock()
    m._send_processed_batch = MagicMock()
    return m


def test_tweet_item_exception_does_not_abort_batch():
    m = _tweet_miner()
    m.analyzer.classify_post.side_effect = [
        _classification(), RuntimeError("boom"), _classification()]
    syn = SimpleNamespace(tweet_batch=[_tweet(1, "a"), _tweet(2, "b"), _tweet(3, "c")])
    Miner._process_and_send_tweets(m, syn, "5VAL")
    # The failed item carries a neutral default; every item is signed and sent.
    assert all(t.analysis is not None for t in syn.tweet_batch)
    assert len(syn.miner_signatures) == 3
    assert m._send_processed_batch.call_count == 1


def test_tweet_failed_classify_retries_once_then_defaults():
    m = _tweet_miner()
    m.analyzer.classify_post.return_value = None
    syn = SimpleNamespace(tweet_batch=[_tweet(1, "x")])
    Miner._process_and_send_tweets(m, syn, "5VAL")
    assert m.analyzer.classify_post.call_count == 2
    assert syn.tweet_batch[0].analysis is not None
    assert len(syn.miner_signatures) == 1


def test_tweet_empty_text_still_signed():
    m = _tweet_miner()
    m.analyzer.classify_post.return_value = _classification()
    syn = SimpleNamespace(tweet_batch=[_tweet(1, None), _tweet(2, "x")])
    Miner._process_and_send_tweets(m, syn, "5VAL")
    assert all(t.analysis is not None for t in syn.tweet_batch)
    assert len(syn.miner_signatures) == 2
    m.analyzer.classify_post.assert_called_once()  # only the non-empty tweet


def _article(id):
    return SimpleNamespace(id=id, title="Title", url="u", source="s",
                           published=None, summary=None,
                           content="some article body", analysis=None)


def _article_miner():
    m = _miner()
    m._article_executor = ThreadPoolExecutor(max_workers=2)
    m.triage_stage = MagicMock()
    m.article_intel_analyzer = MagicMock()
    m.news_analyzer = MagicMock()
    m.wallet = MagicMock()
    m.metagraph = MagicMock()
    m._send_processed_batch = MagicMock()
    return m


def test_article_item_exception_still_sends_blob():
    m = _article_miner()
    m.triage_stage.evaluate.side_effect = RuntimeError("gazetteer exploded")
    syn = SimpleNamespace(article_batch=[_article(1), _article(2), _article(3)])
    Miner._process_and_send_articles(m, syn, "5VAL")
    # Every article got a fallback blob (irrelevant claim + proof_of_read) and
    # the batch still went out instead of timing out.
    assert all(a.analysis is not None for a in syn.article_batch)
    for a in syn.article_batch:
        blob = a.analysis.analysis_data
        assert blob["triage"]["label"] == "irrelevant"
        assert blob["proof_of_read"]
    assert len(syn.miner_signatures) == 3
    assert m._send_processed_batch.call_count == 1


# ---- push-back send retry ------------------------------------------------------

def _send_miner(monkeypatch, outcomes):
    """A Miner wired for _send_processed_batch: `outcomes` is a list of HTTP
    status codes or Exceptions consumed one per forward() call."""
    m = _miner()
    m.wallet = MagicMock()
    m.metagraph = MagicMock()
    m.metagraph.hotkeys = ["5VAL"]
    m.metagraph.axons = [SimpleNamespace()]
    m._resend_queue = MagicMock()
    calls = {"n": 0}
    outcomes = list(outcomes)

    class FakeDendrite:
        def __init__(self, wallet=None):
            pass

        async def forward(self, axons, synapse, timeout, deserialize):
            calls["n"] += 1
            out = outcomes.pop(0)
            if isinstance(out, Exception):
                raise out
            return [SimpleNamespace(dendrite=SimpleNamespace(
                status_code=out, status_message="x"))]

    monkeypatch.setattr("neurons.miner.bt.Dendrite", FakeDendrite)
    return m, calls


def test_send_success_does_not_retry(monkeypatch):
    m, calls = _send_miner(monkeypatch, [200])
    Miner._send_processed_batch(m, SimpleNamespace(), "5VAL", "TweetBatch", time.perf_counter())
    assert calls["n"] == 1
    m.metagraph.sync.assert_not_called()


def test_send_non_200_resyncs_and_retries(monkeypatch):
    m, calls = _send_miner(monkeypatch, [500, 200])
    Miner._send_processed_batch(m, SimpleNamespace(), "5VAL", "TweetBatch", time.perf_counter())
    assert calls["n"] == 2
    assert m.metagraph.sync.call_count == 1


def test_send_exception_retries_once_then_gives_up(monkeypatch):
    m, calls = _send_miner(monkeypatch, [RuntimeError("down"), RuntimeError("down"), 200])
    Miner._send_processed_batch(m, SimpleNamespace(), "5VAL", "ArticleBatch", time.perf_counter())
    assert calls["n"] == 2  # at most one retry
    # Exhausted pushes are parked on the resend queue, not dropped.
    assert m._resend_queue.put.call_count == 1


def test_send_retries_when_hotkey_missing(monkeypatch):
    m, calls = _send_miner(monkeypatch, [200])
    m.metagraph.hotkeys = MagicMock()
    m.metagraph.hotkeys.index.side_effect = [ValueError(), 0]
    Miner._send_processed_batch(m, SimpleNamespace(), "5VAL", "TweetBatch", time.perf_counter())
    assert calls["n"] == 1
    assert m.metagraph.sync.call_count == 1
