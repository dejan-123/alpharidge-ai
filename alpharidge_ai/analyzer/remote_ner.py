"""
RemoteNEREngine — drop-in HTTP client replacement for NERFusionEngine.

When NLP_SERVER_URL is configured, ArticleIntelligenceAnalyzer uses this class
instead of loading local GPU models.  The interface is identical to NERFusionEngine
so no other code in the pipeline changes.

CPU-only components that stay local on each miner
  - AssetExtractor  (keyword gazetteer — regex/JSON, no GPU)
  - TriageStage     (regex + AssetExtractor, no GPU)
  - AspectSentimentScorer / FinABSA  (already device="cpu" in the codebase)
  - text_stats, language_detector, content_hash  (pure Python)

GPU components that are offloaded to the NLP server
  - spaCy trf NER + GLiNER + Flair + ReFinED + WikiNEuRal + FinBERT
  - SentenceTransformer all-MiniLM-L6-v2  (article embeddings)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx
import numpy as np

from alpharidge_ai.analyzer.ner_fusion import NERResult, ResolvedEntity

logger = logging.getLogger("remote_ner")


# ---------------------------------------------------------------------------
# Proxy embedder — mimics the SentenceTransformer interface used by
# ArticleIntelligenceAnalyzer._init_narrative_index() so the narrative
# centroid matrix is built by calling the server rather than a local model.
# ---------------------------------------------------------------------------

class _RemoteEmbedder:
    """Proxy that looks like sentence_transformers.SentenceTransformer to the rest
    of ArticleIntelligenceAnalyzer._init_narrative_index()."""

    def __init__(self, server_url: str, timeout: float) -> None:
        self._url = server_url
        self._timeout = timeout

    def encode(self, texts: List[str], normalize_embeddings: bool = True) -> np.ndarray:
        """Batch-encode texts via the NLP server's /embed/batch endpoint."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        resp = httpx.post(
            f"{self._url}/embed/batch",
            json={"texts": texts, "normalize_embeddings": normalize_embeddings},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return np.array(resp.json()["embeddings"], dtype=np.float32)


# ---------------------------------------------------------------------------
# RemoteNEREngine
# ---------------------------------------------------------------------------

class RemoteNEREngine:
    """
    Drop-in replacement for NERFusionEngine that delegates all GPU-heavy
    inference to the centralised NLP server.

    Interface parity with NERFusionEngine
      .extract_and_resolve(title, body, raw_html)  -> NERResult
      .encode_text(text)                           -> Optional[List[float]]
      ._embedder                                   -> _RemoteEmbedder (not None)
      ._asset_extractor                            -> local AssetExtractor (CPU)

    Usage (via config)
      Set NLP_SERVER_URL=http://<gpu-host>:8765 in .miner_env.
      setup_article_intelligence_analyzer() picks this up automatically.
    """

    def __init__(self, server_url: str, timeout: float = 60.0) -> None:
        self._url = server_url.rstrip("/")
        self._timeout = timeout
        # Proxy embedder — satisfies the _embedder is not None check in
        # _init_narrative_index() and implements .encode(texts, ...).
        self._embedder = _RemoteEmbedder(self._url, timeout)
        # AssetExtractor is pure CPU (regex + JSON data files) — keep local so
        # sector extraction in ArticleIntelligenceAnalyzer.analyze() doesn't need
        # a round trip.
        from alpharidge_ai.analyzer.asset_extractor import AssetExtractor
        self._asset_extractor = AssetExtractor()
        logger.info(f"[RemoteNER] Pointing at NLP server: {self._url}")

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def extract_and_resolve(
        self, title: str, body: str, raw_html: Optional[str] = None
    ) -> NERResult:
        """Send article to the NLP server and deserialise the NERResult."""
        payload: Dict[str, Any] = {"title": title or "", "body": body or ""}
        if raw_html:
            payload["raw_html"] = raw_html
        resp = httpx.post(
            f"{self._url}/ner/extract",
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return NERResult(
            resolved_assets=[ResolvedEntity(**e) for e in data["resolved_assets"]],
            resolved_entities=[ResolvedEntity(**e) for e in data["resolved_entities"]],
            money_values=data["money_values"],
            percentages=data["percentages"],
            dates=data["dates"],
            sentence_sentiments=data["sentence_sentiments"],
            detected_language=data["detected_language"],
        )

    def encode_text(self, text: str) -> Optional[List[float]]:
        """Encode a single string via the NLP server's /embed endpoint."""
        if not text or not text.strip():
            return None
        resp = httpx.post(
            f"{self._url}/embed",
            json={"text": text},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        return resp.json().get("embedding")
