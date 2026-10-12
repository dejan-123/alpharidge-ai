"""
SN45 NLP Inference Server

Centralised FastAPI service that loads all GPU-heavy local models once and
exposes them over HTTP.  Miners call this server for Stage 1 (NER + embeddings)
and then call their own OpenRouter key for Stage 2/3 (LLM calls).

GPU models served here
  - spaCy en_core_web_trf
  - GLiNER urchade/gliner_base  +  urchade/gliner_multi-v2.1
  - Flair flair/ner-english-ontonotes-large
  - Babelscape/wikineural-multilingual-ner
  - Amazon ReFinED  aida_model (Wikidata entity linking)
  - ProsusAI/finbert  (sentence-level sentiment)
  - all-MiniLM-L6-v2  (article embeddings)

Endpoints
  POST /ner/extract        - run full NERFusionEngine.extract_and_resolve()
  POST /embed              - encode a single text string -> 384-d vector
  POST /embed/batch        - encode a list of texts (used by narrative index)
  GET  /health             - liveness check

Run
  ./nlp_server/run_server.sh
  # or directly:
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m nlp_server.server
"""

from __future__ import annotations

# ---- determinism BEFORE any CUDA/torch import ----
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Reduce long-term reserved-VRAM growth from CUDA allocator fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,expandable_segments:True")

import dataclasses
import gc
import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Ensure accelerate's meta-device patch is serialised before any model load
from alpharidge_ai.analyzer.aspect_sentiment import install_meta_init_guard
install_meta_init_guard()

from alpharidge_ai.analyzer.determinism import configure_determinism
configure_determinism()

from alpharidge_ai.analyzer.ner_fusion import NERFusionEngine, NERResult, ResolvedEntity

logger = logging.getLogger("nlp_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_engine: Optional[NERFusionEngine] = None
# Serialise GPU inference: one request at a time prevents OOM and keeps output
# deterministic (cuBLAS workspace is reused, no stream contention).
# Use RLock so NERFusionEngine can safely nest model calls inside the same lock.
_gpu_lock = threading.RLock()

# Periodic VRAM + RAM flush: run every N requests to prevent allocator growth.
_request_counter = 0
_GC_EVERY_N_REQUESTS = 25  # run GC + malloc_trim every N requests
_RESERVED_FLUSH_THRESHOLD_GB = 10.0  # synchronous empty_cache if reserved VRAM grows above this

# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _entity_to_dict(e: ResolvedEntity) -> Dict[str, Any]:
    return dataclasses.asdict(e)


def _ner_result_to_dict(r: NERResult) -> Dict[str, Any]:
    return {
        "resolved_assets": [_entity_to_dict(e) for e in r.resolved_assets],
        "resolved_entities": [_entity_to_dict(e) for e in r.resolved_entities],
        "money_values": r.money_values,
        "percentages": r.percentages,
        "dates": r.dates,
        "sentence_sentiments": r.sentence_sentiments,
        "detected_language": r.detected_language,
    }

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class NERRequest(BaseModel):
    title: str
    body: str
    raw_html: Optional[str] = None


class EmbedRequest(BaseModel):
    text: str


class EmbedBatchRequest(BaseModel):
    texts: List[str]
    normalize_embeddings: bool = True

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SN45 NLP Server",
    description="Centralised GPU inference for SN45 miners",
    version="1.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    global _engine
    logger.info("[SERVER] Loading NERFusionEngine — this takes 1-3 minutes on first run …")
    t0 = time.time()
    _engine = NERFusionEngine()
    logger.info(f"[SERVER] NERFusionEngine ready in {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        import torch
        vram_used_gb  = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else None
        vram_reserved = torch.cuda.memory_reserved()  / 1e9 if torch.cuda.is_available() else None
    except Exception:
        vram_used_gb = vram_reserved = None
    return {
        "status": "ok",
        "engine_loaded": _engine is not None,
        "embedder_loaded": _engine is not None and _engine._embedder is not None,
        "requests_served": _request_counter,
        "vram_allocated_gb": round(vram_used_gb, 2) if vram_used_gb is not None else None,
        "vram_reserved_gb":  round(vram_reserved, 2) if vram_reserved is not None else None,
    }


@app.post("/ner/extract")
def ner_extract(req: NERRequest) -> Dict[str, Any]:
    """Run the full NER fusion pipeline on one article.

    GPU inference is serialised inside NERFusionEngine so concurrent miner
    requests queue only during actual model forward passes, not CPU prep.
    """
    global _request_counter
    if _engine is None:
        raise HTTPException(503, detail="Engine not initialised yet — try again shortly")
    t0 = time.perf_counter()
    result = _engine.extract_and_resolve(req.title, req.body, req.raw_html, gpu_lock=_gpu_lock)
    total_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[SERVER] /ner/extract completed in {total_ms:.1f}ms")

    _request_counter += 1

    # Synchronous flush if PyTorch allocator has cached too much reserved VRAM.
    try:
        import torch
        if torch.cuda.is_available():
            reserved_gb = torch.cuda.memory_reserved() / 1e9
            if reserved_gb > _RESERVED_FLUSH_THRESHOLD_GB:
                _flush_memory(sync=True)
    except Exception:
        pass

    if _request_counter % _GC_EVERY_N_REQUESTS == 0:
        # Run in background — never block the response or delay the next request
        threading.Thread(target=lambda: _flush_memory(sync=False), daemon=True).start()

    return _ner_result_to_dict(result)


def _flush_memory(sync: bool = True) -> None:
    """Release cached VRAM and return fragmented heap to the OS."""
    try:
        import torch
        if torch.cuda.is_available():
            if sync:
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass
    gc.collect()
    gc.collect()  # second pass catches cycles missed by first
    try:
        import ctypes
        ctypes.CDLL(None).malloc_trim(0)
    except Exception:
        pass
    logger.info(f"[SERVER] Memory flush at request #{_request_counter} "
                f"(gc + malloc_trim + cuda cache cleared)")


@app.post("/embed")
def embed(req: EmbedRequest) -> Dict[str, Any]:
    """Encode a single string to a 384-d L2-normalised vector."""
    if _engine is None:
        raise HTTPException(503, detail="Engine not initialised yet")
    if not req.text or not req.text.strip():
        return {"embedding": None}
    t0 = time.perf_counter()
    vec = _engine.encode_text(req.text, gpu_lock=_gpu_lock)
    total_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[SERVER] /embed completed in {total_ms:.1f}ms")

    # Same allocator flush after heavy embedding calls.
    try:
        import torch
        if torch.cuda.is_available():
            reserved_gb = torch.cuda.memory_reserved() / 1e9
            if reserved_gb > _RESERVED_FLUSH_THRESHOLD_GB:
                _flush_memory(sync=True)
    except Exception:
        pass
    return {"embedding": vec}


@app.post("/embed/batch")
def embed_batch(req: EmbedBatchRequest) -> Dict[str, Any]:
    """Encode a list of strings (used by the narrative taxonomy index at miner startup)."""
    if _engine is None:
        raise HTTPException(503, detail="Engine not initialised yet")
    if _engine._embedder is None:
        raise HTTPException(503, detail="SentenceTransformer embedder not loaded")
    if not req.texts:
        return {"embeddings": []}
    t0 = time.perf_counter()
    embeddings = _engine.encode_texts(
        req.texts, normalize_embeddings=req.normalize_embeddings, gpu_lock=_gpu_lock
    )
    total_ms = (time.perf_counter() - t0) * 1000
    logger.info(f"[SERVER] /embed/batch completed {len(req.texts)} texts in {total_ms:.1f}ms")

    try:
        import torch
        if torch.cuda.is_available():
            reserved_gb = torch.cuda.memory_reserved() / 1e9
            if reserved_gb > _RESERVED_FLUSH_THRESHOLD_GB:
                _flush_memory(sync=True)
    except Exception:
        pass
    return {"embeddings": embeddings}


# ---------------------------------------------------------------------------
# Global error handler — never let an uncaught exception reveal a stack trace
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _global_exc(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"[SERVER] Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.getenv("NLP_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("NLP_SERVER_PORT", "8765"))
    workers = int(os.getenv("NLP_SERVER_WORKERS", "1"))  # keep at 1: one GPU
    logger.info(f"[SERVER] Starting on {host}:{port} workers={workers}")
    uvicorn.run(
        "nlp_server.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )
