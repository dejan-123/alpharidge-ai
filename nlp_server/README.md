# SN45 NLP Inference Server

Centralises all GPU-heavy local model inference so that **20+ miners can share a single 8 GB GPU** without each miner needing one.

---

## How it works

The SN45 analysis pipeline has two distinct phases:

| Phase | What runs | Where |
|---|---|---|
| **Stage 1 — Local models** | spaCy trf, GLiNER, Flair, ReFinED, WikiNEuRal, FinBERT, SentenceTransformer | **NLP server (GPU)** |
| **Stage 2/3 — LLM API** | Extract & Classify, Reason & Summarize | **Each miner (own API key)** |
| **CPU helpers** | text_stats, AssetExtractor, TriageStage, FinABSA | **Each miner (no GPU)** |

```
┌─────────────────────────────────────┐
│  NLP Server  (1 machine, 8 GB GPU)  │
│  - NERFusionEngine (all GPU models) │
│  - SentenceTransformer embeddings   │
└──────────────┬──────────────────────┘
               │  HTTP  (JSON)
      ┌────────┴────────┐
      │                 │
  Miner 1 … Miner 20+  (no GPU, cheap CPU machines)
  - text_stats (CPU)
  - TriageStage (CPU)
  - FinABSA (CPU)
  - OpenRouter LLM (own API_KEY)
```

---

## Setup

### 1. GPU server

The NLP server lives inside the same repo and uses the same venv.

```bash
# On the GPU machine, from the repo root:
./nlp_server/run_server.sh
```

Or with PM2:

```bash
pm2 start nlp_server/run_server.sh --name sn45-nlp-server
```

Environment variables (optional):

| Variable | Default | Description |
|---|---|---|
| `NLP_SERVER_HOST` | `0.0.0.0` | Bind address |
| `NLP_SERVER_PORT` | `8765` | Listen port |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Pre-set by the script (do not change) |

The server takes 1–3 minutes to start while it downloads and loads all models on first run.
Check readiness: `curl http://localhost:8765/health`

### 2. Each miner

Add one line to `.miner_env`:

```bash
NLP_SERVER_URL=http://<gpu-server-ip>:8765
```

Each miner still needs its own `API_KEY` (OpenRouter) for Stages 2 and 3.  
No GPU is required on the miner machines.

---

## API reference

All endpoints accept and return JSON.

### `GET /health`
```json
{ "status": "ok", "engine_loaded": true, "embedder_loaded": true }
```

### `POST /ner/extract`
Run the full NER fusion pipeline on one article.

Request:
```json
{ "title": "...", "body": "...", "raw_html": null }
```

Response: serialised `NERResult` — resolved assets, entities, money values, sentence sentiments, detected language.

### `POST /embed`
Encode a single string to a 384-d L2-normalised vector.

Request: `{ "text": "..." }`  
Response: `{ "embedding": [0.12, ...] }`

### `POST /embed/batch`
Encode a list of strings (used at miner startup to build the narrative taxonomy index).

Request: `{ "texts": ["...", "..."], "normalize_embeddings": true }`  
Response: `{ "embeddings": [[...], [...]] }`

---

## Concurrency model

All GPU inference is serialised under a single lock (`threading.Lock`) so concurrent miner requests queue rather than thrash the GPU.

Typical latency:
- NER on a 4 000-char article: **~1 s**
- Single embedding: **~15 ms**
- Batch of 100 narrative texts: **~200 ms**

With 20 miners each processing one article every 10–20 s (dominated by the LLM call), the NLP server comfortably handles the load at ~1 req/s sustained.

---

## Validators

Validators run the analyzer independently to cross-check miner output.  Because validator traffic is much lower (sampled subset), validators can continue using the **local NERFusionEngine** (leave `NLP_SERVER_URL` unset in `.vali_env`).  If you want validators to share the NLP server too, set `NLP_SERVER_URL` in `.vali_env`.

---

## Determinism guarantee

The NLP server launches with `CUBLAS_WORKSPACE_CONFIG=:4096:8` (same as miners/validators in standalone mode) and calls `configure_determinism()` before any model loads.  This ensures miner and validator NER results remain byte-identical for the same article, preserving consensus across the Tier-2 determinism gate.
