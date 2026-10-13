# SN45 Multi-Miner Operator Guide

This guide covers running **20+ miners on a single GPU** using the centralised
NLP inference server architecture.

---

## Architecture overview

```
┌──────────────────────────────────────┐
│   NLP Server  (1 machine, 1 GPU)     │
│   All local GPU models load once:    │
│   spaCy · GLiNER · Flair · ReFinED   │
│   FinBERT · WikiNEuRal · MiniLM      │
│   Port 8765  —  stays always running │
└──────────────┬───────────────────────┘
               │  HTTP (JSON)
   ┌───────────┼───────────┐
   │           │           │
Miner-01   Miner-02  … Miner-20+
(no GPU)   (no GPU)     (no GPU)
Each uses its own OpenRouter API_KEY
for Stage 2 / Stage 3 LLM calls.
```

**What each miner still runs locally (CPU only):**
- `AssetExtractor` — keyword/regex
- `TriageStage` — article triage
- `AspectSentimentScorer` (FinABSA) — CPU-pinned
- OpenRouter API calls (Stage 2 + Stage 3) — own API key

---

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12 | `python3.12 --version` |
| CUDA GPU (≥ 8 GB VRAM) | For the NLP server only |
| tmux | `apt install tmux` |
| Git | `apt install git` |
| Bittensor wallets | One coldkey + one hotkey per miner, all registered on netuid 45 |
| OpenRouter API keys | One per miner — get at openrouter.ai |

---

## 1. Installation

Do this once on the GPU server machine (and on each miner machine if separate).

```bash
git clone https://github.com/your-org/alpharidge-ai.git
cd alpharidge-ai

# Creates .venv, installs PyTorch + all ML dependencies (~44 GB models download on first run)
./install.sh
```

> First boot downloads ~44 GB of models. Run the NLP server once and let it
> fully load before starting any miners.

---

## 2. Configure the fleet

```bash
# Copy the example config — miners.yaml is git-ignored (keeps API keys safe)
cp miners.yaml.example miners.yaml
nano miners.yaml   # or use any editor
```

**`miners.yaml` structure:**

```yaml
global:
  nlp_server_url: "http://localhost:8765"   # change to GPU server IP if separate machines
  netuid: 45
  subtensor_network: finney
  axon_external_ip: "YOUR_PUBLIC_IP"        # your server's public IP
  wallet_name: my_cold_wallet               # bittensor coldkey name
  auto_restart: true                        # auto-restart miner on crash
  restart_delay: 30                         # seconds before restart

miners:
  - name: miner-01
    hotkey: hotkey1          # bittensor hotkey name (bt wallet)
    api_key: sk-or-KEY1      # OpenRouter API key
    axon_port: 18100

  - name: miner-02
    hotkey: hotkey2
    api_key: sk-or-KEY2
    axon_port: 18101

  # Add more miners — each needs unique hotkey, api_key, axon_port
```

> Each `axon_port` must be different and open in your firewall.

---

## 3. Start the NLP server

The NLP server must be running **before** any miner starts.
Run it on the GPU machine and leave it running permanently.

```bash
# Start in a detached tmux session (recommended)
./nlp_server/run_server.sh --tmux
```

**First-time startup takes ~45–90 seconds** while models load.
Check it is ready:

```bash
curl http://localhost:8765/health
# Expected: {"status":"ok","engine_loaded":true,"embedder_loaded":true}
```

**Useful tmux commands for the server:**

```bash
tmux attach -t sn45-nlp       # watch server logs live
# Inside tmux:
Ctrl-b d                      # detach (server keeps running)
tmux kill-session -t sn45-nlp # stop the server
```

---

## 4. Start all miners

```bash
# Start every miner defined in miners.yaml
python scripts/mine.py start

# Or start just one miner
python scripts/mine.py start --name miner-01
```

All miners launch in a tmux session called `sn45-miners`.
Each miner gets its own window inside that session.

---

## 5. Monitor status

### Quick status snapshot

```bash
python scripts/mine.py status
```

### Live auto-refreshing table (every 5 seconds)

```bash
python scripts/mine.py status --watch 5
```

### Also check on-chain registration (slow — queries subtensor)

```bash
python scripts/mine.py status --watch 10 --chain
```

**Status table columns:**

| Column | Meaning |
|---|---|
| **Process** | `● RUN` = tmux window alive · `✗ DOWN` = not running |
| **Axon** | `✓ open` = port accepting connections (real readiness) · `… wait` = starting up |
| **Last seen** | Timestamp of most recent log line |
| **Errors** | Count of ERROR/Exception lines in recent log |
| **Last activity** | Most recent request-related log line |
| **UID** | On-chain UID (only shown with `--chain`) |

### Watch a specific miner's live logs

```bash
# Attaches to that miner's tmux window
python scripts/mine.py logs --name miner-01

# Detach without stopping the miner:
Ctrl-b d

# Or tail the log file directly
tail -f logs/miners/miner-01.log
```

### Browse all miners in tmux

```bash
python scripts/mine.py attach          # attach to the fleet session
# Switch between miner windows:
Ctrl-b w                               # list all windows (select with arrow keys)
Ctrl-b n / Ctrl-b p                    # next / previous window
Ctrl-b d                               # detach
```

---

## 6. Stop and restart miners

```bash
# Stop all miners
python scripts/mine.py stop

# Stop one miner
python scripts/mine.py stop --name miner-03

# Restart all miners
python scripts/mine.py restart

# Restart one miner (e.g. after changing its config)
python scripts/mine.py restart --name miner-01
```

> Miners with `auto_restart: true` restart themselves automatically on crash.
> You only need to run `restart` manually after a config or code change.

---

## 7. Update the repository

When the upstream repo has a new release, run the update script on **every
machine** (GPU server + all miner machines) **before restarting anything**.

```bash
./scripts/update.sh
```

The script will:
- Pull the latest code from `origin/main`
- Tell you if the change is **consensus-critical** (NER / determinism code changed)
- Re-run `pip install -r requirements.txt` if dependencies changed

### After updating

**If the script reports consensus-critical changes:**

> ‼ ALL nodes (NLP server + all miners + all validators) must be updated
> and restarted before any of them starts processing articles again.

```bash
# 1. Stop all miners first
python scripts/mine.py stop

# 2. Restart NLP server (reloads updated model code — takes ~45 s)
tmux kill-session -t sn45-nlp
./nlp_server/run_server.sh --tmux

# 3. Wait until server is healthy
curl http://localhost:8765/health

# 4. Restart all miners
python scripts/mine.py start

# 5. Restart validator (on validator machine)
# kill and re-run your validator process
```

**If the script reports no consensus-critical changes:**

```bash
# Safe to restart nodes independently (no consensus risk)
./nlp_server/run_server.sh --tmux    # restart server
python scripts/mine.py restart       # restart miners
```

---

## 8. Adding a new miner

1. **Create and register a new hotkey:**
   ```bash
   btcli wallet new_hotkey --wallet.name my_cold_wallet --wallet.hotkey hotkey21
   btcli subnet register --netuid 45 --wallet.name my_cold_wallet --wallet.hotkey hotkey21
   ```

2. **Get a new OpenRouter API key** at [openrouter.ai](https://openrouter.ai)

3. **Add an entry to `miners.yaml`:**
   ```yaml
   - name: miner-21
     hotkey: hotkey21
     api_key: sk-or-NEW_KEY
     axon_port: 18120        # pick an unused port
   ```

4. **Start the new miner:**
   ```bash
   python scripts/mine.py start --name miner-21
   ```

No restart of the NLP server or other miners is needed.

---

## 9. Troubleshooting

### Miner shows `✗ DOWN` after start

```bash
python scripts/mine.py logs --name miner-01   # see what went wrong
```

Common causes:
- **NLP server not ready** — check `curl http://localhost:8765/health`
- **Wrong hotkey name** — check `btcli wallet list`
- **Port conflict** — check `ss -tlnp | grep 18100`
- **API_KEY missing** — make sure `api_key` is set in `miners.yaml`

### Axon stuck at `… wait`

The miner process is running but hasn't opened the port yet.
It downloads models on first run (~44 GB). Wait a few minutes and check again.

### NLP server crashes / stops

```bash
# Check if session still exists
tmux ls

# Restart it
./nlp_server/run_server.sh --tmux
```

### Miner is deregistered (UID gone)

```bash
# Check registration
python scripts/mine.py status --chain

# Re-register
btcli subnet register --netuid 45 --wallet.name my_cold_wallet --wallet.hotkey hotkey1

# Then restart that miner
python scripts/mine.py restart --name miner-01
```

### Check GPU usage on the NLP server

```bash
watch -n 2 nvidia-smi
```

Typical: 4–6 GB VRAM used by the NLP server at idle. Under load (many
concurrent NER requests) it may spike briefly but the serialisation lock
prevents OOM.

---

## 9b. Tunable environment variables

| Variable | Default | What it does |
|---|---|---|
| `MINER_ARTICLE_DEADLINE_S` | `400` | Per-batch wall-clock budget for article LLM work. Articles that can't finish in time are emitted as safe `irrelevant` claims instead of burning the batch past the validator's ~900s lease. |
| `MINER_RESEND_EXPIRY_S` | `800` | How long after dispatch the resend worker keeps retrying a failed batch push before dropping it. |
| `MINER_ARTICLE_WORKERS` | `8` | Concurrent per-article processing threads. |
| `MINER_ARTICLE_OUTER_WORKERS` | `2` | Concurrent batches processed in parallel. |
| `MINER_MAX_WORKERS` | `4` | Background threads shared by tweet/telegram loops. |

Set them in `miners.yaml` under the miner's `env:` block (or `.miner_env` for a single-miner setup).

---

## 10. File reference

| Path | Purpose |
|---|---|
| `miners.yaml` | Your fleet config (hotkeys + API keys) — **git-ignored** |
| `miners.yaml.example` | Template to copy from |
| `nlp_server/run_server.sh` | Start the NLP inference server |
| `scripts/mine.py` | Fleet manager (start / stop / status / logs) |
| `scripts/update.sh` | Safe coordinated update from git |
| `logs/miners/<name>.log` | Per-miner log file |
| `.miner_env` | Fallback single-miner env file (optional when using `miners.yaml`) |

---

## 11. Quick reference card

```bash
# NLP server
./nlp_server/run_server.sh --tmux    # start
curl localhost:8765/health           # health check
tmux attach -t sn45-nlp              # watch logs
tmux kill-session -t sn45-nlp        # stop

# Fleet
python scripts/mine.py start         # start all miners
python scripts/mine.py stop          # stop all miners
python scripts/mine.py restart       # restart all miners
python scripts/mine.py status        # one-shot status
python scripts/mine.py status --watch 5        # live table
python scripts/mine.py status --watch 5 --chain  # + on-chain UIDs
python scripts/mine.py logs --name miner-01    # live logs
python scripts/mine.py attach        # enter tmux session

# Update
./scripts/update.sh                  # pull + check consensus impact
```
