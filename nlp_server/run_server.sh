#!/usr/bin/env bash
# Run the SN45 NLP inference server.
#
# Usage:
#   ./nlp_server/run_server.sh              # run in current shell
#   ./nlp_server/run_server.sh --tmux       # launch in a detached tmux session
#
# Optional env overrides:
#   NLP_SERVER_HOST   (default: 0.0.0.0)
#   NLP_SERVER_PORT   (default: 8765)
#   TMUX_SESSION      (default: sn45-nlp)
#   CONDA_ENV         (default: alpharidge_ai)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Deterministic cuBLAS — must be set before any CUDA / PyTorch import.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export BT_NO_PARSE_CLI_ARGS=0

NLP_SERVER_HOST="${NLP_SERVER_HOST:-0.0.0.0}"
NLP_SERVER_PORT="${NLP_SERVER_PORT:-8765}"
TMUX_SESSION="${TMUX_SESSION:-sn45-nlp}"
CONDA_ENV="${CONDA_ENV:-alpharidge_ai}"

# ── resolve Python interpreter (conda > .venv > system) ───────────────────
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
elif command -v conda &>/dev/null && conda env list | grep -q "^${CONDA_ENV} "; then
    PYTHON="$(conda run -n "$CONDA_ENV" which python)"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
else
    echo "[run_server] ERROR: no Python found. Activate a venv or conda env first."
    exit 1
fi

echo "[run_server] Using Python : $PYTHON"
echo "[run_server] Repo root    : $REPO_ROOT"
echo "[run_server] Listening on : ${NLP_SERVER_HOST}:${NLP_SERVER_PORT}"

cd "$REPO_ROOT"

# ── tmux mode ──────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--tmux" ]]; then
    if ! command -v tmux &>/dev/null; then
        echo "[run_server] ERROR: tmux not found. Install it: apt install tmux"
        exit 1
    fi

    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "[run_server] Session '$TMUX_SESSION' already exists."
        echo "  Attach : tmux attach -t $TMUX_SESSION"
        echo "  Kill   : tmux kill-session -t $TMUX_SESSION"
        exit 0
    fi

    # Re-export so the new window inherits them
    export NLP_SERVER_HOST NLP_SERVER_PORT PYTHON REPO_ROOT

    tmux new-session -d -s "$TMUX_SESSION" \
        "cd '$REPO_ROOT' && CUBLAS_WORKSPACE_CONFIG=:4096:8 '$PYTHON' -m nlp_server.server; read"

    echo ""
    echo "[run_server] NLP server started in tmux session '$TMUX_SESSION'."
    echo ""
    echo "  Attach to watch logs : tmux attach -t $TMUX_SESSION"
    echo "  Detach again         : Ctrl-b  d"
    echo "  Kill server          : tmux kill-session -t $TMUX_SESSION"
    echo "  Health check         : curl http://localhost:${NLP_SERVER_PORT}/health"
    exit 0
fi

# ── foreground mode (default) ──────────────────────────────────────────────
exec "$PYTHON" -m nlp_server.server
