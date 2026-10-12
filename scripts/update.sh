#!/usr/bin/env bash
# Coordinated update script for SN45 nodes.
#
# Run this on EVERY machine (NLP server, each miner, validator) before
# restarting any of them.  The NLP server and validators must reach the
# SAME git commit before they start processing articles — otherwise their
# NER outputs will diverge and miners will be penalised for mismatches.
#
# Usage:
#   ./scripts/update.sh                    # pull + show what changed
#   ./scripts/update.sh --restart          # pull + restart miners (tmux) + NLP server
#   ./scripts/update.sh --restart-nlp      # pull + restart NLP server only
#
# Environment (set before calling or edit defaults below):
#   REPO_DIR   — path to the cloned repo (default: parent of this script)
#   VENV_PIP   — pip inside the venv (default: auto-detected from .venv or conda)
#   PYTHON     — python interpreter (default: auto-detected)

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_DIR"

# ── detect python + pip ──────────────────────────────────────────────────────
if [[ -x "$REPO_DIR/.venv/bin/python" ]]; then
    PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"
    VENV_PIP="${VENV_PIP:-$REPO_DIR/.venv/bin/pip}"
elif command -v conda &>/dev/null; then
    PYTHON="${PYTHON:-$(conda run -n alpharidge_ai which python 2>/dev/null || echo python3)}"
    VENV_PIP="${VENV_PIP:-$(conda run -n alpharidge_ai which pip 2>/dev/null || echo pip)}"
else
    PYTHON="${PYTHON:-python3}"
    VENV_PIP="${VENV_PIP:-pip}"
fi

# ── record current commit before pull (full hash) ───────────────────────────
BEFORE=$(git rev-parse HEAD)
BEFORE_SHORT=$(git rev-parse --short HEAD)

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  SN45 update — $(hostname)"
echo "  Repo : $REPO_DIR"
echo "  From : $BEFORE_SHORT"
echo "═══════════════════════════════════════════════════════"

# ── fetch + check remote ─────────────────────────────────────────────────────
git fetch origin

REMOTE=$(git rev-parse origin/main)
REMOTE_SHORT=$(git rev-parse --short origin/main)

if [[ "$BEFORE" == "$REMOTE" ]]; then
    echo "[update] Already up to date ($BEFORE_SHORT)."
    if [[ -z "${1:-}" ]]; then
        echo "[update] Nothing to do."
        exit 0
    fi
    echo "[update] Proceeding with ${1} anyway."
fi

# ── verify we are on main and can fast-forward ───────────────────────────────
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "main" ]]; then
    echo "[update] WARNING: current branch is '$CURRENT_BRANCH', not 'main'."
    echo "         Switching to main before pulling."
    git checkout main
fi

if [[ "$BEFORE" != "$REMOTE" ]]; then
    echo "[update] Pulling $BEFORE_SHORT → $REMOTE_SHORT …"
    git pull --ff-only origin main
fi

AFTER=$(git rev-parse HEAD)
AFTER_SHORT=$(git rev-parse --short HEAD)
echo "[update] Now at $AFTER_SHORT"

# ── show changed files ───────────────────────────────────────────────────────
echo ""
echo "── Files changed ──────────────────────────────────────"
git diff --name-only "$BEFORE" "$AFTER"
echo "────────────────────────────────────────────────────────"

# ── check consensus-critical paths (anchored match) ─────────────────────────
CONSENSUS_CHANGED=0
CONSENSUS_PATHS=(
    "alpharidge_ai/analyzer/ner_fusion.py"
    "alpharidge_ai/analyzer/aspect_sentiment.py"
    "alpharidge_ai/analyzer/determinism.py"
    "alpharidge_ai/analyzer/article_intelligence_analyzer.py"
    "alpharidge_ai/analyzer/data/"
)
CHANGED_FILES=$(git diff --name-only "$BEFORE" "$AFTER")
for p in "${CONSENSUS_PATHS[@]}"; do
    # Anchor match: either exact filename or path starts with the prefix
    if echo "$CHANGED_FILES" | grep -qE "(^|/)${p//\//\\/}"; then
        CONSENSUS_CHANGED=1
        echo "  ⚠️  CONSENSUS-CRITICAL change: $p"
    fi
done

if [[ $CONSENSUS_CHANGED -eq 1 ]]; then
    echo ""
    echo "  ‼  ALL nodes (NLP server + all miners + all validators)"
    echo "     must be updated to $AFTER_SHORT and restarted BEFORE any"
    echo "     of them starts processing articles again."
else
    echo ""
    echo "  ✓  No consensus-critical files changed — safe to"
    echo "     update nodes independently."
fi

# ── install new dependencies if requirements.txt changed ────────────────────
if echo "$CHANGED_FILES" | grep -q "requirements.txt"; then
    echo ""
    echo "[update] requirements.txt changed — running pip install …"
    "$VENV_PIP" install -q -r requirements.txt
    echo "[update] Dependencies updated."
fi

# ── restart ──────────────────────────────────────────────────────────────────
case "${1:-}" in

  --restart)
    # 1. Stop miners first so no requests hit the NLP server while it reloads
    echo ""
    echo "[update] Stopping miners …"
    "$PYTHON" scripts/mine.py stop

    # 2. Restart NLP server
    echo "[update] Restarting NLP server (tmux) …"
    tmux kill-session -t sn45-nlp 2>/dev/null || true
    sleep 2
    bash "$REPO_DIR/nlp_server/run_server.sh" --tmux

    # 3. Wait for NLP server to be healthy (up to 120 s)
    NLP_URL="${NLP_SERVER_URL:-http://localhost:8765}"
    echo "[update] Waiting for NLP server at $NLP_URL/health …"
    WAITED=0
    until curl -sf "$NLP_URL/health" | grep -q '"status":"ok"' 2>/dev/null; do
        if [[ $WAITED -ge 300 ]]; then
            echo "[update] WARNING: NLP server did not become healthy in 300 s."
            echo "         Miners will start anyway — check: curl $NLP_URL/health"
            break
        fi
        sleep 5
        WAITED=$((WAITED + 5))
        echo "[update]   … still loading (${WAITED}s)"
    done
    if [[ $WAITED -lt 300 ]]; then
        echo "[update] NLP server healthy after ${WAITED}s."
    fi

    # 4. Start miners
    echo "[update] Starting miners …"
    "$PYTHON" scripts/mine.py start
    echo "[update] All processes restarted."
    echo "[update] Monitor miners : $PYTHON scripts/mine.py status --watch 5"
    echo "[update] Monitor server : tmux attach -t sn45-nlp"
    ;;

  --restart-nlp)
    echo ""
    echo "[update] Restarting NLP server only …"
    tmux kill-session -t sn45-nlp 2>/dev/null || true
    sleep 2
    bash "$REPO_DIR/nlp_server/run_server.sh" --tmux
    echo "[update] NLP server restarted."
    ;;

  "")
    echo ""
    echo "[update] Code updated. Restart your processes when ready:"
    echo "    All miners + NLP server : ./scripts/update.sh --restart"
    echo "    NLP server only         : ./scripts/update.sh --restart-nlp"
    echo "    Miners only             : $PYTHON scripts/mine.py restart"
    echo "    Validator               : kill and re-run your validator process"
    ;;

  *)
    echo "[update] Unknown flag: ${1}"
    echo "Usage: $0 [--restart | --restart-nlp]"
    exit 1
    ;;
esac

echo ""
echo "[update] Commit: $AFTER_SHORT  host: $(hostname)  done ✓"
