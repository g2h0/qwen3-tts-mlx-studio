#!/usr/bin/env bash
set -e

# ── Change to script directory ────────────────────────────────────────────────
cd "$(dirname "$0")"

# ── Check for virtual environment ─────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
    echo "Virtual environment not found. Run ./install.sh first."
    exit 1
fi

source .venv/bin/activate

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
echo "Starting Qwen3-TTS Studio..."
echo "The UI will open at: http://localhost:7860"
echo "(Press Ctrl+C to stop)"
echo ""

python app.py "$@"
