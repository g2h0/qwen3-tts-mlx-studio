#!/usr/bin/env bash
set -e

# ── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[!!]${NC} $1"; }
fail() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }
info() { echo -e "${BOLD}>>>${NC} $1"; }

# ── Change to script directory ────────────────────────────────────────────────
cd "$(dirname "$0")"

echo ""
echo -e "${BOLD}╔══════════════════════════════════════╗${NC}"
echo -e "${BOLD}║     Qwen3-TTS Studio — Installer     ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════╝${NC}"
echo ""

# ── Preflight: macOS + Apple Silicon ──────────────────────────────────────────
info "Checking system requirements..."

if [[ "$(uname)" != "Darwin" ]]; then
    fail "This app requires macOS. Detected: $(uname)"
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
    fail "This app requires Apple Silicon (M1/M2/M3/M4). Detected: $ARCH"
fi
ok "macOS on Apple Silicon ($ARCH)"

# ── Preflight: Python 3.10+ ──────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    fail "Python 3 not found. Install it from https://www.python.org/downloads/"
fi

PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [[ "$PY_MAJOR" -lt 3 ]] || [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 10 ]]; then
    fail "Python 3.10+ required. Found: Python $PY_VERSION"
fi
ok "Python $PY_VERSION"

# ── Preflight: ffmpeg ─────────────────────────────────────────────────────────
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg found"
else
    warn "ffmpeg not found — required for audio processing."
    echo "   Install with Homebrew:  brew install ffmpeg"
    echo "   (If you don't have Homebrew: https://brew.sh)"
    echo ""
    read -rp "   Continue anyway? (y/N) " choice
    if [[ "$choice" != "y" && "$choice" != "Y" ]]; then
        echo "Install ffmpeg first, then re-run this script."
        exit 1
    fi
fi

# ── Create virtual environment ────────────────────────────────────────────────
info "Setting up Python virtual environment..."

if [[ -d ".venv" ]]; then
    ok "Virtual environment already exists (.venv/)"
else
    python3 -m venv .venv
    ok "Created virtual environment (.venv/)"
fi

source .venv/bin/activate

# ── Install dependencies ──────────────────────────────────────────────────────
info "Installing Python packages (this may take a few minutes)..."

pip install --upgrade pip -q
pip install -U -r requirements.txt -q

ok "All packages installed"

# ── Create directories ────────────────────────────────────────────────────────
mkdir -p outputs/history voices
ok "Created output directories"

# ── Optional: Pre-download models ─────────────────────────────────────────────
echo ""
info "Model download (optional)"
echo "   The TTS models (~6 GB total) can be downloaded now, or they'll"
echo "   download automatically when you first use each voice mode."
echo ""
read -rp "   Download models now? (y/N) " dl_choice

if [[ "$dl_choice" == "y" || "$dl_choice" == "Y" ]]; then
    info "Downloading models (this will take a while)..."
    pip install -q huggingface_hub

    echo "   [1/3] CustomVoice model..."
    huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit --quiet
    ok "CustomVoice model downloaded"

    echo "   [2/3] VoiceDesign model..."
    huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit --quiet
    ok "VoiceDesign model downloaded"

    echo "   [3/3] Base model (for Voice Cloning)..."
    huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit --quiet
    ok "Base model downloaded"
else
    ok "Skipped — models will download on first use"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}Installation complete!${NC}"
echo ""
echo "   To start the app, run:"
echo ""
echo -e "   ${BOLD}./run.sh${NC}"
echo ""
