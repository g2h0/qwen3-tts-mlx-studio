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

# Returns 0 and prints "MAJOR.MINOR" if the given python binary is 3.10–3.13
check_python_compat() {
    local py_bin="$1"
    if [[ ! -x "$py_bin" ]] && ! command -v "$py_bin" &>/dev/null; then
        return 1
    fi
    local ver major minor
    ver=$("$py_bin" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null) || return 1
    major=$(echo "$ver" | cut -d. -f1)
    minor=$(echo "$ver" | cut -d. -f2)
    if [[ "$major" -eq 3 && "$minor" -ge 10 && "$minor" -le 13 ]]; then
        echo "$ver"
        return 0
    fi
    return 1
}

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

# ── Preflight: Resolve compatible Python (3.10–3.13) ────────────────────────
info "Checking for compatible Python (3.10–3.13)..."

PYTHON_CMD=""
PY_VERSION=""

# Step 1: Try system python3
if ver=$(check_python_compat python3); then
    PYTHON_CMD="python3"
    PY_VERSION="$ver"
fi

# Step 2: Try versioned binaries on PATH (prefer 3.12 for best wheel support)
if [[ -z "$PYTHON_CMD" ]]; then
    for minor in 12 13 11 10; do
        if ver=$(check_python_compat "python3.${minor}"); then
            PYTHON_CMD="python3.${minor}"
            PY_VERSION="$ver"
            break
        fi
    done
fi

# Step 3: Try Homebrew prefix explicitly (handles PATH not including brew bin)
if [[ -z "$PYTHON_CMD" ]] && command -v brew &>/dev/null; then
    BREW_PREFIX="$(brew --prefix)"
    for minor in 12 13 11 10; do
        local_bin="${BREW_PREFIX}/bin/python3.${minor}"
        if ver=$(check_python_compat "$local_bin"); then
            PYTHON_CMD="$local_bin"
            PY_VERSION="$ver"
            break
        fi
    done
fi

# Step 4: Offer to brew install python@3.12
if [[ -z "$PYTHON_CMD" ]]; then
    if command -v python3 &>/dev/null; then
        sys_ver=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "unknown")
        warn "System python3 is $sys_ver — outside compatible range (3.10–3.13)."
    else
        warn "No python3 found on PATH."
    fi

    if command -v brew &>/dev/null; then
        echo ""
        echo "   Python 3.12 is recommended (best compatibility with MLX ecosystem)."
        read -rp "   Install python@3.12 via Homebrew? (Y/n) " brew_choice
        if [[ "$brew_choice" != "n" && "$brew_choice" != "N" ]]; then
            info "Installing python@3.12 via Homebrew..."
            brew install python@3.12
            BREW_PREFIX="$(brew --prefix)"
            PYTHON_CMD="${BREW_PREFIX}/bin/python3.12"
            if ver=$(check_python_compat "$PYTHON_CMD"); then
                PY_VERSION="$ver"
            else
                fail "Homebrew installed python@3.12 but it failed validation."
            fi
        else
            fail "No compatible Python available. Install manually: brew install python@3.12"
        fi
    else
        echo ""
        echo "   No compatible Python found and Homebrew is not installed."
        echo ""
        echo "   Option 1: Install Homebrew (https://brew.sh), then:"
        echo "             brew install python@3.12"
        echo ""
        echo "   Option 2: Download Python 3.12 from:"
        echo "             https://www.python.org/downloads/"
        echo ""
        fail "Cannot continue without Python 3.10–3.13."
    fi
fi

ok "Using Python $PY_VERSION ($PYTHON_CMD)"

# ── Preflight: ffmpeg ─────────────────────────────────────────────────────────
if command -v ffmpeg &>/dev/null; then
    ok "ffmpeg found"
else
    warn "ffmpeg not found — required for audio processing."
    if command -v brew &>/dev/null; then
        read -rp "   Install ffmpeg via Homebrew? (Y/n) " ff_choice
        if [[ "$ff_choice" != "n" && "$ff_choice" != "N" ]]; then
            info "Installing ffmpeg via Homebrew..."
            brew install ffmpeg
            ok "ffmpeg installed"
        else
            warn "Skipping ffmpeg — audio processing may not work."
        fi
    else
        echo "   Install with Homebrew:  brew install ffmpeg"
        echo "   (If you don't have Homebrew: https://brew.sh)"
        echo ""
        read -rp "   Continue without ffmpeg? (y/N) " ff_choice
        if [[ "$ff_choice" != "y" && "$ff_choice" != "Y" ]]; then
            exit 1
        fi
    fi
fi

# ── Create virtual environment ────────────────────────────────────────────────
info "Setting up Python virtual environment..."

if [[ -d ".venv" ]]; then
    # Verify existing venv uses a compatible Python
    if ver=$(check_python_compat .venv/bin/python3); then
        ok "Virtual environment already exists (.venv/) — Python $ver"
    else
        VENV_PY_VER=$(.venv/bin/python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "unknown")
        warn "Existing .venv uses Python $VENV_PY_VER (incompatible). Recreating..."
        rm -rf .venv
        "$PYTHON_CMD" -m venv .venv
        ok "Recreated virtual environment with Python $PY_VERSION"
    fi
else
    "$PYTHON_CMD" -m venv .venv
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
