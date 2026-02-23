#!/usr/bin/env bash
set -e

# ── Change to script directory ────────────────────────────────────────────────
cd "$(dirname "$0")"

# ── Tokyonight color theme (ANSI 256) ────────────────────────────────────────
C_BLUE="122"     # #7aa2f7 — primary accent
C_CYAN="117"     # #7dcfff — cyan accent
C_GREEN="149"    # #9ece6a — success
C_RED="204"      # #f7768e — error
C_TEXT="189"     # #c0caf5 — primary text
C_MUTED="60"     # #565f89 — muted text

# ── Bootstrap Gum ────────────────────────────────────────────────────────────
if ! command -v gum &>/dev/null; then
    if [[ -x ".gum_bin/gum" ]]; then
        export PATH="$PWD/.gum_bin:$PATH"
    elif command -v brew &>/dev/null; then
        echo "Installing gum via Homebrew..."
        brew install gum --quiet
    else
        echo "Downloading gum binary..."
        mkdir -p .gum_bin
        curl -fsSL "https://github.com/charmbracelet/gum/releases/download/v0.16.0/gum_0.16.0_Darwin_arm64.tar.gz" \
            | tar xz -C .gum_bin gum
        export PATH="$PWD/.gum_bin:$PATH"
    fi
fi

# ── Check for virtual environment ─────────────────────────────────────────────
if [[ ! -d ".venv" ]]; then
    gum style --foreground "$C_RED" --bold "Virtual environment not found. Run ./install.sh first."
    exit 1
fi

source .venv/bin/activate

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
gum style --border rounded --border-foreground "$C_BLUE" \
    --foreground "$C_TEXT" --bold --padding "1 4" --align center \
    "Qwen3-TTS MLX Studio"
echo ""
gum style --foreground "$C_MUTED" "  The UI will open in your browser at: $(gum style --foreground "$C_CYAN" --bold "http://localhost:7860")"
gum style --foreground "$C_MUTED" "  Press Ctrl+C to stop"
echo ""

python app.py "$@"
