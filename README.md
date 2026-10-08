# Qwen3-TTS MLX Studio

[![tests](https://github.com/g2h0/qwen3-tts-mlx-studio/actions/workflows/tests.yml/badge.svg)](https://github.com/g2h0/qwen3-tts-mlx-studio/actions/workflows/tests.yml)

Local text-to-speech and speech recognition on Apple Silicon, powered by [Qwen3-TTS](https://huggingface.co/Qwen) and [mlx-audio](https://github.com/Blaizzy/mlx-audio). Runs entirely on-device — no API keys, no internet required after the initial model download.

![Qwen3-TTS MLX Studio](assets/screenshot.png)

## Scope

A local, single-user desktop studio for Qwen3-TTS on Apple Silicon. Deliberately out of scope:

- **Server or API deployment** — there is no OpenAI-compatible endpoint and no multi-user serving. This is a Gradio app bound to localhost.
- **Long-form audiobook production** — EPUB/PDF ingestion, chaptering, and whole-book character casting are not planned. Script Mode covers short multi-speaker passages.
- **Non-Apple-Silicon platforms** — mlx is Apple Silicon only by design.
- **Training or fine-tuning** — inference only.

Feature requests outside this scope will usually be closed. Forks are welcome.

## Requirements

- A Mac with Apple Silicon (M1, M2, M3, M4, or M5)
- Python 3.10–3.13 (3.12 recommended — best wheel support across the MLX stack)
- [Homebrew](https://brew.sh) (the installer uses it to set up dependencies)

If you don't have Homebrew yet, open Terminal and paste:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## Tabs

**Custom Voice** — Generate speech with one of nine built-in voices. Add style instructions to shape tone, pace, and emotion. Supports batch generation for long texts.

**Voice Design** — Describe a voice in plain language and the model creates it. Designed voices can be saved to the Voice Library. Supports batch generation.

**Voice Clone** — Clone a voice from a short reference audio clip (3-30 seconds). Provide the clip and a reference transcript of exactly what was spoken — the "Transcribe Reference" button can auto-fill it using on-device speech recognition. Reference clips are auto-trimmed of leading/trailing silence (toggleable) and can optionally be denoised. Batch generation runs as true batched inference sharing the reference.

**YT Voice Clone** — Clone a voice directly from a YouTube video. Paste a URL, select a timestamp range, and the transcript auto-fills from subtitles. A "Transcribe Clip" button is available when subtitles are missing or inaccurate. Clips are cached in `.yt_cache/`.

**Script Mode** — Write multi-speaker scripts with `SPEAKER: Dialogue` formatting and assign a different voice to each speaker. Lines are batched by voice type to minimise model swaps (cloned speakers batch their lines against a shared reference), then stitched together with configurable silence gaps.

**Transcription** — Transcribe audio files locally. Upload a file or record with your microphone, pick a language (or leave it on Auto-detect), and watch the transcript stream in live. Supports up to ~20 minutes of audio. Transcripts can be saved as `.txt` files.

**Voice Library** — Browse, preview, rename, delete, and import saved voices. Voices from Voice Design, Voice Clone, and YT Voice Clone all appear here.

**History** — Every completed generation is logged with mode, language, text, and duration. Replay audio, save files, or view the generation settings that produced a take.

**Settings** — Model size and quantization; generation controls with plain-language tooltips (temperature, top-k, top-p, repetition penalty, max length, auto-stop timeout); batch size; output folder and auto-save; export format (WAV/MP3/OGG) with MP3 bitrate; post-processing (normalize loudness, trim silence, reduce reference background noise); "speed up repeat runs" compilation toggle; speech-recognition, YT-cache, and model-cache management (view/delete downloaded models).

## While it generates

Generation is never a black box:

- A live status shows seconds of audio generated as it runs ("Generating… 12.4s"); batch runs show "Segment k/N", Script Mode shows "Line k/N".
- Every generating tab has a **Stop** button. Stopping keeps the audio generated so far in the player — you can listen to it and save it manually, but partial takes are not auto-saved and don't enter History.
- The **auto-stop timeout** (Settings) ends a runaway generation the same way, keeping the partial audio. Number-heavy text (dates, scientific notation, acronym lists) is the usual culprit for runaways.
- Transcription streams its text live on all three entry points (Transcription tab, "Transcribe Reference", "Transcribe Clip"), with the same Stop support.

## Setup

**1. Clone the repository**

Open Terminal and run:

```bash
git clone https://github.com/g2h0/qwen3-tts-mlx-studio.git
cd qwen3-tts-mlx-studio
```

**2. Run the installer**

```bash
./install.sh
```

The installer will:

- Check that you're on Apple Silicon with a compatible Python version (and offer to install one via Homebrew if needed)
- Install ffmpeg if it's missing (required for audio processing)
- Create a Python virtual environment (`.venv/`)
- Install all Python dependencies
- Optionally pre-download the TTS models (size varies by quantization) — you can skip this and they'll download automatically the first time you use each mode

## Usage

Start the app:

```bash
./run.sh
```

The UI opens in your browser at `http://localhost:7860`. Press Ctrl+C in the terminal to stop.

**Options:**

```bash
./run.sh --model-size 0.6B   # Smaller, faster model (default: 1.7B)
./run.sh --quant 4bit        # 4-bit quantization — smallest footprint
./run.sh --quant 8bit        # 8-bit quantization (default: bf16)
./run.sh --host 0.0.0.0      # Listen on all interfaces (e.g. access from another device)
./run.sh --port 8080          # Custom port
./run.sh --share              # Create a public Gradio link
```

## Models

### TTS

Three model variants, one per generation mode. Only one is loaded at a time (~6 GB for 1.7B bf16).

| Mode | Variant | Default Repo |
|---|---|---|
| Custom Voice | CustomVoice | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16` |
| Voice Design | VoiceDesign | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` |
| Voice Clone | Base | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` |

All three variants are available in `bf16`, `8bit`, `6bit`, and `4bit` quantizations. Select in Settings or via `--quant`.

### ASR (Speech Recognition)

A single model used for transcription across all tabs. It loads on demand and unloads automatically after each transcription to free memory.

| Model | Repo |
|---|---|
| Qwen3-ASR 1.7B | `mlx-community/Qwen3-ASR-1.7B-8bit` |

TTS and ASR models are mutually exclusive — only one can be in memory at a time. Switching between them is handled automatically.

To pre-download models manually:

```bash
source .venv/bin/activate
huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16
huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16
huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16
huggingface-cli download mlx-community/Qwen3-ASR-1.7B-8bit
```

Replace `bf16` with `8bit`, `6bit`, or `4bit` for smaller models.

## Output

Audio is generated at 24 kHz mono and saved to `./outputs/` by default. Supported export formats are WAV (32-bit float), MP3 (configurable bitrate), and OGG/Vorbis — selectable in Settings. Optional post-processing includes loudness normalization (EBU R128) and leading/trailing silence trimming (both via ffmpeg). Reference audio for Voice Clone can optionally be denoised with DeepFilterNet (toggle in Settings). Transcripts are saved as `.txt` files in the same directory. Voice library profiles are stored in `./voices/`.

## Supported Languages

Auto-detect (default), or explicitly: English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

## Project Layout

```
app.py            — Entrypoint: CLI args, startup checks, UI assembly
generation.py     — Shared pipeline (validation, live progress, Stop/auto-stop, history, autosave)
engine.py         — Model load/unload/inference (thread-safe, TTS + ASR)
state.py          — Runtime settings (AppSettings) and app context
ui/
  strings.py      — All user-facing text
  components.py   — Shared UI builders and table formatters
  tabs/           — One module per tab (layout + event wiring)
voice_library.py  — Voice profile storage
yt_voice.py       — YouTube clip extraction and subtitle alignment
audio_utils.py    — Audio concatenation, text splitting, and format export
script_parser.py  — Multi-speaker script parser
history.py        — Generation history log
config.py         — Constants and defaults
theme.py          — Dark theme and custom CSS
tests/            — Unit tests (engine faked — no models, no Apple Silicon needed)
scripts/          — Smoke test and launch check
.github/          — CI workflow and issue templates
requirements.txt  — Runtime pins (load-bearing — see Development)
install.sh        — One-step installer
uninstall.sh      — Remove venv, caches, and downloaded models
run.sh            — App launcher
```

## Development

```bash
.venv/bin/python -m pytest tests/ -q       # unit tests — ~2s, loads no models
./scripts/launch_check.sh                  # boots the app, then shuts it down
.venv/bin/python scripts/smoke_test.py     # real models end to end, ~5 min
```

The unit suite fakes the engine, so it needs neither Apple Silicon nor a model
download. CI runs it on every push and pull request, alongside a job that
verifies the dependency pins still resolve on a clean Apple Silicon machine.

The pins in `requirements.txt` are load-bearing rather than cosmetic: `mlx`
excludes 0.31.2 because it segfaults batched generation, and `mlx-audio` floors
at 0.4.8 because 0.4.6 and 0.4.7 regress ASR audio loading. Run the smoke test
before raising either — CI can prove they install, not that they work.

## Troubleshooting

**"Virtual environment not found"** — Run `./install.sh` first before `./run.sh`.

**Model download is slow** — The first run downloads models from HuggingFace (~6 GB for bf16, less for quantized). On a slow connection you can pre-download models (see the Models section above) or let the installer do it.

**Out of memory** — The 1.7B bf16 model uses ~6 GB of unified memory. If you're running low, try `./run.sh --model-size 0.6B` or `./run.sh --quant 4bit` for the smallest footprint.

**ffmpeg not found** — Install it with `brew install ffmpeg`. It's required for audio processing and YouTube clip extraction.

**YouTube clip extraction fails with "403 Forbidden"** — YouTube changes its download protections regularly, and older yt-dlp versions stop working. Upgrade it with `.venv/bin/pip install -U "yt-dlp[default]"`. Installing [Deno](https://deno.com) (`brew install deno`) is also recommended; yt-dlp uses it to solve YouTube's JavaScript challenges.

**Generation runs away on number-heavy text** — Dates, scientific notation, and acronym lists can make the model keep talking far past the end of your text. Press **Stop**; it keeps the audio generated so far. The auto-stop timeout in Settings does the same automatically, and lowering "Max length (tokens)" caps the damage entirely.

## License

[MIT](LICENSE)
