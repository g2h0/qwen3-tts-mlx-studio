# Qwen3-TTS Studio

Local text-to-speech on Apple Silicon, powered by [Qwen3-TTS](https://huggingface.co/Qwen) and [mlx-audio](https://github.com/Blaizzy/mlx-audio). Runs entirely on-device — no API keys, no internet required after the initial model download.

## Requirements

- macOS with Apple Silicon (M1 or later)
- Python 3.10+ (3.12 recommended)
- [Homebrew](https://brew.sh)

## Setup

```bash
./install.sh
```

This creates a `.venv`, installs dependencies, and optionally pre-downloads the TTS models (~6 GB total). Models also auto-download on first use if you skip that step.

## Usage

```bash
./run.sh
```

Opens the UI at `http://localhost:7860`.

```bash
./run.sh --model-size 0.6B   # Smaller, faster model (default: 1.7B)
./run.sh --quant bf16        # Full precision (default: 8bit)
./run.sh --host 0.0.0.0     # Listen on all interfaces
./run.sh --port 8080         # Custom port
./run.sh --share             # Public Gradio link
```

## Tabs

**Custom Voice** — Generate speech with one of nine built-in speaker presets. Add a style instruction to shape tone, pace, and emotion. Supports batch generation for long texts.

**Voice Design** — Describe a voice in plain language and the model creates it. Designed voices can be saved to the Voice Library. Supports batch generation.

**Voice Cloning** — Clone a voice from a short reference audio clip (3–30 seconds). Provide the clip and an exact transcript of what was spoken. Supports batch generation.

**YT Voice Clone** — Clone a voice directly from a YouTube video. Paste a URL, select a timestamp range, and the transcript auto-fills from subtitles. Clips are cached in `.yt_cache/`.

**Script Mode** — Write multi-speaker scripts with `SPEAKER: Dialogue` formatting and assign a different voice to each speaker. Lines are batched by model type to minimise swaps, then stitched together with configurable silence gaps.

**Voice Library** — Browse, preview, rename, delete, and import saved voices. Voices from Voice Design, Voice Cloning, and YT Voice Clone all appear here.

**History** — Every generation is logged with mode, language, text, and duration. Replay audio, save WAV files, or view original parameters.

**Settings** — Model size and quantization, generation parameters (temperature, top-k, top-p, repetition penalty, max tokens, timeout), output directory, auto-save toggle, JIT compilation toggle, YT cache management, and model cache management (view/delete downloaded models).

## Models

Three model variants, one per generation mode. Only one is loaded at a time (~6 GB for a 1.7B 8-bit model).

| Mode | Variant | Default Repo |
|---|---|---|
| Custom Voice | CustomVoice | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` |
| Voice Design | VoiceDesign | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit` |
| Voice Cloning | Base | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit` |

To pre-download manually:

```bash
source .venv/bin/activate
huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit
huggingface-cli download mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit
```

## Output

Audio is 24 kHz mono WAV (32-bit float), saved to `./outputs/` by default. Voice library profiles are stored in `./voices/`.

## Supported Languages

English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

## Project Layout

```
app.py            — Gradio UI, tabs, and event wiring
engine.py         — Model load/unload/inference (thread-safe)
voice_library.py  — Voice profile storage
yt_voice.py       — YouTube clip extraction and subtitle alignment
audio_utils.py    — Audio concatenation and text splitting
script_parser.py  — Multi-speaker script parser
history.py        — Generation history log
config.py         — Constants and defaults
theme.py          — Dark theme and custom CSS
```

## License

[MIT](LICENSE)
