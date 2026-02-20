import argparse
import os
import concurrent.futures
import shutil
import sys
import time
from datetime import datetime

import gradio as gr
import numpy as np
import soundfile as sf

from audio_utils import concatenate_audio, split_text
from config import (
    DEFAULT_AUTOSAVE,
    DEFAULT_BATCH_SPLIT_MODE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_SIZE,
    DEFAULT_QUANTIZATION,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_SCRIPT_SILENCE_MS,
    DEFAULT_SILENCE_GAP_MS,
    DEFAULT_SPEAKERS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    ENABLE_JIT_COMPILE,
    HISTORY_DIR,
    LANGUAGES,
    MAX_BATCH_SEGMENTS,
    MAX_SCRIPT_SPEAKERS,
    OUTPUT_DIR,
    SERVER_HOST,
    SERVER_PORT,
    VOICE_LIBRARY_DIR,
    YT_CACHE_DIR,
)
from engine import TTSEngine
from history import GenerationHistory
from script_parser import parse_script, group_by_model_type
from theme import build_theme, custom_css
from voice_library import VoiceLibrary
from yt_voice import get_yt_extractor

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Qwen3-TTS Studio")
parser.add_argument("--host", default=SERVER_HOST, help="Server host")
parser.add_argument("--port", type=int, default=SERVER_PORT, help="Server port")
parser.add_argument(
    "--model-size", choices=["0.6B", "1.7B"], default=DEFAULT_MODEL_SIZE
)
parser.add_argument(
    "--quant", choices=["8bit", "bf16"], default=DEFAULT_QUANTIZATION
)
parser.add_argument(
    "--share", action="store_true", help="Create public Gradio link"
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Startup validation
# ---------------------------------------------------------------------------
def check_startup():
    warnings = []
    if sys.version_info < (3, 10):
        warnings.append("Python 3.10+ required")
    try:
        import mlx_audio  # noqa: F401
    except ImportError:
        warnings.append("mlx-audio not installed — run: pip install mlx-audio")
    if not shutil.which("ffmpeg"):
        warnings.append("ffmpeg not found — run: brew install ffmpeg")
    try:
        # Skip ffmpeg — already checked above
        yt_missing = [w for w in get_yt_extractor().check_dependencies()
                      if "ffmpeg" not in w]
        warnings.extend(yt_missing)
    except Exception as e:
        warnings.append(f"YT Voice Clone init error: {e}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if warnings:
        for w in warnings:
            print(f"WARNING: {w}")
    return warnings

startup_warnings = check_startup()

# ---------------------------------------------------------------------------
# Engine, library & history
# ---------------------------------------------------------------------------
engine = TTSEngine()
engine.model_size = args.model_size
engine.quantization = args.quant

library = VoiceLibrary()
history = GenerationHistory()
yt_extractor = get_yt_extractor()

# ---------------------------------------------------------------------------
# Runtime settings (mutated by Settings tab)
# ---------------------------------------------------------------------------
app_settings = {
    "temperature": DEFAULT_TEMPERATURE,
    "top_k": DEFAULT_TOP_K,
    "top_p": DEFAULT_TOP_P,
    "repetition_penalty": DEFAULT_REPETITION_PENALTY,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "timeout": DEFAULT_TIMEOUT,
    "output_dir": OUTPUT_DIR,
    "autosave": DEFAULT_AUTOSAVE,
    "default_language": "English",
}

# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------
class GenerationTimeout(Exception):
    pass


def generate_with_timeout(func, *func_args, timeout_seconds=120, **func_kwargs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(func, *func_args, **func_kwargs)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            raise GenerationTimeout(
                "Generation timed out — try shorter text or lower max_new_tokens"
            )


# ---------------------------------------------------------------------------
# Handler helpers
# ---------------------------------------------------------------------------
def _gen_kwargs():
    """Build kwargs dict for engine generate calls from current settings."""
    return {
        "temperature": app_settings["temperature"],
        "top_k": app_settings["top_k"],
        "top_p": app_settings["top_p"],
        "repetition_penalty": app_settings["repetition_penalty"],
        "max_tokens": app_settings["max_tokens"],
    }


def save_audio(audio_tuple, prefix="output"):
    """Save generated audio to the configured output directory."""
    if audio_tuple is None:
        gr.Warning("No audio to save.")
        return "No audio to save"
    sr, audio = audio_tuple
    out_dir = app_settings["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"{prefix}_{timestamp}.wav")
    sf.write(path, audio, sr)
    return f"Saved: {path}"


def _get_hf_cache_dir() -> str:
    """Return the HuggingFace hub cache directory path."""
    return (
        os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    )


def _is_model_cached(repo_id: str) -> bool:
    """Check whether a HuggingFace model repo is already in the local cache."""
    hf_home = _get_hf_cache_dir()
    cache_name = "models--" + repo_id.replace("/", "--")
    snapshots = os.path.join(hf_home, cache_name, "snapshots")
    return os.path.isdir(snapshots) and bool(os.listdir(snapshots))


def _voice_choices():
    """Return list of saved voice names for dropdowns."""
    return ["None"] + [v["name"] for v in library.list_voices()]


def _voice_table():
    """Return voice data for the library dataframe."""
    voices = library.list_voices()
    if not voices:
        return [["(empty)", "", "", ""]]
    return [
        [v["name"], v.get("source", ""), v.get("language", ""), v.get("description", "")]
        for v in voices
    ]


# ---------------------------------------------------------------------------
# Generation handlers
# ---------------------------------------------------------------------------
def generate_custom_voice(text, speaker, language, instruct):
    if not text.strip():
        gr.Warning("Please enter text to speak.")
        yield None, "Enter text first"
        return
    if not engine.is_model_loaded("custom_voice"):
        repo_id = engine.get_repo_id("custom_voice")
        if _is_model_cached(repo_id):
            yield None, f"Loading model into memory… ({repo_id})"
        else:
            yield None, (
                f"Downloading model on first run (~6 GB) — this may take several minutes… ({repo_id})"
            )
    try:
        start = time.time()
        sr, audio = generate_with_timeout(
            engine.generate_custom_voice, text, speaker, language, instruct,
            timeout_seconds=app_settings["timeout"],
            **_gen_kwargs(),
        )
        elapsed = time.time() - start
        result = (sr, audio)
        # Record to history
        history.add(
            mode="custom_voice", text=text, language=language,
            audio=result, speaker=speaker,
            voice_params=instruct if instruct else "",
        )
        save_msg = ""
        if app_settings["autosave"]:
            save_msg = " | " + save_audio(result, "custom")
        yield (
            gr.update(value=result),
            f"Generated in {elapsed:.1f}s | Model: {engine.get_repo_id('custom_voice')}{save_msg}",
        )
    except GenerationTimeout as e:
        gr.Warning(str(e))
        yield None, str(e)
    except Exception as e:
        gr.Warning(f"Generation failed: {e}")
        yield None, f"Error: {e}"


def generate_voice_design(text, language, instruct):
    if not text.strip():
        gr.Warning("Please enter text to speak.")
        yield None, "Enter text first"
        return
    if not instruct.strip():
        gr.Warning("Please describe the voice you want.")
        yield None, "Describe the voice first"
        return
    if not engine.is_model_loaded("voice_design"):
        repo_id = engine.get_repo_id("voice_design")
        if _is_model_cached(repo_id):
            yield None, f"Loading model into memory… ({repo_id})"
        else:
            yield None, (
                f"Downloading model on first run (~6 GB) — this may take several minutes… ({repo_id})"
            )
    try:
        start = time.time()
        sr, audio = generate_with_timeout(
            engine.generate_voice_design, text, language, instruct,
            timeout_seconds=app_settings["timeout"],
            **_gen_kwargs(),
        )
        elapsed = time.time() - start
        result = (sr, audio)
        # Record to history
        history.add(
            mode="voice_design", text=text, language=language,
            audio=result, voice_params=instruct,
        )
        save_msg = ""
        if app_settings["autosave"]:
            save_msg = " | " + save_audio(result, "design")
        yield (
            gr.update(value=result),
            f"Generated in {elapsed:.1f}s | Model: {engine.get_repo_id('voice_design')}{save_msg}",
        )
    except GenerationTimeout as e:
        gr.Warning(str(e))
        yield None, str(e)
    except Exception as e:
        gr.Warning(f"Generation failed: {e}")
        yield None, f"Error: {e}"


def generate_voice_clone(text, ref_audio, ref_text, language, library_voice):
    if not text.strip():
        gr.Warning("Please enter text to speak.")
        yield None, "Enter text first"
        return
    # If library voice selected, override ref_audio/ref_text
    if library_voice and library_voice != "None":
        try:
            voice = library.load_voice(library_voice)
            ref_audio = library.get_ref_audio_path(library_voice)
            ref_text = voice["ref_text"]
        except FileNotFoundError:
            gr.Warning(f"Voice '{library_voice}' not found in library.")
            yield None, "Voice not found"
            return
    if not ref_audio:
        gr.Warning("Please upload reference audio or select from library.")
        yield None, "No reference audio"
        return
    if not ref_text or not ref_text.strip():
        gr.Warning("Reference transcript is required for voice cloning.")
        yield None, "No reference transcript"
        return
    if not engine.is_model_loaded("base"):
        repo_id = engine.get_repo_id("base")
        if _is_model_cached(repo_id):
            yield None, f"Loading model into memory… ({repo_id})"
        else:
            yield None, (
                f"Downloading model on first run (~6 GB) — this may take several minutes… ({repo_id})"
            )
    try:
        start = time.time()
        sr, audio = generate_with_timeout(
            engine.generate_voice_clone, text, ref_audio, ref_text, language,
            timeout_seconds=app_settings["timeout"],
            **_gen_kwargs(),
        )
        elapsed = time.time() - start
        result = (sr, audio)
        # Record to history
        voice_info = library_voice if library_voice and library_voice != "None" else "uploaded"
        history.add(
            mode="voice_clone", text=text, language=language,
            audio=result, voice_params=f"ref: {voice_info}",
        )
        save_msg = ""
        if app_settings["autosave"]:
            save_msg = " | " + save_audio(result, "clone")
        yield (
            gr.update(value=result),
            f"Generated in {elapsed:.1f}s | Model: {engine.get_repo_id('base')}{save_msg}",
        )
    except GenerationTimeout as e:
        gr.Warning(str(e))
        yield None, str(e)
    except Exception as e:
        gr.Warning(f"Generation failed: {e}")
        yield None, f"Error: {e}"


# ---------------------------------------------------------------------------
# Batch generation handlers
# ---------------------------------------------------------------------------
def _run_batch_custom_voice(text, speaker, language, instruct, split_mode, silence_ms, progress=gr.Progress()):
    segments = split_text(text, split_mode)
    if not segments:
        gr.Warning("No text segments found.")
        return None, [["(empty)", "", ""]], "No segments"
    if len(segments) > MAX_BATCH_SEGMENTS:
        gr.Warning(f"Too many segments ({len(segments)}). Max is {MAX_BATCH_SEGMENTS}.")
        return None, [["(error)", "", ""]], f"Too many segments (max {MAX_BATCH_SEGMENTS})"

    audio_parts = []
    table_rows = []
    succeeded, failed = 0, 0

    for i, seg in enumerate(progress.tqdm(segments, desc="Generating segments")):
        preview = seg[:50] + "..." if len(seg) > 50 else seg
        try:
            sr, audio = generate_with_timeout(
                engine.generate_custom_voice, seg, speaker, language, instruct,
                timeout_seconds=app_settings["timeout"],
                **_gen_kwargs(),
            )
            duration = len(audio) / sr
            audio_parts.append((sr, audio))
            table_rows.append([str(i + 1), preview, f"{duration:.1f}s"])
            succeeded += 1
        except Exception as e:
            table_rows.append([str(i + 1), preview, f"Failed: {e}"])
            failed += 1

    if not audio_parts:
        return None, table_rows, "All segments failed"

    combined = concatenate_audio(audio_parts, silence_ms=int(silence_ms))
    # Record combined to history
    history.add(
        mode="custom_voice", text=f"[Batch: {succeeded} segments]",
        language=language, audio=combined, speaker=speaker,
        voice_params=f"batch ({split_mode})",
    )
    status_msg = f"Generated {succeeded}/{len(segments)} segments"
    if failed:
        status_msg += f" ({failed} failed)"
    return gr.update(value=combined), table_rows, status_msg


def _run_batch_voice_design(text, language, instruct, split_mode, silence_ms, progress=gr.Progress()):
    if not instruct.strip():
        gr.Warning("Please describe the voice.")
        return None, [["(empty)", "", ""]], "Describe voice first"
    segments = split_text(text, split_mode)
    if not segments:
        gr.Warning("No text segments found.")
        return None, [["(empty)", "", ""]], "No segments"
    if len(segments) > MAX_BATCH_SEGMENTS:
        gr.Warning(f"Too many segments ({len(segments)}). Max is {MAX_BATCH_SEGMENTS}.")
        return None, [["(error)", "", ""]], f"Too many segments (max {MAX_BATCH_SEGMENTS})"

    audio_parts = []
    table_rows = []
    succeeded, failed = 0, 0

    for i, seg in enumerate(progress.tqdm(segments, desc="Generating segments")):
        preview = seg[:50] + "..." if len(seg) > 50 else seg
        try:
            sr, audio = generate_with_timeout(
                engine.generate_voice_design, seg, language, instruct,
                timeout_seconds=app_settings["timeout"],
                **_gen_kwargs(),
            )
            duration = len(audio) / sr
            audio_parts.append((sr, audio))
            table_rows.append([str(i + 1), preview, f"{duration:.1f}s"])
            succeeded += 1
        except Exception as e:
            table_rows.append([str(i + 1), preview, f"Failed: {e}"])
            failed += 1

    if not audio_parts:
        return None, table_rows, "All segments failed"

    combined = concatenate_audio(audio_parts, silence_ms=int(silence_ms))
    history.add(
        mode="voice_design", text=f"[Batch: {succeeded} segments]",
        language=language, audio=combined, voice_params=f"batch ({split_mode})",
    )
    status_msg = f"Generated {succeeded}/{len(segments)} segments"
    if failed:
        status_msg += f" ({failed} failed)"
    return gr.update(value=combined), table_rows, status_msg


def _run_batch_voice_clone(text, ref_audio, ref_text, language, library_voice, split_mode, silence_ms, progress=gr.Progress()):
    # Resolve library voice
    if library_voice and library_voice != "None":
        try:
            voice = library.load_voice(library_voice)
            ref_audio = library.get_ref_audio_path(library_voice)
            ref_text = voice["ref_text"]
        except FileNotFoundError:
            gr.Warning(f"Voice '{library_voice}' not found.")
            return None, [["(error)", "", ""]], "Voice not found"
    if not ref_audio:
        gr.Warning("No reference audio.")
        return None, [["(error)", "", ""]], "No reference audio"
    if not ref_text or not ref_text.strip():
        gr.Warning("Reference transcript required.")
        return None, [["(error)", "", ""]], "No transcript"

    segments = split_text(text, split_mode)
    if not segments:
        gr.Warning("No text segments found.")
        return None, [["(empty)", "", ""]], "No segments"
    if len(segments) > MAX_BATCH_SEGMENTS:
        gr.Warning(f"Too many segments ({len(segments)}). Max is {MAX_BATCH_SEGMENTS}.")
        return None, [["(error)", "", ""]], f"Too many segments (max {MAX_BATCH_SEGMENTS})"

    audio_parts = []
    table_rows = []
    succeeded, failed = 0, 0

    for i, seg in enumerate(progress.tqdm(segments, desc="Generating segments")):
        preview = seg[:50] + "..." if len(seg) > 50 else seg
        try:
            sr, audio = generate_with_timeout(
                engine.generate_voice_clone, seg, ref_audio, ref_text, language,
                timeout_seconds=app_settings["timeout"],
                **_gen_kwargs(),
            )
            duration = len(audio) / sr
            audio_parts.append((sr, audio))
            table_rows.append([str(i + 1), preview, f"{duration:.1f}s"])
            succeeded += 1
        except Exception as e:
            table_rows.append([str(i + 1), preview, f"Failed: {e}"])
            failed += 1

    if not audio_parts:
        return None, table_rows, "All segments failed"

    combined = concatenate_audio(audio_parts, silence_ms=int(silence_ms))
    voice_info = library_voice if library_voice and library_voice != "None" else "uploaded"
    history.add(
        mode="voice_clone", text=f"[Batch: {succeeded} segments]",
        language=language, audio=combined, voice_params=f"batch ref: {voice_info}",
    )
    status_msg = f"Generated {succeeded}/{len(segments)} segments"
    if failed:
        status_msg += f" ({failed} failed)"
    return gr.update(value=combined), table_rows, status_msg


# ---------------------------------------------------------------------------
# Script mode handlers
# ---------------------------------------------------------------------------
def parse_script_handler(raw_text):
    """Parse script and return speaker info for voice assignment UI."""
    if not raw_text.strip():
        gr.Warning("Enter a script first.")
        return gr.update(), "Enter a script first", {}

    parsed = parse_script(raw_text)

    if parsed.errors:
        gr.Warning(parsed.errors[0])
        return gr.update(), "; ".join(parsed.errors), {}

    speaker_info = {
        "speakers": parsed.speakers,
        "line_count": len(parsed.lines),
        "lines_per_speaker": {},
    }
    for line in parsed.lines:
        speaker_info["lines_per_speaker"].setdefault(line.speaker, 0)
        speaker_info["lines_per_speaker"][line.speaker] += 1

    summary_parts = [f"Found {len(parsed.speakers)} speakers, {len(parsed.lines)} lines:"]
    for spk in parsed.speakers:
        count = speaker_info["lines_per_speaker"].get(spk, 0)
        summary_parts.append(f"  {spk}: {count} lines")

    # Build visibility updates for speaker slots
    updates = []
    for i in range(MAX_SCRIPT_SPEAKERS):
        if i < len(parsed.speakers):
            updates.append(gr.update(visible=True, label=f"Speaker: {parsed.speakers[i]}"))
        else:
            updates.append(gr.update(visible=False))

    return updates, "\n".join(summary_parts), speaker_info


def generate_script_handler(raw_text, assignments_state, silence_ms, progress=gr.Progress()):
    """Generate audio for a parsed multi-speaker script.

    assignments_state is a dict mapping speaker name to voice config:
      {speaker: {"mode": ..., "speaker": ..., "language": ..., "instruct": ..., "library_voice": ...}}
    """
    if not raw_text.strip():
        gr.Warning("Enter a script first.")
        return None, [["(empty)", "", "", ""]], "Enter script first"

    parsed = parse_script(raw_text)
    if parsed.errors:
        gr.Warning(parsed.errors[0])
        return None, [["(error)", "", "", ""]], parsed.errors[0]

    if not assignments_state:
        gr.Warning("Parse the script and assign voices first.")
        return None, [["(error)", "", "", ""]], "Parse script first"

    # Group lines by model type for efficient model swapping
    groups = group_by_model_type(parsed.lines, assignments_state)

    # Generate audio for each line, grouped by model type
    audio_by_line_number = {}  # line_number -> (sr, audio)
    table_rows = []
    succeeded, failed = 0, 0

    model_type_labels = {
        "custom_voice": "Custom Voice",
        "voice_design": "Voice Design",
        "base": "Voice Clone",
    }

    total_lines = len(parsed.lines)
    done = 0

    for model_type, lines in groups.items():
        label = model_type_labels.get(model_type, model_type)
        line_nums = [str(l.line_number) for l in lines]
        progress(done / total_lines, desc=f"Generating {label} lines ({', '.join(line_nums)})...")

        for line in lines:
            assignment = assignments_state.get(line.speaker, {})
            mode = assignment.get("mode", "custom_voice")
            lang = assignment.get("language", "English")

            try:
                if mode == "custom_voice":
                    sr, audio = generate_with_timeout(
                        engine.generate_custom_voice,
                        line.text,
                        assignment.get("speaker", DEFAULT_SPEAKERS[0]),
                        lang,
                        assignment.get("instruct", ""),
                        timeout_seconds=app_settings["timeout"],
                        **_gen_kwargs(),
                    )
                elif mode == "voice_design":
                    sr, audio = generate_with_timeout(
                        engine.generate_voice_design,
                        line.text,
                        lang,
                        assignment.get("instruct", ""),
                        timeout_seconds=app_settings["timeout"],
                        **_gen_kwargs(),
                    )
                elif mode == "voice_clone":
                    lib_voice = assignment.get("library_voice", "")
                    if not lib_voice or lib_voice == "None":
                        raise ValueError("No library voice selected for clone mode")
                    voice = library.load_voice(lib_voice)
                    ref_audio_path = library.get_ref_audio_path(lib_voice)
                    ref_text = voice["ref_text"]
                    sr, audio = generate_with_timeout(
                        engine.generate_voice_clone,
                        line.text, ref_audio_path, ref_text, lang,
                        timeout_seconds=app_settings["timeout"],
                        **_gen_kwargs(),
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                audio_by_line_number[line.line_number] = (sr, audio)
                succeeded += 1
            except Exception as e:
                audio_by_line_number[line.line_number] = None
                failed += 1

            done += 1
            progress(done / total_lines)

    # Reassemble in script order and build results table
    audio_segments = []
    for line in parsed.lines:
        preview = line.text[:40] + "..." if len(line.text) > 40 else line.text
        result = audio_by_line_number.get(line.line_number)
        if result is not None:
            sr, audio = result
            duration = len(audio) / sr
            audio_segments.append((sr, audio))
            table_rows.append([str(line.line_number), line.speaker, preview, f"{duration:.1f}s"])
        else:
            table_rows.append([str(line.line_number), line.speaker, preview, "Failed"])

    if not audio_segments:
        return None, table_rows, "All lines failed"

    combined = concatenate_audio(audio_segments, silence_ms=int(silence_ms))
    # Record to history
    speakers_used = ", ".join(parsed.speakers[:4])
    if len(parsed.speakers) > 4:
        speakers_used += "..."
    history.add(
        mode="custom_voice",
        text=f"[Script: {succeeded} lines, speakers: {speakers_used}]",
        language="Multi", audio=combined,
        voice_params="script mode",
    )

    status_msg = f"Generated {succeeded}/{total_lines} lines"
    if failed:
        status_msg += f" ({failed} failed)"
    return gr.update(value=combined), table_rows, status_msg


# ---------------------------------------------------------------------------
# History handlers
# ---------------------------------------------------------------------------
def history_preview(entry_id):
    """Load audio for a history entry."""
    if not entry_id or entry_id == "(empty)":
        return None
    audio = history.get_audio(entry_id)
    if audio is None:
        gr.Warning("Audio not found for this entry.")
        return None
    return audio


def history_delete(entry_id):
    """Delete a single history entry."""
    if not entry_id or entry_id == "(empty)":
        return history.table_data(), "Select an entry first"
    history.delete_entry(entry_id)
    return history.table_data(), f"Deleted entry {entry_id}"


def history_clear():
    """Clear all history."""
    history.clear()
    return history.table_data(), "History cleared"


def history_save_wav(entry_id):
    """Save a history entry's audio to the output directory."""
    if not entry_id or entry_id == "(empty)":
        return "Select an entry first"
    audio = history.get_audio(entry_id)
    if audio is None:
        return "Audio not found"
    return save_audio(audio, "history")


def history_regenerate(entry_id):
    """Get params from a history entry for regeneration."""
    if not entry_id or entry_id == "(empty)":
        return "Select an entry first"
    entry = history.get_entry(entry_id)
    if entry is None:
        return "Entry not found"
    parts = [
        f"Mode: {entry.mode.replace('_', ' ').title()}",
        f"Language: {entry.language}",
    ]
    if entry.speaker:
        parts.append(f"Speaker: {entry.speaker}")
    if entry.voice_params:
        parts.append(f"Params: {entry.voice_params}")
    parts.append(f"Text: {entry.text}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Save-to-library handlers
# ---------------------------------------------------------------------------
def save_design_to_library(audio_tuple, name, language, description, spoken_text):
    if audio_tuple is None:
        gr.Warning("Generate audio first before saving to library.")
        return "No audio to save"
    if not name.strip():
        gr.Warning("Please enter a name for this voice.")
        return "Enter a voice name"
    sr, audio = audio_tuple
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tmp_path = os.path.join(OUTPUT_DIR, f"_tmp_design_{int(time.time())}.wav")
    sf.write(tmp_path, audio, sr)
    library.save_voice(
        name=name,
        ref_audio_path=tmp_path,
        ref_text=spoken_text,
        language=language,
        description=description,
        source="design",
    )
    os.remove(tmp_path)
    return f"Voice '{name}' saved to library"


def save_clone_to_library(ref_audio, ref_text, name, language):
    if not ref_audio:
        gr.Warning("No reference audio to save.")
        return "No reference audio"
    if not name.strip():
        gr.Warning("Please enter a name for this voice.")
        return "Enter a voice name"
    if not ref_text or not ref_text.strip():
        gr.Warning("Reference transcript is required.")
        return "Enter transcript"
    library.save_voice(
        name=name,
        ref_audio_path=ref_audio,
        ref_text=ref_text,
        language=language,
        source="clone",
    )
    return f"Voice '{name}' saved to library"


# ---------------------------------------------------------------------------
# YT Voice Clone handlers
# ---------------------------------------------------------------------------
def fetch_yt_info(url):
    if not url or not url.strip():
        return gr.update(), "_Enter a URL above._", {}, "Enter a YouTube URL"
    try:
        info = yt_extractor.fetch_info(url.strip())
        dur = info.get("duration") or 0
        h, rem = divmod(int(dur), 3600)
        m, s = divmod(rem, 60)
        dur_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        if info.get("has_manual_subs"):
            sub_note = "Manual subtitles"
        elif info.get("has_auto_subs"):
            sub_note = "Auto-generated subtitles (may need editing)"
        else:
            sub_note = "No subtitles — enter transcript manually"
        md = (
            f"**{info['title']}**\n\n"
            f"Duration: {dur_str}  |  {sub_note}"
            + (f"\n\nChannel: {info['uploader']}" if info.get("uploader") else "")
        )
        state = {
            "id": info["id"],
            "title": info["title"],
            "duration": dur,
            "language": info.get("language"),
        }
        return (
            gr.update(value=info.get("thumbnail") or None),
            md,
            state,
            f"✓ Fetched: {info['title'][:60]}",
        )
    except (ValueError, RuntimeError) as e:
        gr.Warning(str(e))
        return gr.update(), f"**Error:** {e}", {}, str(e)
    except Exception as e:
        gr.Warning(f"Failed to fetch video info: {e}")
        return gr.update(), f"**Error:** {e}", {}, f"Error: {e}"


def extract_yt_clip(url, start_str, end_str, video_state, progress=gr.Progress()):
    if not url or not url.strip():
        return gr.update(), gr.update(), "Enter a YouTube URL first"
    if not video_state or not video_state.get("id"):
        return gr.update(), gr.update(), "Fetch video info first (Step 1)"

    vid_dur = video_state.get("duration")

    # Resolve start — blank defaults to beginning of video
    try:
        start_sec = yt_extractor.parse_timestamp(start_str or "0")
    except ValueError as e:
        gr.Warning(str(e))
        return gr.update(), gr.update(), f"Bad start time: {e}"

    # Resolve end — blank defaults to video duration (or full video)
    if not end_str or not end_str.strip():
        if not vid_dur:
            gr.Warning("Enter an end time — video duration unknown.")
            return gr.update(), gr.update(), "Enter an end time"
        end_sec = float(vid_dur)
    else:
        try:
            end_sec = yt_extractor.parse_timestamp(end_str)
        except ValueError:
            gr.Warning("Invalid end time — use mm:ss")
            return gr.update(), gr.update(), "Enter a valid end time"

    if end_sec <= start_sec:
        gr.Warning("End time must be after start time.")
        return gr.update(), gr.update(), "End must be after start"

    clip_dur = end_sec - start_sec
    if clip_dur < 3.0:
        gr.Warning("Clip must be at least 3 seconds.")
        return gr.update(), gr.update(), "Clip too short (min 3 s)"
    if clip_dur > 60.0:
        gr.Warning("Clip capped at 60 seconds.")
        end_sec = start_sec + 60.0
        clip_dur = 60.0
    elif clip_dur > 30.0:
        gr.Warning(f"Clip is {clip_dur:.0f}s — 5-20 s gives best clone quality.")

    if vid_dur and end_sec > vid_dur + 1:
        gr.Warning(f"End time exceeds video duration ({vid_dur:.0f}s).")
        return gr.update(), gr.update(), "End time beyond video end"

    video_id = video_state["id"]

    try:
        progress(0.0, desc="Starting…")
        wav_path, subs_ok = yt_extractor.download_clip(
            url.strip(), video_id, start_sec, end_sec,
            progress_cb=lambda f, d: progress(f, desc=d),
        )
        progress(0.90, desc="Extracting transcript…")
        transcript = yt_extractor.extract_transcript(video_id, start_sec, end_sec)
        progress(1.0, desc="Done")

        note = "transcript auto-filled" if transcript else "no subtitles — enter transcript manually"
        return (
            gr.update(value=wav_path),
            gr.update(value=transcript),
            f"✓ {clip_dur:.1f}s clip extracted — {note}",
        )
    except Exception as e:
        gr.Warning(f"Extraction failed: {e}")
        return gr.update(), gr.update(), f"Error: {e}"


def clone_yt_voice(text, ref_audio, transcript, language, voice_name):
    errors = []
    if not text or not text.strip():
        errors.append("text to synthesize")
    if not ref_audio:
        errors.append("reference clip (extract one first)")
    if not transcript or not transcript.strip():
        errors.append("reference transcript")
    if not voice_name or not voice_name.strip():
        errors.append("voice name")
    if errors:
        msg = "Required: " + ", ".join(errors)
        gr.Warning(msg)
        return gr.update(), msg, msg, gr.update(), gr.update()

    try:
        t0 = time.time()
        result = generate_with_timeout(
            engine.generate_voice_clone,
            text.strip(), ref_audio, transcript.strip(), language,
            timeout_seconds=app_settings["timeout"],
            **_gen_kwargs(),
        )
        elapsed = time.time() - t0
        lib_msg = save_clone_to_library(
            ref_audio, transcript.strip(), voice_name.strip(), language
        )
        history.add(
            mode="voice_clone",
            text=text.strip(),
            language=language,
            audio=result,
            voice_params=f"yt: {voice_name.strip()}",
        )
        extra = ""
        if app_settings["autosave"]:
            extra = " | " + save_audio(result, "yt_clone")
        status_msg = f"Generated in {elapsed:.1f}s | {lib_msg}{extra}"
        return (
            gr.update(value=result),
            status_msg,
            f"✓ {lib_msg}",
            gr.update(choices=_voice_choices()),
            gr.update(value=_voice_table()),
        )
    except GenerationTimeout as e:
        gr.Warning(str(e))
        return gr.update(), str(e), str(e), gr.update(), gr.update()
    except Exception as e:
        gr.Warning(f"Generation failed: {e}")
        return gr.update(), f"Error: {e}", f"Error: {e}", gr.update(), gr.update()


def clear_yt_cache():
    n = yt_extractor.clear_cache()
    return f"YT cache cleared — {n} entr{'y' if n == 1 else 'ies'} removed"


# ---------------------------------------------------------------------------
# Library management handlers
# ---------------------------------------------------------------------------
def preview_voice(voice_name):
    if not voice_name or voice_name == "(empty)":
        return None
    try:
        path = library.get_ref_audio_path(voice_name)
        if os.path.isfile(path):
            return path
    except Exception:
        pass
    return None


def delete_voice(voice_name):
    if not voice_name or voice_name == "(empty)":
        return _voice_table(), "Select a voice first"
    if not library.delete_voice(voice_name):
        return _voice_table(), f"Voice '{voice_name}' not found"
    return _voice_table(), f"Deleted '{voice_name}'"


def rename_voice(old_name, new_name):
    if not old_name or old_name == "(empty)":
        return _voice_table(), "Select a voice first"
    if not new_name.strip():
        return _voice_table(), "Enter a new name"
    ok = library.rename_voice(old_name, new_name.strip())
    if ok:
        return _voice_table(), f"Renamed '{old_name}' to '{new_name.strip()}'"
    return _voice_table(), f"Rename failed (name may already exist)"


def import_voice(audio_path, transcript, name, language):
    if not audio_path:
        gr.Warning("Upload audio to import.")
        return _voice_table(), "Upload audio first"
    if not name.strip():
        gr.Warning("Enter a name for the imported voice.")
        return _voice_table(), "Enter a name"
    if not transcript or not transcript.strip():
        gr.Warning("Transcript required for imported voice.")
        return _voice_table(), "Enter transcript"
    library.save_voice(
        name=name.strip(),
        ref_audio_path=audio_path,
        ref_text=transcript.strip(),
        language=language,
        description="Imported voice",
        source="import",
    )
    return _voice_table(), f"Imported '{name.strip()}'"


# ---------------------------------------------------------------------------
# Settings handlers
# ---------------------------------------------------------------------------
def apply_settings(
    model_size, quantization,
    temperature, top_k, top_p, repetition_penalty, max_tokens, timeout,
    output_dir, autosave, jit_compile, default_language,
):
    model_changed = (
        model_size != engine.model_size
        or quantization != engine.quantization
        or jit_compile != engine.jit_compile
    )
    engine.model_size = model_size
    engine.quantization = quantization
    engine.jit_compile = jit_compile
    if model_changed:
        engine.unload_model()

    app_settings["temperature"] = temperature
    app_settings["top_k"] = int(top_k)
    app_settings["top_p"] = top_p
    app_settings["repetition_penalty"] = repetition_penalty
    app_settings["max_tokens"] = int(max_tokens)
    app_settings["timeout"] = int(timeout)
    app_settings["output_dir"] = output_dir.strip() or OUTPUT_DIR
    app_settings["autosave"] = autosave
    app_settings["default_language"] = default_language

    os.makedirs(app_settings["output_dir"], exist_ok=True)

    parts = [f"size: {model_size}, quant: {quantization}"]
    if model_changed:
        parts.append("model unloaded")
    status = f"Settings applied — {', '.join(parts)}."
    lang_update = gr.update(value=default_language)
    return status, lang_update, lang_update, lang_update, lang_update, lang_update


def reset_generation_defaults():
    return (
        gr.update(value=DEFAULT_TEMPERATURE),
        gr.update(value=DEFAULT_TOP_K),
        gr.update(value=DEFAULT_TOP_P),
        gr.update(value=DEFAULT_REPETITION_PENALTY),
        gr.update(value=DEFAULT_MAX_TOKENS),
        gr.update(value=DEFAULT_TIMEOUT),
    )


def unload_model():
    engine.unload_model()
    return "Model unloaded. RAM freed."


def delete_cached_models():
    """Delete all Qwen3-TTS model files from the HuggingFace cache."""
    engine.unload_model()
    hf_cache = _get_hf_cache_dir()
    if not os.path.isdir(hf_cache):
        return "HuggingFace cache directory not found", "No model loaded"
    deleted = []
    failed = []
    for entry in os.listdir(hf_cache):
        if entry.startswith("models--mlx-community--Qwen3-TTS-12Hz-"):
            path = os.path.join(hf_cache, entry)
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                    deleted.append(entry.replace("models--mlx-community--", ""))
                except OSError as e:
                    failed.append(f"{entry.replace('models--mlx-community--', '')}: {e}")
    parts = []
    if deleted:
        parts.append(f"Deleted {len(deleted)} model(s): {', '.join(deleted)}")
    if failed:
        parts.append(f"Failed to delete {len(failed)}: {'; '.join(failed)}")
    if parts:
        return " | ".join(parts), "No model loaded"
    return "No Qwen3-TTS models found in cache", "No model loaded"


def get_model_status():
    if engine.current_model is None:
        return "No model loaded"
    repo = engine.get_repo_id(engine.current_model_type)
    return f"Loaded: {repo}"


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Qwen3-TTS Studio") as app:

    gr.HTML(
        "<div class='app-header'>"
        "<h1>Qwen3-TTS Studio</h1>"
        "<p class='subtitle'>Local AI Text-to-Speech &middot; MLX &middot; Apple Silicon</p>"
        "</div>"
    )

    with gr.Tabs():
        # =================================================================
        # Tab 1: Custom Voice
        # =================================================================
        with gr.Tab("Custom Voice"):
            with gr.Row():
                with gr.Column(scale=2):
                    cv_text = gr.Textbox(
                        label="Text",
                        lines=5,
                        placeholder="Enter text to speak...",
                    )
                    with gr.Row():
                        cv_speaker = gr.Dropdown(
                            choices=DEFAULT_SPEAKERS,
                            value=DEFAULT_SPEAKERS[0],
                            label="Speaker",
                        )
                        cv_language = gr.Dropdown(
                            choices=LANGUAGES,
                            value="English",
                            label="Language",
                        )
                    cv_instruct = gr.Textbox(
                        label="Style Instruction (optional)",
                        lines=1,
                        placeholder="e.g. Speak warmly, Sound excited...",
                    )
                    gr.Markdown(
                        "_Tip: 1–4 sentences work best. Very long text may hit the 120 s timeout._",
                        elem_classes=["text-hint"],
                    )
                    cv_generate = gr.Button("Generate", variant="primary")
                    # Batch Mode Accordion
                    with gr.Accordion("Batch Mode", open=False, elem_classes=["batch-accordion"]):
                        with gr.Row():
                            cv_batch_split = gr.Radio(
                                ["paragraph", "sentence", "line"],
                                value=DEFAULT_BATCH_SPLIT_MODE,
                                label="Split Mode",
                            )
                            cv_batch_silence = gr.Slider(
                                0, 2000, value=DEFAULT_SILENCE_GAP_MS, step=50,
                                label="Silence Gap (ms)",
                            )
                        cv_batch_generate = gr.Button("Generate Batch", variant="primary")
                        cv_batch_table = gr.Dataframe(
                            headers=["#", "Text", "Status"],
                            value=[["", "", ""]],
                            label="Batch Results",
                            interactive=False,
                        )
                        cv_batch_audio = gr.Audio(label="Combined Output", type="numpy", interactive=False, buttons=["download"])
                        with gr.Row():
                            cv_batch_save = gr.Button("Save Combined WAV")
                            cv_batch_status = gr.Textbox(label="Batch Status", interactive=False)
                with gr.Column(scale=1, elem_classes=["output-col"]):
                    cv_audio = gr.Audio(label="Output", type="numpy", interactive=False, buttons=["download"])
                    cv_save = gr.Button("Save WAV")
                    cv_save_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder="Save path appears here…",
                        elem_classes=["save-status-text"],
                    )

        # =================================================================
        # Tab 2: Voice Design
        # =================================================================
        with gr.Tab("Voice Design"):
            with gr.Row():
                with gr.Column(scale=2):
                    vd_text = gr.Textbox(
                        label="Text",
                        lines=5,
                        placeholder="Enter text to speak...",
                    )
                    vd_instruct = gr.Textbox(
                        label="Voice Description",
                        lines=2,
                        placeholder="e.g. A deep, calm male narrator with a British accent",
                    )
                    vd_language = gr.Dropdown(
                        choices=LANGUAGES, value="English", label="Language"
                    )
                    gr.Markdown(
                        "_Tip: 1–4 sentences work best. Very long text may hit the 120 s timeout._",
                        elem_classes=["text-hint"],
                    )
                    vd_generate = gr.Button("Generate", variant="primary")
                    with gr.Accordion("Save to Voice Library", open=False, elem_classes=["lib-save-accordion"]):
                        with gr.Row():
                            vd_lib_name = gr.Textbox(
                                label="Voice Name", placeholder="my_narrator", scale=2
                            )
                            vd_lib_save = gr.Button("Save Voice to Library", scale=1)
                        vd_lib_status = gr.Textbox(
                            show_label=False, interactive=False,
                            placeholder="Library status…",
                            elem_classes=["save-status-text"],
                        )
                    # Batch Mode Accordion
                    with gr.Accordion("Batch Mode", open=False, elem_classes=["batch-accordion"]):
                        with gr.Row():
                            vd_batch_split = gr.Radio(
                                ["paragraph", "sentence", "line"],
                                value=DEFAULT_BATCH_SPLIT_MODE,
                                label="Split Mode",
                            )
                            vd_batch_silence = gr.Slider(
                                0, 2000, value=DEFAULT_SILENCE_GAP_MS, step=50,
                                label="Silence Gap (ms)",
                            )
                        vd_batch_generate = gr.Button("Generate Batch", variant="primary")
                        vd_batch_table = gr.Dataframe(
                            headers=["#", "Text", "Status"],
                            value=[["", "", ""]],
                            label="Batch Results",
                            interactive=False,
                        )
                        vd_batch_audio = gr.Audio(label="Combined Output", type="numpy", interactive=False, buttons=["download"])
                        with gr.Row():
                            vd_batch_save = gr.Button("Save Combined WAV")
                            vd_batch_status = gr.Textbox(label="Batch Status", interactive=False)
                with gr.Column(scale=1, elem_classes=["output-col"]):
                    vd_audio = gr.Audio(label="Output", type="numpy", interactive=False, buttons=["download"])
                    vd_save = gr.Button("Save WAV")
                    vd_save_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder="Save path appears here…",
                        elem_classes=["save-status-text"],
                    )

        # =================================================================
        # Tab 3: Voice Cloning
        # =================================================================
        with gr.Tab("Voice Cloning"):
            gr.HTML(
                "<div class='info-notice'>"
                "<strong>Reference transcript must exactly match what is spoken in the audio.</strong> "
                "Use a clean 3–30 second clip for best results."
                "</div>"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Row():
                        vc_language = gr.Dropdown(
                            choices=LANGUAGES, value="English", label="Language"
                        )
                        vc_library_voice = gr.Dropdown(
                            choices=_voice_choices(),
                            value="None",
                            label="Load from Library",
                        )
                    vc_ref_audio = gr.Audio(
                        label="Reference Audio",
                        type="filepath",
                        sources=["upload", "microphone"],
                        buttons=["download"],
                    )
                    vc_ref_text = gr.Textbox(
                        label="Reference Transcript (required)",
                        lines=2,
                        placeholder="Exact text spoken in reference audio",
                    )
                    vc_text = gr.Textbox(
                        label="Text to Speak",
                        lines=5,
                        placeholder="Enter text to speak in the cloned voice...",
                    )
                    vc_generate = gr.Button("Generate", variant="primary")
                    with gr.Accordion("Save to Voice Library", open=False, elem_classes=["lib-save-accordion"]):
                        with gr.Row():
                            vc_lib_name = gr.Textbox(
                                label="Voice Name", placeholder="my_clone", scale=2
                            )
                            vc_lib_save = gr.Button("Save Voice to Library", scale=1)
                        vc_lib_status = gr.Textbox(
                            show_label=False, interactive=False,
                            placeholder="Library status…",
                            elem_classes=["save-status-text"],
                        )
                    # Batch Mode Accordion
                    with gr.Accordion("Batch Mode", open=False, elem_classes=["batch-accordion"]):
                        with gr.Row():
                            vc_batch_split = gr.Radio(
                                ["paragraph", "sentence", "line"],
                                value=DEFAULT_BATCH_SPLIT_MODE,
                                label="Split Mode",
                            )
                            vc_batch_silence = gr.Slider(
                                0, 2000, value=DEFAULT_SILENCE_GAP_MS, step=50,
                                label="Silence Gap (ms)",
                            )
                        vc_batch_generate = gr.Button("Generate Batch", variant="primary")
                        vc_batch_table = gr.Dataframe(
                            headers=["#", "Text", "Status"],
                            value=[["", "", ""]],
                            label="Batch Results",
                            interactive=False,
                        )
                        vc_batch_audio = gr.Audio(label="Combined Output", type="numpy", interactive=False, buttons=["download"])
                        with gr.Row():
                            vc_batch_save = gr.Button("Save Combined WAV")
                            vc_batch_status = gr.Textbox(label="Batch Status", interactive=False)
                with gr.Column(scale=1, elem_classes=["output-col"]):
                    vc_audio = gr.Audio(label="Output", type="numpy", interactive=False, buttons=["download"])
                    vc_save = gr.Button("Save WAV")
                    vc_save_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder="Save path appears here…",
                        elem_classes=["save-status-text"],
                    )

        # =================================================================
        # Tab 4: YT Voice Clone
        # =================================================================
        with gr.Tab("YT Voice Clone"):
            gr.HTML(
                "<div class='info-notice'>"
                "Clone any voice from YouTube: fetch a video, pick a timestamp range, "
                "auto-extract the clip with aligned transcript, then generate."
                "</div>"
            )
            yt_video_state = gr.State({"id": None, "title": None, "duration": None})

            with gr.Row():
                # Left column — step-by-step controls
                with gr.Column(scale=3):
                    gr.Markdown("**Step 1 — Video URL**", elem_classes=["yt-step"])
                    yt_url = gr.Textbox(
                        label="YouTube URL",
                        placeholder="https://www.youtube.com/watch?v=...",
                        show_label=False,
                    )
                    yt_fetch_btn = gr.Button("Fetch Video Info", variant="secondary")

                    gr.Markdown("**Step 2 — Select Clip**", elem_classes=["yt-step"])
                    with gr.Row():
                        yt_start = gr.Textbox(
                            label="Start Time (mm:ss)", placeholder="blank = beginning", scale=1
                        )
                        yt_end = gr.Textbox(
                            label="End Time (mm:ss)", placeholder="blank = end of video", scale=1
                        )
                    yt_extract_btn = gr.Button("Extract Clip", variant="primary")

                    gr.Markdown("**Step 3 — Review Transcript**", elem_classes=["yt-step"])
                    yt_transcript = gr.Textbox(
                        label="Reference Transcript (auto-filled from subtitles — edit if needed)",
                        lines=3,
                        placeholder="Transcript appears here after extraction, or enter manually…",
                    )

                    gr.Markdown("**Step 4 — Generate & Save**", elem_classes=["yt-step"])
                    yt_text = gr.Textbox(
                        label="Text to Synthesize",
                        lines=4,
                        placeholder="Enter text to speak in the cloned voice…",
                    )
                    with gr.Row():
                        yt_language = gr.Dropdown(
                            choices=LANGUAGES, value="English", label="Language", scale=1
                        )
                        yt_voice_name = gr.Textbox(
                            label="Voice Name (saved to library)",
                            placeholder="yt_speaker",
                            scale=2,
                        )
                    yt_clone_btn = gr.Button("Clone & Save to Library", variant="primary")

                # Right column — preview + output panel
                with gr.Column(scale=1, elem_classes=["output-col"]):
                    yt_thumbnail = gr.Image(
                        label="Video Thumbnail",
                        height=180,
                    )
                    yt_video_info = gr.Markdown(
                        "_Video info appears here after fetching._"
                    )
                    yt_clip_audio = gr.Audio(
                        label="Reference Clip Preview",
                        type="filepath",
                        interactive=False,
                        buttons=["download"],
                    )
                    yt_audio_out = gr.Audio(
                        label="Generated Audio", type="numpy", interactive=False, buttons=["download"]
                    )
                    yt_status = gr.Textbox(
                        label="Status", interactive=False, elem_classes=["save-status-text"]
                    )

        # =================================================================
        # Tab 5: Script Mode
        # =================================================================
        with gr.Tab("Script Mode") as script_tab:
            gr.HTML(
                "<div class='info-notice'>"
                "<strong>Multi-Speaker Script</strong> &nbsp;—&nbsp; "
                "Each line: <code>SPEAKER: Dialogue text</code> &nbsp; Lines without a label are narration."
                "</div>"
            )
            with gr.Row():
                with gr.Column(scale=2):
                    sm_script = gr.Textbox(
                        label="Script",
                        lines=10,
                        placeholder=(
                            "NARRATOR: Once upon a time, there lived a curious inventor.\n"
                            "EMMA: Father, look what I found in the attic!\n"
                            "FATHER: That is something I built a long time ago."
                        ),
                        elem_classes=["script-editor"],
                    )
                    with gr.Row():
                        sm_parse_btn = gr.Button("Parse Script", variant="primary", scale=1)
                        sm_silence = gr.Slider(
                            0, 2000, value=DEFAULT_SCRIPT_SILENCE_MS, step=50,
                            label="Silence Between Lines (ms)", scale=2,
                        )
                    sm_parse_status = gr.Textbox(label="Parse Result", interactive=False, lines=2)

                    # Voice assignment state
                    sm_assignments = gr.State({})

                    gr.Markdown("### Voice Assignments")
                    # Pre-allocate speaker slots (show/hide based on parse)
                    sm_speaker_groups = []
                    sm_speaker_modes = []
                    sm_speaker_speakers = []
                    sm_speaker_instructs = []
                    sm_speaker_languages = []
                    sm_speaker_lib_voices = []

                    for i in range(MAX_SCRIPT_SPEAKERS):
                        with gr.Group(visible=False, elem_classes=[f"speaker-slot-{i}"]) as grp:
                            sm_speaker_groups.append(grp)
                            with gr.Row():
                                mode = gr.Radio(
                                    ["Custom Voice", "Voice Design", "Voice Clone"],
                                    value="Custom Voice",
                                    label=f"Speaker {i+1} Mode",
                                    scale=2,
                                )
                                sm_speaker_modes.append(mode)
                                lang = gr.Dropdown(
                                    choices=LANGUAGES, value="English",
                                    label="Language", scale=1,
                                )
                                sm_speaker_languages.append(lang)
                            with gr.Row():
                                spk = gr.Dropdown(
                                    choices=DEFAULT_SPEAKERS,
                                    value=DEFAULT_SPEAKERS[0],
                                    label="Speaker",
                                    scale=1,
                                )
                                sm_speaker_speakers.append(spk)
                                inst = gr.Textbox(
                                    label="Instruct / Description",
                                    placeholder="Style instruction or voice description",
                                    scale=2,
                                )
                                sm_speaker_instructs.append(inst)
                                lib_v = gr.Dropdown(
                                    choices=_voice_choices(),
                                    value="None",
                                    label="Library Voice",
                                    scale=1,
                                )
                                sm_speaker_lib_voices.append(lib_v)

                    with gr.Row():
                        sm_generate_btn = gr.Button("Generate Script", variant="primary")
                        sm_save_btn = gr.Button("Save Combined WAV")

                with gr.Column(scale=1):
                    sm_audio = gr.Audio(label="Combined Output", type="numpy", interactive=False, buttons=["download"])
                    sm_table = gr.Dataframe(
                        headers=["Line", "Speaker", "Text", "Status"],
                        value=[["", "", "", ""]],
                        label="Script Results",
                        interactive=False,
                    )
                    sm_status = gr.Textbox(label="Status", interactive=False)

        # =================================================================
        # Tab 6: Voice Library
        # =================================================================
        with gr.Tab("Voice Library"):
            with gr.Row():
                with gr.Column(scale=2):
                    lib_table = gr.Dataframe(
                        headers=["Name", "Source", "Language", "Description"],
                        value=_voice_table(),
                        label="Saved Voices",
                        interactive=False,
                        max_height=300,
                    )
                    with gr.Row():
                        lib_selected = gr.Textbox(
                            label="Voice Name",
                            placeholder="Type or paste voice name",
                            scale=2,
                        )
                        lib_preview_btn = gr.Button("Preview", scale=1)
                        lib_delete_btn = gr.Button("Delete", scale=1)
                    with gr.Row():
                        lib_new_name = gr.Textbox(
                            label="Rename To", placeholder="new_name", scale=2
                        )
                        lib_rename_btn = gr.Button("Rename", scale=1)
                    lib_preview_audio = gr.Audio(label="Reference Audio Preview", buttons=["download"])
                    lib_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder="Status…",
                        elem_classes=["save-status-text"],
                    )
                with gr.Column(scale=1, elem_classes=["output-col"]):
                    gr.Markdown("### Import Voice")
                    lib_import_audio = gr.Audio(
                        label="Audio File", type="filepath", buttons=["download"]
                    )
                    lib_import_transcript = gr.Textbox(
                        label="Transcript", lines=3
                    )
                    lib_import_name = gr.Textbox(
                        label="Name", placeholder="imported_voice"
                    )
                    lib_import_language = gr.Dropdown(
                        choices=LANGUAGES, value="English", label="Language"
                    )
                    lib_import_btn = gr.Button("Import Voice", variant="primary")

        # =================================================================
        # Tab 7: History
        # =================================================================
        with gr.Tab("History"):
            hist_table = gr.Dataframe(
                headers=["ID", "Time", "Mode", "Text", "Duration"],
                value=history.table_data(),
                label="Generation History",
                interactive=False,
                elem_classes=["history-table"],
                max_height=360,
            )
            with gr.Row():
                hist_selected = gr.Textbox(
                    label="Entry ID",
                    placeholder="Paste entry ID to preview/manage",
                    scale=3,
                )
                hist_preview_btn = gr.Button("Preview", scale=1)
                hist_delete_btn = gr.Button("Delete Entry", scale=1)
            with gr.Row():
                hist_save_btn = gr.Button("Save WAV", scale=1)
                hist_regen_btn = gr.Button("Show Params", scale=1)
                hist_clear_btn = gr.Button("Clear All History", scale=1)
            hist_status = gr.Textbox(
                show_label=False, interactive=False,
                placeholder="Status…",
                elem_classes=["save-status-text"],
            )
            with gr.Row():
                hist_audio = gr.Audio(
                    label="Audio Preview", type="numpy", interactive=False, scale=1, buttons=["download"]
                )
                hist_regen_info = gr.Textbox(
                    label="Regeneration Params", interactive=False, lines=3, scale=1
                )

        # =================================================================
        # Tab 8: Settings
        # =================================================================
        with gr.Tab("Settings"):
            with gr.Row():
                # --- Model column ---
                with gr.Column(scale=1):
                    gr.Markdown("### Model")
                    set_size = gr.Radio(
                        ["0.6B", "1.7B"],
                        value=engine.model_size,
                        label="Model Size",
                    )
                    set_quant = gr.Radio(
                        ["8bit", "bf16"],
                        value=engine.quantization,
                        label="Quantization",
                    )
                    set_status = gr.Textbox(
                        label="Loaded Model",
                        value=get_model_status(),
                        interactive=False,
                        elem_classes=["model-status"],
                    )
                    set_unload = gr.Button("Unload Model / Free RAM")
                    set_jit = gr.Checkbox(
                        value=ENABLE_JIT_COMPILE,
                        label="JIT compile model (faster after first run; unloads model on change)",
                    )
                    gr.Markdown("### Model Cache")
                    gr.Textbox(
                        label="HuggingFace Cache Directory",
                        value=os.path.abspath(_get_hf_cache_dir()),
                        interactive=False,
                        elem_classes=["model-status"],
                    )
                    set_delete_models = gr.Button(
                        "Delete Downloaded Models",
                        variant="stop",
                    )
                    set_delete_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder="Models will be re-downloaded on next use.",
                        elem_classes=["save-status-text"],
                    )

                # --- Generation column ---
                with gr.Column(scale=1):
                    gr.Markdown("### Generation")
                    set_temperature = gr.Slider(
                        0.0, 1.5, value=DEFAULT_TEMPERATURE, step=0.05,
                        label="Temperature",
                    )
                    set_top_k = gr.Slider(
                        0, 100, value=DEFAULT_TOP_K, step=1,
                        label="Top-K",
                    )
                    set_top_p = gr.Slider(
                        0.0, 1.0, value=DEFAULT_TOP_P, step=0.05,
                        label="Top-P",
                    )
                    set_rep_penalty = gr.Slider(
                        1.0, 2.0, value=DEFAULT_REPETITION_PENALTY, step=0.05,
                        label="Repetition Penalty",
                    )
                    set_max_tokens = gr.Slider(
                        512, 8192, value=DEFAULT_MAX_TOKENS, step=256,
                        label="Max Tokens",
                    )
                    set_timeout = gr.Slider(
                        30, 300, value=DEFAULT_TIMEOUT, step=10,
                        label="Generation Timeout (seconds)",
                    )
                    set_reset = gr.Button("Reset to Defaults")

                # --- Output column ---
                with gr.Column(scale=1):
                    gr.Markdown("### Output")
                    set_output_dir = gr.Textbox(
                        value=OUTPUT_DIR,
                        label="Output Directory",
                    )
                    set_autosave = gr.Checkbox(
                        value=DEFAULT_AUTOSAVE,
                        label="Auto-save generated audio",
                    )
                    set_default_language = gr.Dropdown(
                        choices=LANGUAGES,
                        value="English",
                        label="Default Language",
                    )
                    gr.Markdown("### YT Cache")
                    set_yt_cache_btn = gr.Button("Clear YT Cache")
                    set_yt_cache_status = gr.Textbox(
                        show_label=False, interactive=False,
                        placeholder=f"Cache: {YT_CACHE_DIR}/",
                        elem_classes=["save-status-text"],
                    )
                    gr.Markdown("### Storage Paths")
                    gr.Textbox(
                        label="Voice Library",
                        value=os.path.abspath(VOICE_LIBRARY_DIR),
                        interactive=False,
                        elem_classes=["model-status"],
                    )
                    gr.Textbox(
                        label="History",
                        value=os.path.abspath(HISTORY_DIR),
                        interactive=False,
                        elem_classes=["model-status"],
                    )

            set_apply = gr.Button("Apply Settings", variant="primary")

    # Status bar
    status = gr.Textbox(
        show_label=False,
        interactive=False,
        elem_classes=["status-bar"],
        value="Ready" + (f" | Warnings: {'; '.join(startup_warnings)}" if startup_warnings else ""),
    )

    # ===================================================================
    # Event wiring
    # ===================================================================

    # --- Custom Voice ---
    cv_generate.click(
        fn=generate_custom_voice,
        inputs=[cv_text, cv_speaker, cv_language, cv_instruct],
        outputs=[cv_audio, status],
        show_progress="minimal",
    )
    cv_save.click(
        fn=lambda audio: save_audio(audio, "custom"),
        inputs=[cv_audio],
        outputs=[cv_save_status],
    )
    cv_batch_generate.click(
        fn=_run_batch_custom_voice,
        inputs=[cv_text, cv_speaker, cv_language, cv_instruct, cv_batch_split, cv_batch_silence],
        outputs=[cv_batch_audio, cv_batch_table, cv_batch_status],
        show_progress="full",
    )
    cv_batch_save.click(
        fn=lambda audio: save_audio(audio, "batch_custom"),
        inputs=[cv_batch_audio],
        outputs=[cv_batch_status],
    )

    # --- Voice Design ---
    vd_generate.click(
        fn=generate_voice_design,
        inputs=[vd_text, vd_language, vd_instruct],
        outputs=[vd_audio, status],
        show_progress="minimal",
    )
    vd_save.click(
        fn=lambda audio: save_audio(audio, "design"),
        inputs=[vd_audio],
        outputs=[vd_save_status],
    )
    vd_batch_generate.click(
        fn=_run_batch_voice_design,
        inputs=[vd_text, vd_language, vd_instruct, vd_batch_split, vd_batch_silence],
        outputs=[vd_batch_audio, vd_batch_table, vd_batch_status],
        show_progress="full",
    )
    vd_batch_save.click(
        fn=lambda audio: save_audio(audio, "batch_design"),
        inputs=[vd_batch_audio],
        outputs=[vd_batch_status],
    )

    def _save_design_and_refresh(*args):
        result = save_design_to_library(*args)
        return result, gr.update(choices=_voice_choices())

    vd_lib_save.click(
        fn=_save_design_and_refresh,
        inputs=[vd_audio, vd_lib_name, vd_language, vd_instruct, vd_text],
        outputs=[vd_lib_status, vc_library_voice],
    )

    # --- Voice Cloning ---
    vc_generate.click(
        fn=generate_voice_clone,
        inputs=[vc_text, vc_ref_audio, vc_ref_text, vc_language, vc_library_voice],
        outputs=[vc_audio, status],
        show_progress="minimal",
    )
    vc_save.click(
        fn=lambda audio: save_audio(audio, "clone"),
        inputs=[vc_audio],
        outputs=[vc_save_status],
    )
    vc_batch_generate.click(
        fn=_run_batch_voice_clone,
        inputs=[vc_text, vc_ref_audio, vc_ref_text, vc_language, vc_library_voice,
                vc_batch_split, vc_batch_silence],
        outputs=[vc_batch_audio, vc_batch_table, vc_batch_status],
        show_progress="full",
    )
    vc_batch_save.click(
        fn=lambda audio: save_audio(audio, "batch_clone"),
        inputs=[vc_batch_audio],
        outputs=[vc_batch_status],
    )

    def _save_clone_and_refresh(*args):
        result = save_clone_to_library(*args)
        return result, gr.update(choices=_voice_choices())

    vc_lib_save.click(
        fn=_save_clone_and_refresh,
        inputs=[vc_ref_audio, vc_ref_text, vc_lib_name, vc_language],
        outputs=[vc_lib_status, vc_library_voice],
    )

    # Refresh library dropdown when clone tab is selected
    def refresh_clone_library():
        return gr.update(choices=_voice_choices(), value="None")

    vc_library_voice.focus(
        fn=refresh_clone_library,
        outputs=[vc_library_voice],
    )

    # --- YT Voice Clone ---
    yt_fetch_btn.click(
        fn=fetch_yt_info,
        inputs=[yt_url],
        outputs=[yt_thumbnail, yt_video_info, yt_video_state, yt_status],
        show_progress="minimal",
    )
    yt_extract_btn.click(
        fn=extract_yt_clip,
        inputs=[yt_url, yt_start, yt_end, yt_video_state],
        outputs=[yt_clip_audio, yt_transcript, yt_status],
        show_progress="full",
    )

    yt_clone_btn.click(
        fn=clone_yt_voice,
        inputs=[yt_text, yt_clip_audio, yt_transcript, yt_language, yt_voice_name],
        outputs=[yt_audio_out, status, yt_status, vc_library_voice, lib_table],
        show_progress="minimal",
    )

    # --- Script Mode ---
    def _parse_and_update_slots(raw_text):
        """Parse script and update speaker slot visibility."""
        if not raw_text.strip():
            gr.Warning("Enter a script first.")
            updates = [gr.update(visible=False) for _ in range(MAX_SCRIPT_SPEAKERS)]
            return *updates, "Enter a script first", {}

        parsed = parse_script(raw_text)

        if parsed.errors:
            gr.Warning(parsed.errors[0])
            updates = [gr.update(visible=False) for _ in range(MAX_SCRIPT_SPEAKERS)]
            return *updates, "; ".join(parsed.errors), {}

        # Build summary
        summary_parts = [f"Found {len(parsed.speakers)} speakers, {len(parsed.lines)} lines:"]
        lines_per_speaker = {}
        for line in parsed.lines:
            lines_per_speaker.setdefault(line.speaker, 0)
            lines_per_speaker[line.speaker] += 1
        for spk in parsed.speakers:
            count = lines_per_speaker.get(spk, 0)
            summary_parts.append(f"  {spk}: {count} lines")

        # Build visibility updates for speaker slots
        updates = []
        for i in range(MAX_SCRIPT_SPEAKERS):
            if i < len(parsed.speakers):
                updates.append(gr.update(visible=True))
            else:
                updates.append(gr.update(visible=False))

        # Initial assignments state
        assignments = {}
        for spk in parsed.speakers:
            assignments[spk] = {
                "mode": "custom_voice",
                "speaker": DEFAULT_SPEAKERS[0],
                "language": "English",
                "instruct": "",
                "library_voice": "None",
            }

        return *updates, "\n".join(summary_parts), assignments

    sm_parse_btn.click(
        fn=_parse_and_update_slots,
        inputs=[sm_script],
        outputs=[*sm_speaker_groups, sm_parse_status, sm_assignments],
    )

    # Update assignments state when speaker slot controls change
    def _build_assignments_from_slots(current_assignments, script_text,
                                      *slot_values):
        """Rebuild assignments dict from all speaker slot values."""
        if not current_assignments or not script_text.strip():
            return current_assignments

        parsed = parse_script(script_text)
        if parsed.errors or not parsed.speakers:
            return current_assignments

        # slot_values: for each of MAX_SCRIPT_SPEAKERS slots:
        #   mode, speaker, instruct, language, library_voice
        values_per_slot = 5
        assignments = {}
        mode_map = {
            "Custom Voice": "custom_voice",
            "Voice Design": "voice_design",
            "Voice Clone": "voice_clone",
        }
        for i, spk in enumerate(parsed.speakers):
            if i >= MAX_SCRIPT_SPEAKERS:
                break
            base = i * values_per_slot
            mode_label = slot_values[base] if base < len(slot_values) else "Custom Voice"
            assignments[spk] = {
                "mode": mode_map.get(mode_label, "custom_voice"),
                "speaker": slot_values[base + 1] if base + 1 < len(slot_values) else DEFAULT_SPEAKERS[0],
                "instruct": slot_values[base + 2] if base + 2 < len(slot_values) else "",
                "language": slot_values[base + 3] if base + 3 < len(slot_values) else "English",
                "library_voice": slot_values[base + 4] if base + 4 < len(slot_values) else "None",
            }

        return assignments

    # Collect all slot control components in order
    all_slot_controls = []
    for i in range(MAX_SCRIPT_SPEAKERS):
        all_slot_controls.extend([
            sm_speaker_modes[i],
            sm_speaker_speakers[i],
            sm_speaker_instructs[i],
            sm_speaker_languages[i],
            sm_speaker_lib_voices[i],
        ])

    # Wire generate button
    def _generate_script_with_assignments(raw_text, assignments, silence_ms, *slot_values, progress=gr.Progress()):
        """Build fresh assignments from slot values, then generate."""
        # Rebuild assignments from current slot values
        fresh = _build_assignments_from_slots(assignments, raw_text, *slot_values)
        return generate_script_handler(raw_text, fresh, silence_ms, progress)

    sm_generate_btn.click(
        fn=_generate_script_with_assignments,
        inputs=[sm_script, sm_assignments, sm_silence, *all_slot_controls],
        outputs=[sm_audio, sm_table, sm_status],
        show_progress="full",
    )
    sm_save_btn.click(
        fn=lambda audio: save_audio(audio, "script"),
        inputs=[sm_audio],
        outputs=[sm_status],
    )

    # Refresh library voice dropdowns in all speaker slots when the tab is selected
    def _refresh_script_lib_voices():
        choices = _voice_choices()
        return [gr.update(choices=choices) for _ in range(MAX_SCRIPT_SPEAKERS)]

    script_tab.select(
        fn=_refresh_script_lib_voices,
        outputs=sm_speaker_lib_voices,
    )

    # --- Voice Library ---
    lib_preview_btn.click(
        fn=preview_voice,
        inputs=[lib_selected],
        outputs=[lib_preview_audio],
    )
    lib_delete_btn.click(
        fn=delete_voice,
        inputs=[lib_selected],
        outputs=[lib_table, lib_status],
    )
    lib_rename_btn.click(
        fn=rename_voice,
        inputs=[lib_selected, lib_new_name],
        outputs=[lib_table, lib_status],
    )
    lib_import_btn.click(
        fn=import_voice,
        inputs=[lib_import_audio, lib_import_transcript, lib_import_name, lib_import_language],
        outputs=[lib_table, lib_status],
    )

    # --- History ---
    hist_preview_btn.click(
        fn=history_preview,
        inputs=[hist_selected],
        outputs=[hist_audio],
    )
    hist_delete_btn.click(
        fn=history_delete,
        inputs=[hist_selected],
        outputs=[hist_table, hist_status],
    )
    hist_clear_btn.click(
        fn=history_clear,
        outputs=[hist_table, hist_status],
    )
    hist_save_btn.click(
        fn=history_save_wav,
        inputs=[hist_selected],
        outputs=[hist_status],
    )
    hist_regen_btn.click(
        fn=history_regenerate,
        inputs=[hist_selected],
        outputs=[hist_regen_info],
    )

    # --- Settings ---
    set_apply.click(
        fn=apply_settings,
        inputs=[
            set_size, set_quant,
            set_temperature, set_top_k, set_top_p, set_rep_penalty,
            set_max_tokens, set_timeout,
            set_output_dir, set_autosave, set_jit, set_default_language,
        ],
        outputs=[
            set_status,
            cv_language, vd_language, vc_language, yt_language, lib_import_language,
        ],
    )
    set_reset.click(
        fn=reset_generation_defaults,
        outputs=[set_temperature, set_top_k, set_top_p, set_rep_penalty, set_max_tokens, set_timeout],
    )
    set_unload.click(
        fn=unload_model,
        outputs=[set_status],
    )
    set_delete_models.click(
        fn=delete_cached_models,
        outputs=[set_delete_status, set_status],
    )
    set_yt_cache_btn.click(fn=clear_yt_cache, outputs=[set_yt_cache_status])

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.queue(max_size=5).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        css=custom_css,
        theme=build_theme(),
    )
