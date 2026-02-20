"""Audio utility functions for batch and script generation."""

import re

import numpy as np

from config import DEFAULT_SAMPLE_RATE


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize a numpy audio array to prevent clipping."""
    if audio.size == 0:
        return audio
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return audio
    return audio * (target_peak / peak)


def concatenate_audio(
    segments: list[tuple[int, np.ndarray]],
    silence_ms: int = 300,
) -> tuple[int, np.ndarray]:
    """Join multiple (sample_rate, audio) tuples with silence gaps.

    All segments are resampled to the first segment's sample rate if they differ,
    and peak-normalized before joining.
    """
    if not segments:
        raise ValueError("No audio segments to concatenate")
    if len(segments) == 1:
        sr, audio = segments[0]
        return (sr, normalize_audio(audio))

    sr = segments[0][0]
    silence_samples = int(sr * silence_ms / 1000)
    silence = np.zeros(silence_samples, dtype=np.float32)

    parts = []
    for i, (seg_sr, seg_audio) in enumerate(segments):
        normalized = normalize_audio(seg_audio)
        if seg_sr != sr:
            # Simple linear resampling for mismatched sample rates
            ratio = sr / seg_sr
            new_len = int(len(normalized) * ratio)
            indices = np.linspace(0, len(normalized) - 1, new_len)
            normalized = np.interp(indices, np.arange(len(normalized)), normalized).astype(np.float32)
        parts.append(normalized)
        if i < len(segments) - 1:
            parts.append(silence)

    return (sr, np.concatenate(parts))


def split_text(text: str, mode: str = "paragraph") -> list[str]:
    """Split text into segments by paragraph, sentence, or line.

    Returns non-empty stripped segments.
    """
    if mode == "paragraph":
        segments = re.split(r"\n\s*\n", text)
    elif mode == "sentence":
        # Split on sentence-ending punctuation followed by space or end
        segments = re.split(r"(?<=[.!?])\s+", text)
    elif mode == "line":
        segments = text.split("\n")
    else:
        raise ValueError(f"Unknown split mode: {mode}")

    return [s.strip() for s in segments if s.strip()]
