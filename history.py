"""Generation history with in-memory cache and disk persistence."""

import json
import os
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import soundfile as sf

from config import HISTORY_DIR, MAX_HISTORY_AUDIO_CACHE, MAX_HISTORY_ENTRIES


@dataclass
class HistoryEntry:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    mode: str = ""          # "custom_voice" | "voice_design" | "voice_clone"
    text: str = ""
    language: str = ""
    duration: float = 0.0   # seconds
    speaker: str = ""       # for custom voice
    voice_params: str = ""  # instruct/description/library voice name
    audio_file: str = ""    # filename within HISTORY_DIR


INDEX_FILE = "index.json"


class GenerationHistory:
    """Hybrid in-memory + disk history for generated audio."""

    def __init__(self, history_dir: str = HISTORY_DIR):
        self.history_dir = history_dir
        os.makedirs(self.history_dir, exist_ok=True)
        self._entries: list[HistoryEntry] = []
        self._audio_cache: OrderedDict[str, tuple[int, np.ndarray]] = OrderedDict()
        self._load_index()

    def _index_path(self) -> str:
        return os.path.join(self.history_dir, INDEX_FILE)

    def _load_index(self):
        path = self._index_path()
        if not os.path.isfile(path):
            return
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            self._entries = [HistoryEntry(**e) for e in raw]
        except (json.JSONDecodeError, TypeError) as e:
            print(f"WARNING: Could not load history index, starting fresh: {e}")
            self._entries = []

    def _save_index(self):
        path = self._index_path()
        with open(path, "w") as f:
            json.dump([asdict(e) for e in self._entries], f, indent=2)

    def add(
        self,
        mode: str,
        text: str,
        language: str,
        audio: tuple[int, np.ndarray],
        speaker: str = "",
        voice_params: str = "",
    ) -> HistoryEntry:
        """Record a generation. Saves WAV to disk and updates index."""
        sr, audio_data = audio
        duration = len(audio_data) / sr

        entry = HistoryEntry(
            mode=mode,
            text=text,
            language=language,
            duration=round(duration, 2),
            speaker=speaker,
            voice_params=voice_params,
        )
        entry.audio_file = f"{entry.id}.wav"

        # Save WAV
        wav_path = os.path.join(self.history_dir, entry.audio_file)
        sf.write(wav_path, audio_data, sr)

        # Add to in-memory cache
        self._audio_cache[entry.id] = (sr, audio_data)
        while len(self._audio_cache) > MAX_HISTORY_AUDIO_CACHE:
            self._audio_cache.popitem(last=False)

        # Add to entries, trim oldest
        self._entries.insert(0, entry)
        while len(self._entries) > MAX_HISTORY_ENTRIES:
            removed = self._entries.pop()
            self._audio_cache.pop(removed.id, None)
            old_wav = os.path.join(self.history_dir, removed.audio_file)
            if os.path.isfile(old_wav):
                os.remove(old_wav)

        self._save_index()
        return entry

    def list_entries(self) -> list[HistoryEntry]:
        """Return all entries, newest first."""
        return list(self._entries)

    def get_audio(self, entry_id: str) -> tuple[int, np.ndarray] | None:
        """Get audio for an entry, from cache or disk."""
        if entry_id in self._audio_cache:
            return self._audio_cache[entry_id]

        # Find entry
        entry = next((e for e in self._entries if e.id == entry_id), None)
        if entry is None:
            return None

        wav_path = os.path.join(self.history_dir, entry.audio_file)
        if not os.path.isfile(wav_path):
            return None

        audio_data, sr = sf.read(wav_path, dtype="float32")
        result = (sr, audio_data)

        # Cache it
        self._audio_cache[entry_id] = result
        while len(self._audio_cache) > MAX_HISTORY_AUDIO_CACHE:
            self._audio_cache.popitem(last=False)

        return result

    def get_entry(self, entry_id: str) -> HistoryEntry | None:
        """Get a single entry by ID."""
        return next((e for e in self._entries if e.id == entry_id), None)

    def delete_entry(self, entry_id: str) -> bool:
        """Delete a single history entry."""
        entry = next((e for e in self._entries if e.id == entry_id), None)
        if entry is None:
            return False

        self._entries.remove(entry)
        self._audio_cache.pop(entry_id, None)
        wav_path = os.path.join(self.history_dir, entry.audio_file)
        if os.path.isfile(wav_path):
            os.remove(wav_path)
        self._save_index()
        return True

    def clear(self):
        """Delete all history entries and files."""
        for entry in self._entries:
            wav_path = os.path.join(self.history_dir, entry.audio_file)
            try:
                if os.path.isfile(wav_path):
                    os.remove(wav_path)
            except OSError:
                pass
        self._entries.clear()
        self._audio_cache.clear()
        self._save_index()

    def table_data(self) -> list[list[str]]:
        """Return data formatted for a Gradio Dataframe."""
        if not self._entries:
            return [["(empty)", "", "", "", ""]]
        rows = []
        for e in self._entries:
            text_preview = e.text[:60] + "..." if len(e.text) > 60 else e.text
            mode_label = e.mode.replace("_", " ").title()
            rows.append([e.id, e.timestamp, mode_label, text_preview, f"{e.duration:.1f}s"])
        return rows
