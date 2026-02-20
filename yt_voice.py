"""YouTube audio extraction and subtitle alignment for voice cloning."""
import glob
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
import tempfile

try:
    import pysrt
    _PYSRT_OK = True
except ImportError:
    _PYSRT_OK = False

logger = logging.getLogger(__name__)


class YTVoiceExtractor:
    """Downloads YouTube clips and extracts time-aligned transcripts."""

    def __init__(self, cache_dir: str):
        try:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_dir = cache_dir
        except OSError:
            self.cache_dir = tempfile.mkdtemp(prefix="yt_cache_")
            logger.warning("Cache dir permission denied; using temp: %s", self.cache_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_dependencies(self) -> list[str]:
        """Return list of missing-dependency messages."""
        missing = []
        if not shutil.which("yt-dlp"):
            missing.append("yt-dlp not found — run: pip install yt-dlp")
        if not shutil.which("ffmpeg"):
            missing.append("ffmpeg not found — run: brew install ffmpeg")
        if not _PYSRT_OK:
            missing.append("pysrt not installed (subtitles disabled) — run: pip install pysrt")
        return missing

    def fetch_info(self, url: str) -> dict:
        """Fetch video metadata without downloading."""
        cmd = ["yt-dlp", "--dump-single-json", "--no-playlist", "--no-warnings", url]
        result = self._run_command(cmd, timeout=30)
        info = json.loads(result.stdout)

        subtitles = info.get("subtitles") or {}
        auto_caps = info.get("automatic_captions") or {}
        has_manual = any(k.startswith("en") for k in subtitles)
        has_auto = any(k.startswith("en") for k in auto_caps)

        return {
            "id": info["id"],
            "title": info.get("title", "Unknown"),
            "thumbnail": info.get("thumbnail", ""),
            "duration": info.get("duration"),
            "uploader": info.get("uploader", ""),
            "has_manual_subs": has_manual,
            "has_auto_subs": has_auto,
            "has_subs": has_manual or has_auto,
            "language": info.get("language"),
        }

    def parse_timestamp(self, ts: str) -> float:
        """Parse 'mm:ss', 'hh:mm:ss', or raw seconds string to float."""
        ts = ts.strip()
        if not ts:
            raise ValueError("Empty timestamp")
        if re.fullmatch(r"\d+(\.\d+)?", ts):
            return float(ts)
        parts = ts.split(":")
        try:
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except ValueError:
            pass
        raise ValueError(f"Cannot parse timestamp '{ts}' — use mm:ss or hh:mm:ss")

    def download_clip(
        self,
        url: str,
        video_id: str,
        start_sec: float,
        end_sec: float,
        progress_cb=None,
    ) -> tuple[str, bool]:
        """Download audio clip + subtitles. Returns (wav_path, subs_available)."""
        key = self._cache_key(video_id, start_sec, end_sec)
        clip_dir = os.path.join(self.cache_dir, key)
        wav_path = os.path.join(clip_dir, "clip.wav")

        # Cache hit
        if os.path.isfile(wav_path):
            srt_files = glob.glob(os.path.join(clip_dir, "*.srt"))
            if progress_cb:
                progress_cb(1.0, "Using cached clip")
            return wav_path, bool(srt_files)

        os.makedirs(clip_dir, exist_ok=True)

        if progress_cb:
            progress_cb(0.05, "Downloading audio section…")

        # yt-dlp: download best audio for the time section only
        raw_template = os.path.join(clip_dir, "raw.%(ext)s")
        dl_cmd = [
            "yt-dlp",
            "--format", "bestaudio",
            "--download-sections", f"*{start_sec}-{end_sec}",
            "--force-keyframes-at-cuts",
            "--no-playlist", "--no-warnings",
            "--output", raw_template,
            url,
        ]
        self._run_command_with_retry(dl_cmd, timeout=300)

        # Locate downloaded file (exclude incomplete .part files)
        raw_files = [f for f in glob.glob(os.path.join(clip_dir, "raw.*"))
                     if not f.endswith(".part")]
        if not raw_files:
            raise RuntimeError("Audio download produced no output file")
        raw_file = raw_files[0]

        if progress_cb:
            progress_cb(0.60, "Converting to WAV…")

        # ffmpeg: 24 kHz mono WAV with EBU R128 loudness normalisation
        ff_cmd = [
            "ffmpeg", "-y", "-i", raw_file,
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar", "24000", "-ac", "1",
            wav_path,
        ]
        self._run_command(ff_cmd, timeout=120)
        os.remove(raw_file)  # keep cache lean

        if progress_cb:
            progress_cb(0.80, "Downloading subtitles…")

        # yt-dlp: subtitles only (full video; filtered by time in Python)
        subs_template = os.path.join(clip_dir, "subs")
        subs_cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-subs", "--write-auto-subs",
            "--convert-subs", "srt",
            "--sub-langs", "en.*,en",
            "--no-playlist", "--no-warnings",
            "--output", subs_template,
            url,
        ]
        try:
            self._run_command(subs_cmd, timeout=60)
        except Exception:
            pass  # subtitles are optional

        srt_files = glob.glob(os.path.join(clip_dir, "*.srt"))
        if progress_cb:
            progress_cb(1.0, "Done")
        return wav_path, bool(srt_files)

    def extract_transcript(self, video_id: str, start_sec: float, end_sec: float) -> str:
        """Return cleaned transcript text for the time range, or empty string."""
        key = self._cache_key(video_id, start_sec, end_sec)
        clip_dir = os.path.join(self.cache_dir, key)
        srt_files = glob.glob(os.path.join(clip_dir, "*.srt"))
        if not srt_files:
            return ""
        # Prefer plain 'en' over 'en-US'; shorter filename = more specific
        srt_files.sort(key=lambda p: (len(os.path.basename(p)), p))
        try:
            return self._parse_srt_for_range(srt_files[0], start_sec, end_sec)
        except Exception as e:
            logger.warning("SRT parse failed: %s", e)
            return ""

    def clear_cache(self) -> int:
        """Delete all cached clips. Returns count removed."""
        entries = [d for d in os.listdir(self.cache_dir)
                   if os.path.isdir(os.path.join(self.cache_dir, d))]
        for entry in entries:
            shutil.rmtree(os.path.join(self.cache_dir, entry), ignore_errors=True)
        return len(entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_key(self, video_id: str, start_sec: float, end_sec: float) -> str:
        raw = f"{video_id}_{start_sec:.3f}_{end_sec:.3f}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _parse_srt_for_range(self, srt_path: str, start_sec: float, end_sec: float) -> str:
        """Filter SRT to time range, clean HTML, deduplicate rolling auto-captions."""
        if not _PYSRT_OK:
            return self._parse_srt_fallback(srt_path, start_sec, end_sec)

        subs = pysrt.open(srt_path, encoding="utf-8", error_handling=pysrt.ERROR_PASS)
        slack = 1.0  # 1 s slack for keyframe inaccuracy
        entries = []
        for sub in subs:
            s = sub.start.ordinal / 1000.0
            e = sub.end.ordinal / 1000.0
            if s < end_sec + slack and e > start_sec - slack:
                text = re.sub(r"<[^>]+>", "", sub.text)
                text = re.sub(r"\[.*?\]", "", text)
                text = re.sub(r"♪[^♪]*♪?", "", text)
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    entries.append(text)

        return self._dedup_rolling(entries)

    def _parse_srt_fallback(self, srt_path: str, start_sec: float, end_sec: float) -> str:
        """Regex-based SRT parser used when pysrt is unavailable."""
        ts_re = re.compile(r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)")
        entries = []
        current_start = current_end = None
        lines_buf: list[str] = []

        def flush():
            if current_start is None:
                return
            if current_start < end_sec + 1 and current_end > start_sec - 1:
                text = re.sub(r"<[^>]+>", "", " ".join(lines_buf))
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    entries.append(text)

        with open(srt_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip()
                m = ts_re.match(line)
                if m:
                    flush()
                    h1, m1, s1, ms1 = int(m[1]), int(m[2]), int(m[3]), int(m[4])
                    h2, m2, s2, ms2 = int(m[5]), int(m[6]), int(m[7]), int(m[8])
                    current_start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
                    current_end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
                    lines_buf = []
                elif line and not line.isdigit():
                    lines_buf.append(line)
        flush()
        return self._dedup_rolling(entries)

    @staticmethod
    def _dedup_rolling(entries: list[str]) -> str:
        """Remove rolling auto-caption overlap (YouTube auto-subs repeat last N words)."""
        result_words: list[str] = []
        for entry in entries:
            words = entry.split()
            if not words:
                continue
            if not result_words:
                result_words.extend(words)
                continue
            # Find longest suffix of accumulated words matching prefix of new entry
            overlap = 0
            for length in range(min(len(words), len(result_words), 10), 0, -1):
                if result_words[-length:] == words[:length]:
                    overlap = length
                    break
            result_words.extend(words[overlap:])
        return " ".join(result_words).strip()

    def _run_command(self, cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            stderr = result.stderr or ""
            if any(p in stderr for p in ["Video unavailable", "Private video",
                                          "This video is not available"]):
                raise ValueError("Video is private or unavailable.")
            if any(p in stderr for p in ["Sign in to confirm", "age-restricted"]):
                raise ValueError("Video is age-restricted (requires sign-in).")
            if "HTTP Error 403" in stderr or "HTTP Error 404" in stderr:
                raise ValueError("Access denied. Try: pip install -U yt-dlp")
            raise RuntimeError(f"{cmd[0]} error: {stderr[:300]}")
        return result

    def _run_command_with_retry(self, cmd: list[str],
                                 max_retries: int = 2, timeout: int = 120):
        transient = ("Temporary failure", "Connection reset", "timed out",
                     "HTTP Error 429", "HTTP Error 503")
        last_exc: Exception = RuntimeError("Unknown error")
        for attempt in range(max_retries + 1):
            try:
                return self._run_command(cmd, timeout)
            except (RuntimeError, subprocess.TimeoutExpired) as e:
                last_exc = e
                if attempt < max_retries and any(t in str(e) for t in transient):
                    time.sleep(2 ** attempt)
                    continue
                raise


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------
_instance: "YTVoiceExtractor | None" = None


def get_yt_extractor() -> YTVoiceExtractor:
    global _instance
    if _instance is None:
        from config import YT_CACHE_DIR
        _instance = YTVoiceExtractor(YT_CACHE_DIR)
    return _instance
