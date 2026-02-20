"""Multi-speaker script parser for Script Mode."""

import re
from dataclasses import dataclass, field

from config import MAX_SCRIPT_SPEAKERS

SPEAKER_PATTERN = re.compile(r"^([A-Za-z0-9_ -]+):\s*(.+)$")
DEFAULT_NARRATOR = "NARRATOR"


@dataclass
class ScriptLine:
    line_number: int
    speaker: str
    text: str


@dataclass
class ParsedScript:
    lines: list[ScriptLine] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def parse_script(raw_text: str) -> ParsedScript:
    """Parse a multi-speaker script into structured lines.

    Format: SPEAKER_NAME: Dialogue text here
    Lines not matching the pattern are treated as narration.

    Returns a ParsedScript with lines, unique speakers, and any errors.
    """
    result = ParsedScript()
    seen_speakers: dict[str, str] = {}  # normalized -> original

    for i, raw_line in enumerate(raw_text.split("\n"), start=1):
        line = raw_line.strip()
        if not line:
            continue

        match = SPEAKER_PATTERN.match(line)
        if match:
            speaker_raw = match.group(1).strip()
            text = match.group(2).strip()
            # Normalize speaker name for deduplication
            speaker_key = speaker_raw.upper()
            if speaker_key not in seen_speakers:
                seen_speakers[speaker_key] = speaker_raw
            speaker = seen_speakers[speaker_key]
        else:
            speaker = DEFAULT_NARRATOR
            text = line
            speaker_key = DEFAULT_NARRATOR
            if speaker_key not in seen_speakers:
                seen_speakers[speaker_key] = DEFAULT_NARRATOR

        if text:
            result.lines.append(ScriptLine(line_number=i, speaker=speaker, text=text))

    result.speakers = list(seen_speakers.values())

    if len(result.speakers) > MAX_SCRIPT_SPEAKERS:
        result.errors.append(
            f"Too many speakers ({len(result.speakers)}). Maximum is {MAX_SCRIPT_SPEAKERS}."
        )

    if not result.lines:
        result.errors.append("No valid lines found in script.")

    return result


def group_by_model_type(
    lines: list[ScriptLine],
    assignments: dict[str, dict],
) -> dict[str, list[ScriptLine]]:
    """Group script lines by the model type needed for each speaker.

    assignments maps speaker name to a dict with at least a "mode" key:
      {"mode": "custom_voice" | "voice_design" | "voice_clone", ...}

    Returns dict mapping model_type to list of ScriptLines.
    """
    groups: dict[str, list[ScriptLine]] = {}
    for line in lines:
        assignment = assignments.get(line.speaker, {})
        mode = assignment.get("mode", "custom_voice")
        # Map mode to engine model type
        model_type_map = {
            "custom_voice": "custom_voice",
            "voice_design": "voice_design",
            "voice_clone": "base",
        }
        model_type = model_type_map.get(mode, "custom_voice")
        groups.setdefault(model_type, []).append(line)
    return groups
