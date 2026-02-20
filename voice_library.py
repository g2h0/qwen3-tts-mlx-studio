import json
import os
import shutil
from datetime import datetime

from config import VOICE_LIBRARY_DIR


class VoiceLibrary:
    """Manages persistent voice profiles stored on disk."""

    def __init__(self, library_dir: str = VOICE_LIBRARY_DIR):
        self.library_dir = library_dir
        os.makedirs(self.library_dir, exist_ok=True)

    def list_voices(self) -> list[dict]:
        """Return all saved voice profiles."""
        voices = []
        if not os.path.isdir(self.library_dir):
            return voices
        for name in sorted(os.listdir(self.library_dir)):
            profile_path = os.path.join(self.library_dir, name, "profile.json")
            if os.path.isfile(profile_path):
                with open(profile_path, "r") as f:
                    voices.append(json.load(f))
        return voices

    def save_voice(
        self,
        name: str,
        ref_audio_path: str,
        ref_text: str,
        language: str,
        description: str = "",
        source: str = "clone",
    ) -> str:
        """Save a new voice profile. Returns profile directory path."""
        safe_name = self._sanitize_name(name)
        voice_dir = os.path.join(self.library_dir, safe_name)
        os.makedirs(voice_dir, exist_ok=True)

        # Copy reference audio
        dest_audio = os.path.join(voice_dir, "reference.wav")
        shutil.copy2(ref_audio_path, dest_audio)

        # Write profile metadata
        profile = {
            "name": safe_name,
            "type": "clone" if source == "clone" else "design",
            "language": language,
            "ref_text": ref_text,
            "ref_audio": "reference.wav",
            "description": description,
            "created": datetime.now().isoformat(timespec="seconds"),
            "source": source,
        }
        profile_path = os.path.join(voice_dir, "profile.json")
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=4)

        return voice_dir

    def load_voice(self, name: str) -> dict:
        """Load a voice profile by name. Returns dict with metadata."""
        profile_path = os.path.join(self.library_dir, name, "profile.json")
        if not os.path.isfile(profile_path):
            raise FileNotFoundError(f"Voice profile '{name}' not found")
        with open(profile_path, "r") as f:
            return json.load(f)

    def delete_voice(self, name: str) -> bool:
        """Delete a voice profile."""
        voice_dir = os.path.join(self.library_dir, name)
        if os.path.isdir(voice_dir):
            shutil.rmtree(voice_dir)
            return True
        return False

    def rename_voice(self, old_name: str, new_name: str) -> bool:
        """Rename a voice profile."""
        old_dir = os.path.join(self.library_dir, old_name)
        safe_new = self._sanitize_name(new_name)
        new_dir = os.path.join(self.library_dir, safe_new)
        if not os.path.isdir(old_dir):
            return False
        if os.path.exists(new_dir):
            return False
        os.rename(old_dir, new_dir)
        # Update profile.json with new name
        profile_path = os.path.join(new_dir, "profile.json")
        if os.path.isfile(profile_path):
            with open(profile_path, "r") as f:
                profile = json.load(f)
            profile["name"] = safe_new
            with open(profile_path, "w") as f:
                json.dump(profile, f, indent=4)
        return True

    def get_ref_audio_path(self, name: str) -> str:
        """Get absolute path to reference audio for a saved voice."""
        return os.path.abspath(
            os.path.join(self.library_dir, name, "reference.wav")
        )

    def _sanitize_name(self, name: str) -> str:
        """Sanitize voice name for use as directory name."""
        safe = "".join(c if c.isalnum() or c in "_- " else "" for c in name)
        return safe.strip().replace(" ", "_") or "unnamed"
