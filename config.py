# Model defaults
DEFAULT_MODEL_SIZE = "1.7B"
DEFAULT_QUANTIZATION = "8bit"

# HuggingFace repo template
REPO_TEMPLATE = "mlx-community/Qwen3-TTS-12Hz-{size}-{variant}-{quant}"

# Model variant mapping
MODEL_VARIANTS = {
    "custom_voice": "CustomVoice",
    "voice_design": "VoiceDesign",
    "base": "Base",
}

# Supported languages
LANGUAGES = [
    "English", "Chinese", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese",
    "Spanish", "Italian",
]

# Default speakers — must match model's actual speaker IDs exactly (case-sensitive)
DEFAULT_SPEAKERS = [
    "serena", "vivian", "uncle_fu", "ryan", "aiden",
    "ono_anna", "sohee", "eric", "dylan",
]

# Audio settings
DEFAULT_SAMPLE_RATE = 24000
OUTPUT_DIR = "./outputs"
VOICE_LIBRARY_DIR = "./voices"
HISTORY_DIR = "./outputs/history"
YT_CACHE_DIR = ".yt_cache"

# History settings
MAX_HISTORY_ENTRIES = 50
MAX_HISTORY_AUDIO_CACHE = 10

# Batch settings
DEFAULT_BATCH_SPLIT_MODE = "paragraph"
DEFAULT_SILENCE_GAP_MS = 300
MAX_BATCH_SEGMENTS = 50

# Script settings
MAX_SCRIPT_SPEAKERS = 8
DEFAULT_SCRIPT_SILENCE_MS = 500

# Generation defaults
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 50
DEFAULT_TOP_P = 1.0
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT = 120

# Output defaults
DEFAULT_AUTOSAVE = False

# JIT compilation — set False to disable if you hit issues
ENABLE_JIT_COMPILE = True

# Gradio settings
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 7860
