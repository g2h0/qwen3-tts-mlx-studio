import gc
import threading

import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model

from config import REPO_TEMPLATE, MODEL_VARIANTS, DEFAULT_MODEL_SIZE, DEFAULT_QUANTIZATION, ENABLE_JIT_COMPILE

LOCK_TIMEOUT = 10  # seconds to wait for lock before raising


class TTSEngine:
    """Manages model loading, unloading, and inference."""

    def __init__(self):
        self.current_model = None
        self.current_model_type = None  # "custom_voice" | "voice_design" | "base"
        self.model_size = DEFAULT_MODEL_SIZE
        self.quantization = DEFAULT_QUANTIZATION
        self.jit_compile = ENABLE_JIT_COMPILE
        self._lock = threading.Lock()

    def _acquire_lock(self):
        """Acquire lock with timeout so callers don't hang forever."""
        if not self._lock.acquire(timeout=LOCK_TIMEOUT):
            raise RuntimeError(
                "Engine is busy with a previous generation. Please wait and try again."
            )

    def get_repo_id(self, model_type: str) -> str:
        """Build HuggingFace repo ID from model_type + size + quant."""
        variant = MODEL_VARIANTS[model_type]
        return REPO_TEMPLATE.format(
            size=self.model_size, variant=variant, quant=self.quantization
        )

    def _load_model(self, model_type: str):
        """Load model, unloading previous if different. Caller must hold lock."""
        if self.current_model_type == model_type:
            return
        self._unload_model_unlocked()
        repo_id = self.get_repo_id(model_type)
        self.current_model = load_model(repo_id)
        self.current_model_type = model_type
        if self.jit_compile:
            self._apply_compile()
        mx.eval(self.current_model.parameters())

    def _apply_compile(self):
        """Wrap talker forward passes with mx.compile for faster inference."""
        talker = getattr(self.current_model, "talker", None)
        if talker is None:
            return
        talker.__call__ = mx.compile(talker.__call__, shapeless=True)
        code_pred = getattr(talker, "code_predictor", None)
        if code_pred is not None:
            code_pred.__call__ = mx.compile(code_pred.__call__, shapeless=True)

    def is_model_loaded(self, model_type: str) -> bool:
        """Return True if this model type is already in memory (no swap needed)."""
        return self.current_model_type == model_type

    def unload_model(self):
        """Free memory from current model (thread-safe)."""
        self._acquire_lock()
        try:
            self._unload_model_unlocked()
        finally:
            self._lock.release()

    def _unload_model_unlocked(self):
        """Free memory from current model. Caller must hold lock."""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.current_model_type = None
            gc.collect()

    def generate_custom_voice(self, text, speaker, language, instruct="", **kwargs) -> tuple:
        """Returns (sample_rate, numpy_audio_array)."""
        self._acquire_lock()
        try:
            self._load_model("custom_voice")
            results = list(
                self.current_model.generate_custom_voice(
                    text=text, speaker=speaker, language=language, instruct=instruct,
                    **kwargs,
                )
            )
            return self._to_numpy(results[0])
        finally:
            self._lock.release()

    def generate_voice_design(self, text, language, instruct, **kwargs) -> tuple:
        """Returns (sample_rate, numpy_audio_array)."""
        self._acquire_lock()
        try:
            self._load_model("voice_design")
            results = list(
                self.current_model.generate_voice_design(
                    text=text, language=language, instruct=instruct,
                    **kwargs,
                )
            )
            return self._to_numpy(results[0])
        finally:
            self._lock.release()

    def generate_voice_clone(self, text, ref_audio_path, ref_text, language="English", **kwargs) -> tuple:
        """Returns (sample_rate, numpy_audio_array)."""
        self._acquire_lock()
        try:
            self._load_model("base")
            results = list(
                self.current_model.generate(
                    text=text, ref_audio=ref_audio_path, ref_text=ref_text,
                    language=language,
                    **kwargs,
                )
            )
            return self._to_numpy(results[0])
        finally:
            self._lock.release()

    def _to_numpy(self, result) -> tuple:
        """Convert mlx array result to numpy for Gradio Audio component."""
        audio = np.array(result.audio, dtype=np.float32)
        sr = getattr(result, "sample_rate", 24000)
        return (sr, audio)
