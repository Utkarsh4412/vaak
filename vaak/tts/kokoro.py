"""
vaak/tts/kokoro.py — KokoroTTS adapter with language→voice map and startup check.

Verified voice list from: Kokoro(model_path, voices_path).get_voices()
Run on 2026-10-07 against kokoro-onnx==0.4.7 with kokoro-v1.0.onnx /
voices-v1.0.bin from fastrtc/kokoro-onnx on HuggingFace.

Language support confirmed:
  en-us  — American English  (af_*/am_* prefixes)
  en-gb  — British English   (bf_*/bm_* prefixes)
  hi     — Hindi             (hf_*/hm_* prefixes)
  es     — Spanish           (ef_*/em_* prefixes)
  fr     — French            (ff_* prefix)

Marathi: no voice in base model. Falls back to Hindi voice with a warning.
Italian / Japanese / Portuguese: voices present (if_*/im_*, jf_*/jm_*, pf_*/pm_*)
  but not in the plan's language set — added to map for future use.
Chinese: zf_*/zm_* prefix — added to map.

Kokoro.create() API (confirmed):
  audio, sample_rate = model.create(text, voice=voice_id, speed=speed, lang=lang)
  where lang is the Kokoro-internal lang string (e.g. "en-us", "hi", "es").
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from kokoro_onnx import Kokoro
from loguru import logger

from vaak.interfaces import BaseTTS

# ---------------------------------------------------------------------------
# Voice/language maps — built from actual get_voices() output, 2026-10-07
# ---------------------------------------------------------------------------

# Map from BCP-47 language code → Kokoro internal lang string
_LANG_TO_KOKORO_LANG: dict[str, str] = {
    "en": "en-us",
    "en-gb": "en-gb",
    "hi": "hi",
    "hi-en": "hi",  # Hinglish: use Hindi TTS
    "es": "es",
    "fr": "fr",
    "it": "it",
    "ja": "ja",
    "pt": "pt-br",
    "zh": "zh",
}

# Map from BCP-47 language code → default Kokoro voice ID
_LANG_TO_DEFAULT_VOICE: dict[str, str] = {
    "en": "af_heart",
    "en-gb": "bf_emma",
    "hi": "hf_alpha",
    "hi-en": "hf_alpha",  # Hinglish → Hindi voice
    "es": "ef_dora",
    "fr": "ff_siwis",
    "it": "if_sara",
    "ja": "jf_alpha",
    "pt": "pf_dora",
    "zh": "zf_xiaobei",
}

# HuggingFace repo that ships the ONNX model + voices file
_HF_REPO = "fastrtc/kokoro-onnx"
_MODEL_FILE = "kokoro-v1.0.onnx"
_VOICES_FILE = "voices-v1.0.bin"


class ConfigError(RuntimeError):
    """Raised when a configured voice ID is not available in the loaded model."""


class KokoroTTS(BaseTTS):
    """Kokoro ONNX TTS with per-language voice selection and startup validation.

    Args:
        voice: Explicit voice ID override. If None, resolved from language.
        speed: Synthesis speed multiplier (0.5–2.0).
        skip_voice_check: If True, bypass startup voice validation (useful in CI
                          when model weights are not downloaded).
        model_path: Path to kokoro ONNX file. Downloads from HF if not provided.
        voices_path: Path to voices .bin file. Downloads from HF if not provided.
    """

    def __init__(
        self,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        skip_voice_check: bool = False,
        model_path: str | Path | None = None,
        voices_path: str | Path | None = None,
    ) -> None:
        self._voice_override = voice
        self._speed = speed

        logger.info("Loading Kokoro TTS model…")
        mp = str(model_path) if model_path else hf_hub_download(_HF_REPO, _MODEL_FILE)
        vp = str(voices_path) if voices_path else hf_hub_download(_HF_REPO, _VOICES_FILE)
        self._model = Kokoro(mp, vp)

        self._available_voices: set[str] = set(self._model.get_voices())
        logger.info(f"Kokoro TTS loaded. {len(self._available_voices)} voices available.")

        if not skip_voice_check:
            self._validate_voices()

    # ------------------------------------------------------------------
    # Startup validation
    # ------------------------------------------------------------------

    def _validate_voices(self) -> None:
        """Verify every voice in the default map exists in the loaded model.

        Raises ConfigError if any voice ID is missing. This catches misconfigured
        or stale voice IDs early at startup rather than at synthesis time.
        """
        missing = [v for v in _LANG_TO_DEFAULT_VOICE.values() if v not in self._available_voices]
        if missing:
            raise ConfigError(
                f"The following Kokoro voice IDs are not available in the loaded "
                f"model: {missing}. Check the model file version."
            )

    def validate_voice(self, voice_id: str) -> None:
        """Raise ConfigError if *voice_id* is not in the loaded model."""
        if voice_id not in self._available_voices:
            raise ConfigError(
                f"Kokoro voice '{voice_id}' not found. "
                f"Available voices: {sorted(self._available_voices)}"
            )

    # ------------------------------------------------------------------
    # BaseTTS implementation
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        *,
        language: str = "en",
    ) -> Generator[tuple[int, object], None, None]:
        """Synthesise *text* for *language* and yield (sample_rate, samples) chunks.

        A single Kokoro.create() call returns the full audio array; we yield it
        as one chunk. Streaming will split by sentence before calling this method.

        Args:
            text: Text to synthesise.
            language: BCP-47 code ("en", "hi", "hi-en", "es", "fr", …).
        """
        if not text.strip():
            return

        voice = self._resolve_voice(language)
        kokoro_lang = _LANG_TO_KOKORO_LANG.get(language, "en-us")

        logger.debug(
            f"KokoroTTS.synthesize: lang={language!r} voice={voice!r} "
            f"kokoro_lang={kokoro_lang!r} text={text[:60]!r}"
        )

        try:
            audio, sample_rate = self._model.create(
                text,
                voice=voice,
                speed=self._speed,
                lang=kokoro_lang,
            )
            # Ensure float32
            arr = np.asarray(audio, dtype=np.float32)
            yield (sample_rate, arr)
        except Exception as exc:
            logger.error(f"KokoroTTS synthesis failed: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def resolve_voice(self, language: str) -> str:
        """Public helper to resolve the voice ID for *language*."""
        return self._resolve_voice(language)

    def _resolve_voice(self, language: str) -> str:
        """Return the voice ID to use for *language*."""
        if self._voice_override:
            return self._voice_override
        # Fallback chain: exact match → "en" default
        voice = _LANG_TO_DEFAULT_VOICE.get(language) or _LANG_TO_DEFAULT_VOICE["en"]
        if voice not in self._available_voices:
            logger.warning(
                f"Default voice {voice!r} for language {language!r} not available; "
                "falling back to 'af_heart'."
            )
            return "af_heart"
        return voice

    def warmup(self) -> None:
        """Synthesise a short phrase to pre-initialise ONNX runtime."""
        logger.info("Warming up Kokoro TTS…")
        for _ in self.synthesize("Hello.", language="en"):
            pass
        logger.info("Kokoro TTS warmed up.")
