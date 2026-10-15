"""
vaak/config.py — Pydantic AgentConfig with YAML + env-var + CLI merge.

Load order (each later level overrides earlier):
  1. Defaults (in model fields below)
  2. YAML file (path from --config or VAAK_CONFIG env var, defaulting to config.yaml)
  3. Environment variables (VAAK_* prefix)
  4. CLI overrides (passed explicitly by the caller)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from loguru import logger
from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class STTConfig(BaseModel):
    backend: Literal["moonshine", "faster_whisper", "router"] = "moonshine"
    """STT backend. moonshine = English fast path; faster_whisper = multilingual; router = language-based routing."""

    faster_whisper_model: str = "small"
    """Model size for faster_whisper backend (tiny/base/small/medium/large-v3)."""

    faster_whisper_compute_type: str = "int8"
    """Quantisation for faster_whisper (int8 is CPU-safe)."""


class LLMConfig(BaseModel):
    model: str = "llama3.2:1b"
    """Ollama model tag."""

    max_tokens: int = Field(default=200, gt=0, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    keep_alive: str = "10m"
    """Ollama keep_alive value — how long to keep model loaded between requests."""

    ollama_host: str = "http://localhost:11434"
    """Base URL of the Ollama server."""


# Voice IDs confirmed from kokoro-onnx 0.4.7 get_voices() output, 2026-10-07.
# Full list produced by: Kokoro.get_voices()
_LANG_TO_DEFAULT_VOICE: dict[str, str] = {
    "en": "af_heart",  # American English female
    "en-gb": "bf_emma",  # British English female
    "hi": "hf_alpha",  # Hindi female
    "es": "ef_dora",  # Spanish female
    "fr": "ff_siwis",  # French female
}

# Mapping from lang code → Kokoro lang param.
# Kokoro.create() requires lang= alongside voice=.
LANG_TO_KOKORO_LANG: dict[str, str] = {
    "en": "en-us",
    "en-gb": "en-gb",
    "hi": "hi",
    "hi-en": "hi",  # Hinglish: use Hindi Kokoro lang; STT stays on Moonshine
    "es": "es",
    "fr": "fr",
}


class TTSConfig(BaseModel):
    voice: str | None = None
    """Explicit Kokoro voice ID. If None, chosen from language."""

    speed: float = Field(default=1.0, ge=0.5, le=2.0)

    skip_voice_check: bool = False
    """Set True to skip the startup Kokoro voice validation (useful in CI)."""

    def resolve_voice(self, language: str) -> str:
        """Return the voice ID to use for *language*, falling back to 'en'."""
        if self.voice:
            return self.voice
        # Hinglish → Hindi voice
        lang_key = language if language in _LANG_TO_DEFAULT_VOICE else "en"
        return _LANG_TO_DEFAULT_VOICE[lang_key]


class MemoryConfig(BaseModel):
    memory_turns: int = Field(default=4, ge=0)
    """Number of conversation turns to keep in the rolling deque (0 = no memory)."""


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    # Language routing
    language: Literal["en", "en-gb", "hi", "hi-en", "es", "fr"] = "en"
    """Default session language. hi-en = Hinglish."""

    # System prompt
    system_prompt: str = (
        "You are a responsive voice assistant participating in a WebRTC audio call. "
        "Keep replies short and clear — your words will be spoken aloud, "
        "so avoid emojis, markdown, and special characters."
    )
    system_prompt_file: str | None = None
    """Path to a .txt file whose content overrides system_prompt."""

    llm_error_reply: str = "I'm having trouble thinking right now. Please try again later."
    """Fallback text to speak if the LLM generation fails."""

    # Sub-configs
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    # Server / UI
    host: str = "127.0.0.1"
    port: int = Field(default=7860, ge=1, le=65535)
    share: bool = False
    phone: bool = False

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # VAD / barge-in
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Voice activity detection threshold (0.0–1.0). Higher = less sensitive."""

    min_silence_ms: int = Field(default=500, ge=100, le=5000)
    """Minimum silence duration in ms before VAD fires end-of-speech."""

    @model_validator(mode="after")
    def _load_system_prompt_file(self) -> AgentConfig:
        if self.system_prompt_file:
            p = Path(self.system_prompt_file)
            if p.exists():
                self.system_prompt = p.read_text(encoding="utf-8").strip()
            else:
                logger.warning(
                    f"system_prompt_file '{self.system_prompt_file}' not found; "
                    "using inline system_prompt."
                )
        return self


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _env_overrides() -> dict:
    """Read VAAK_* environment variables and return a partial config dict."""
    overrides: dict = {}
    mapping = {
        "VAAK_LANGUAGE": ("language",),
        "VAAK_LOG_LEVEL": ("log_level",),
        "VAAK_HOST": ("host",),
        "VAAK_PORT": ("port",),
        "VAAK_SHARE": ("share",),
        "VAAK_LLM_MODEL": ("llm", "model"),
        "VAAK_LLM_HOST": ("llm", "ollama_host"),
        "VAAK_LLM_ERROR_REPLY": ("llm_error_reply",),
        "VAAK_STT_BACKEND": ("stt", "backend"),
        "VAAK_TTS_VOICE": ("tts", "voice"),
        "VAAK_TTS_SPEED": ("tts", "speed"),
        "VAAK_MEMORY_TURNS": ("memory", "memory_turns"),
    }
    for env_key, path in mapping.items():
        val = os.environ.get(env_key)
        if val is None:
            continue
        # Parse booleans and ints where needed
        if env_key in {"VAAK_PORT", "VAAK_MEMORY_TURNS"}:
            val = int(val)  # type: ignore[assignment]
        elif env_key in {"VAAK_SHARE"}:
            val = val.lower() in {"1", "true", "yes"}  # type: ignore[assignment]
        elif env_key == "VAAK_TTS_SPEED":
            val = float(val)  # type: ignore[assignment]

        node = overrides
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = val
    return overrides


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates base)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(
    config_path: str | Path | None = None,
    cli_overrides: dict | None = None,
) -> AgentConfig:
    """Load config from YAML file, then VAAK_* env vars, then CLI overrides.

    Args:
        config_path: Path to YAML file. Defaults to VAAK_CONFIG env var,
                     then "config.yaml" in the cwd (silently ignored if absent).
        cli_overrides: Flat dict of top-level key overrides (e.g. from argparse).

    Returns:
        Validated AgentConfig instance.
    """
    raw: dict = {}

    # 1. YAML file
    path = Path(config_path or os.environ.get("VAAK_CONFIG", "config.yaml"))
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        raw = loaded
        logger.debug(f"Loaded config from {path}")
    else:
        logger.debug(f"Config file {path} not found — using defaults.")

    # 2. VAAK_* env vars
    _deep_merge(raw, _env_overrides())

    # 3. CLI overrides
    if cli_overrides:
        _deep_merge(raw, {k: v for k, v in cli_overrides.items() if v is not None})

    return AgentConfig.model_validate(raw)
