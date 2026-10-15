# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]

### Added
- `BargeInController`: thread-safe coordinator for mid-reply cancellation.
- Mid-reply barge-in cancellation in streaming pipeline with `is_speaking` guard.
- `vad_threshold` and `min_silence_ms` configuration parameters passed to `ReplyOnPause`.
- Unit and integration tests for barge-in stopping TTS and preserving history.
- `SummarizingMemory`: periodic conversation turn summarization at configurable threshold.
- Integrated `SummarizingMemory` into `Session` with backwards-compatible history interface.
- Automatic Hinglish (`hi-en`) prompt injection for conversational bilingual responses.
- Language mapping resolution for Hindi and Hinglish TTS voices.
- End-to-end integration tests for multilingual Hindi and Hinglish pipeline turns.

## [0.1.0] - 2026-10-10

### Added
- `vaak.metrics`: `StageTimer` context manager and `LatencyReport` dataclass for tracking pipeline performance.
- `benchmarks/bench_pipeline.py`: Benchmark script to measure STT → LLM → TTS latency using local WAV fixtures.
- Pipeline now tracks latency per stage and logs a JSON `LatencyReport` per turn.

## [0.1.0-dev] - 2026-10-07

### Added
- `vaak/` package with `BaseSTT`, `BaseLLM`, `BaseTTS` abstract interfaces
- `AgentConfig` Pydantic model — YAML + env-var + CLI merge; `language` and `stt_backend` fields
- `Session` per-connection dataclass — unique uuid4 id, language, rolling history deque, cancel event
- `MoonshineSTT` — English-only fast path via `fastrtc.get_stt_model()`
- `OllamaLLM` — non-streaming baseline with `keep_alive` and startup warmup ping
- `KokoroTTS` — language→voice map (en, en-gb, hi, es, fr); startup voice validation; `lang=` param
- `Pipeline.run()` — STT→LLM→TTS; cancel-checked between TTS chunks; history only on clean completion
- `VoiceHandler` FastRTC adapter — `copy()` creates a fresh `Session(uuid4())` per WebRTC connection
- `vaak` CLI entry point (`uv run vaak --help`)

### Fixed
- `pyproject.toml` `requires-python`: `>=3.13` → `>=3.12,<3.13`
- `pyproject.toml` `name`: `local-voice-agent` → `vaak`; added `build-system` and `[project.scripts]`
- `av==14.2.0` pinned (prebuilt `cp312-win_amd64` wheel; `14.4.0` lacks one)
- `gradio<5.40` pinned (`wasm_utils` removed in `5.40`, breaking `fastrtc` import)
- Shared-memory bug: `Session` created per WebRTC connection, not shared across callers

### Removed
- `local_voice_chat.py` original implementation (replaced by `vaak.cli`)

### Upstream
- Forked from [jesuscopado/local-voice-ai-agent](https://github.com/jesuscopado/local-voice-ai-agent) (MIT)
- LICENSE and upstream attribution preserved

---

*Previous entries (pre-fork history):*
- Flattened layout
- Removed redundant script and duplicate README; added pyyaml
