# Voice AI Agent

[![CI](https://github.com/Utkarsh4412/vaak/actions/workflows/ci.yml/badge.svg)](https://github.com/Utkarsh4412/vaak/actions/workflows/ci.yml)

An object-oriented, local voice chat agent built on FastRTC, Ollama, and Kokoro. 
This agent acts as a virtual conversational partner that runs 100% locally on your machine.

## Multilingual Support

vaak supports multi-turn conversations in English, Hindi, Hinglish, Spanish, and French with automatic voice selection and language routing:

| Language | Code | Default STT | Default TTS Voice | Notes |
|---|---|---|---|---|
| English (US) | `en` | Moonshine (fast path) | `af_heart` | Low-latency English baseline |
| English (UK) | `en-gb` | Moonshine | `bf_emma` | British English female voice |
| Hindi | `hi` | Faster-Whisper (`small` int8) | `hf_alpha` | Full Hindi STT + Kokoro Hindi TTS |
| Hinglish | `hi-en` | Moonshine / Faster-Whisper | `hf_alpha` | Conversational Latin-script Hinglish with Hindi TTS |
| Spanish | `es` | Faster-Whisper | `ef_dora` | Spanish female voice |
| French | `fr` | Faster-Whisper | `ff_siwis` | French female voice |

### Running in Hindi or Hinglish

```powershell
# Hindi conversation (Faster-Whisper STT + Kokoro Hindi voice)
uv run vaak --language hi

# Hinglish conversation (Natural Latin-script mix of Hindi & English)
uv run vaak --language hi-en
```

## Architecture Highlights
- **Object-Oriented Design**: Encapsulates state into a session handler to safely support multiple concurrent WebRTC stream sessions without memory bleed.
- **Streaming Pipeline & Barge-In**: Sentence-chunked token streaming ensures first audio plays before LLM reply finishes; mid-turn barge-in detection cancels TTS immediately.
- **Summarizing Memory**: Automatically condenses older conversation turns into structured summaries (`summarize_threshold`, default 6) to prevent context overflow.
- **Robust Configuration**: Supports overlapping configuration through defaults, YAML config files, and CLI arguments.
- **Performance Logging**: Implements precise `perf_counter` timing for STT, LLM TTFT, and TTS chunks.

## Setup

1. Install [uv](https://github.com/astral-sh/uv) and [Ollama](https://ollama.ai/).
2. Clone the repo and navigate to the directory.
3. Install dependencies:
   ```powershell
   uv sync
   ```
4. Download the base language model:
   ```powershell
   ollama pull llama3.2:1b
   ```

## Usage

```powershell
# Standard local web UI
uv run vaak

# Public sharing via Gradio
uv run vaak --share

# Phone interface (experimental)
uv run vaak --phone
```

### Advanced Configuration

You can provide a `config.yaml` file (see `config.example.yaml`) or use CLI flags to override behavior:

| Flag | Description |
|---|---|
| `--model` | Ollama model to use (default: llama3.2:1b) |
| `--config` | Path to YAML config |
| `--language` | Language code (`en`, `hi`, `hi-en`, `es`, `fr`) |
| `--system-prompt` | Path to system prompt txt |
| `--max-tokens` | Max completion tokens |
| `--temperature` | Sampling temperature |
| `--log-level` | Console logging verbosity |

## Development

Install dev dependencies and pre-commit hooks:
```powershell
uv sync
uv run pre-commit install
```

Lint:
```powershell
uv run ruff check .
```

Format:
```powershell
uv run ruff format .
```

Type-check:
```powershell
uv run mypy vaak
```

Run tests:
```powershell
uv run pytest
```

Run tests with coverage:
```powershell
uv run pytest --cov=vaak --cov-report=term-missing
```

## License
MIT License
