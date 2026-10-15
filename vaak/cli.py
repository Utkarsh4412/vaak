"""
vaak/cli.py — CLI entry point for the `vaak` command.

Registered in pyproject.toml as:
  [project.scripts]
  vaak = "vaak.cli:main"

Usage:
  uv run vaak [options]

All options mirror the original interface so existing
invocations keep working.
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from vaak.agent import attach_app_endpoints, build_agent
from vaak.config import load_config


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vaak",
        description=(
            "Local-first multilingual real-time voice agent. "
            "Streams audio via WebRTC: Moonshine STT -> Ollama LLM -> Kokoro TTS."
        ),
    )
    p.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help="Path to YAML config file (default: config.yaml in cwd).",
    )
    p.add_argument("--model", help="Override Ollama model name (e.g. llama3.2:1b).")
    p.add_argument(
        "--system-prompt",
        metavar="FILE",
        help="Path to a .txt file containing the system prompt.",
    )
    p.add_argument("--max-tokens", type=int, help="Max tokens for LLM generation.")
    p.add_argument("--temperature", type=float, help="Sampling temperature (0–2).")
    p.add_argument("--top-p", type=float, help="Nucleus sampling top-p (0–1).")
    p.add_argument(
        "--language",
        choices=["en", "en-gb", "hi", "hi-en", "es", "fr"],
        help="Session language (default: en).",
    )
    p.add_argument(
        "--stt-backend",
        choices=["moonshine", "faster_whisper"],
        help="STT backend to use (default: moonshine).",
    )
    p.add_argument("--phone", action="store_true", help="Use FastRTC phone interface.")
    p.add_argument("--share", action="store_true", help="Create a Gradio public link.")
    p.add_argument("--server-name", help="Gradio server name (e.g. 0.0.0.0).")
    p.add_argument(
        "--skip-voice-check",
        action="store_true",
        help="Skip Kokoro voice validation at startup (useful if model not downloaded).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: list[str] | None = None) -> None:
    """Entry point for `uv run vaak`."""
    args = _build_parser().parse_args(argv)

    # Configure loguru
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    # Build CLI overrides dict — only non-None values override the config
    cli_overrides: dict = {}
    if args.model:
        cli_overrides.setdefault("llm", {})["model"] = args.model
    if args.max_tokens is not None:
        cli_overrides.setdefault("llm", {})["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        cli_overrides.setdefault("llm", {})["temperature"] = args.temperature
    if args.top_p is not None:
        cli_overrides.setdefault("llm", {})["top_p"] = args.top_p
    if args.language:
        cli_overrides["language"] = args.language
    if args.stt_backend:
        cli_overrides.setdefault("stt", {})["backend"] = args.stt_backend
    if args.system_prompt:
        cli_overrides["system_prompt_file"] = args.system_prompt
    if args.share:
        cli_overrides["share"] = True
    if args.phone:
        cli_overrides["phone"] = True
    if args.server_name:
        cli_overrides["host"] = args.server_name
    if args.skip_voice_check:
        cli_overrides.setdefault("tts", {})["skip_voice_check"] = True
    if args.log_level:
        cli_overrides["log_level"] = args.log_level

    config = load_config(config_path=args.config, cli_overrides=cli_overrides)
    handler = build_agent(config)
    stream = handler.build_stream()

    if config.phone:
        logger.info("Starting phone interface…")
        stream.fastphone()
    else:
        logger.info(f"Starting web interface on {config.host}:{config.port} …")
        attach_app_endpoints(stream.ui.app, handler)
        launch_kwargs: dict = {
            "share": config.share,
            "server_name": config.host,
            "server_port": config.port,
        }
        stream.ui.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
