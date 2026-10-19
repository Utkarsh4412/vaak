"""Unit tests for vaak.cli module."""

from unittest.mock import MagicMock, patch

from vaak.cli import _build_parser, main


def test_build_parser_defaults():
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.config is None
    assert args.model is None
    assert args.system_prompt is None
    assert args.max_tokens is None
    assert args.temperature is None
    assert args.top_p is None
    assert args.language is None
    assert args.stt_backend is None
    assert args.phone is False
    assert args.share is False
    assert args.server_name is None
    assert args.skip_voice_check is False
    assert args.log_level == "INFO"


def test_build_parser_custom_args():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--model",
            "llama3.2:3b",
            "--language",
            "hi",
            "--stt-backend",
            "faster_whisper",
            "--max-tokens",
            "300",
            "--temperature",
            "0.8",
            "--top-p",
            "0.95",
            "--share",
            "--phone",
            "--server-name",
            "0.0.0.0",
            "--skip-voice-check",
            "--log-level",
            "DEBUG",
        ]
    )
    assert args.model == "llama3.2:3b"
    assert args.language == "hi"
    assert args.stt_backend == "faster_whisper"
    assert args.max_tokens == 300
    assert args.temperature == 0.8
    assert args.top_p == 0.95
    assert args.share is True
    assert args.phone is True
    assert args.server_name == "0.0.0.0"
    assert args.skip_voice_check is True
    assert args.log_level == "DEBUG"


@patch("vaak.cli.build_agent")
@patch("vaak.cli.attach_app_endpoints")
def test_main_web_flow(mock_attach, mock_build_agent):
    mock_handler = MagicMock()
    mock_stream = MagicMock()
    mock_stream.ui.app = MagicMock()
    mock_handler.build_stream.return_value = mock_stream
    mock_build_agent.return_value = mock_handler

    main(["--model", "llama3.2:1b", "--language", "en", "--log-level", "WARNING"])

    assert mock_build_agent.called
    assert mock_handler.build_stream.called
    assert mock_attach.called
    assert mock_stream.ui.launch.called


@patch("vaak.cli.build_agent")
def test_main_phone_flow(mock_build_agent):
    mock_handler = MagicMock()
    mock_stream = MagicMock()
    mock_handler.build_stream.return_value = mock_stream
    mock_build_agent.return_value = mock_handler

    main(["--phone"])

    assert mock_stream.fastphone.called
    assert not mock_stream.ui.launch.called
