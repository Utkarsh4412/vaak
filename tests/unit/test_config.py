"""Unit tests for AgentConfig — validation rules and load_config merging."""

import pytest
from pydantic import ValidationError

from vaak.config import AgentConfig, LLMConfig, MemoryConfig, STTConfig, TTSConfig, load_config


class TestAgentConfigDefaults:
    def test_default_language_is_english(self):
        cfg = AgentConfig()
        assert cfg.language == "en"

    def test_default_llm_model(self):
        cfg = AgentConfig()
        assert cfg.llm.model == "llama3.2:1b"

    def test_default_stt_backend(self):
        cfg = AgentConfig()
        assert cfg.stt.backend == "moonshine"

    def test_default_memory_turns(self):
        cfg = AgentConfig()
        assert cfg.memory.memory_turns == 4

    def test_default_share_is_false(self):
        cfg = AgentConfig()
        assert cfg.share is False

    def test_default_vad_threshold(self):
        cfg = AgentConfig()
        assert cfg.vad_threshold == 0.5

    def test_default_min_silence_ms(self):
        cfg = AgentConfig()
        assert cfg.min_silence_ms == 500

    def test_default_max_concurrent_sessions(self):
        cfg = AgentConfig()
        assert cfg.max_concurrent_sessions == 4


class TestAgentConfigValidation:
    def test_invalid_language_rejected(self):
        with pytest.raises(ValidationError):
            AgentConfig(language="klingon")

    def test_invalid_log_level_rejected(self):
        with pytest.raises(ValidationError):
            AgentConfig(log_level="VERBOSE")

    def test_max_tokens_must_be_positive(self):
        with pytest.raises(ValidationError):
            AgentConfig(llm=LLMConfig(max_tokens=0))

    def test_max_tokens_upper_bound(self):
        with pytest.raises(ValidationError):
            AgentConfig(llm=LLMConfig(max_tokens=9999))

    def test_temperature_out_of_range(self):
        with pytest.raises(ValidationError):
            AgentConfig(llm=LLMConfig(temperature=3.0))

    def test_tts_speed_lower_bound(self):
        with pytest.raises(ValidationError):
            AgentConfig(tts=TTSConfig(speed=0.1))

    def test_memory_turns_non_negative(self):
        with pytest.raises(ValidationError):
            AgentConfig(memory=MemoryConfig(memory_turns=-1))

    def test_port_out_of_range(self):
        with pytest.raises(ValidationError):
            AgentConfig(port=99999)

    def test_vad_threshold_out_of_range(self):
        with pytest.raises(ValidationError):
            AgentConfig(vad_threshold=-0.1)
        with pytest.raises(ValidationError):
            AgentConfig(vad_threshold=1.1)

    def test_min_silence_ms_out_of_range(self):
        with pytest.raises(ValidationError):
            AgentConfig(min_silence_ms=50)
        with pytest.raises(ValidationError):
            AgentConfig(min_silence_ms=6000)

    def test_max_concurrent_sessions_must_be_positive(self):
        with pytest.raises(ValidationError):
            AgentConfig(max_concurrent_sessions=0)


class TestAgentConfigAcceptsValid:
    def test_all_supported_languages(self):
        for lang in ("en", "en-gb", "hi", "hi-en", "es", "fr"):
            cfg = AgentConfig(language=lang)
            assert cfg.language == lang

    def test_zero_memory_turns_allowed(self):
        cfg = AgentConfig(memory=MemoryConfig(memory_turns=0))
        assert cfg.memory.memory_turns == 0

    def test_faster_whisper_backend(self):
        cfg = AgentConfig(stt=STTConfig(backend="faster_whisper"))
        assert cfg.stt.backend == "faster_whisper"

    def test_vad_threshold_and_min_silence_valid(self):
        cfg = AgentConfig(vad_threshold=0.8, min_silence_ms=1000)
        assert cfg.vad_threshold == 0.8
        assert cfg.min_silence_ms == 1000


class TestLoadConfig:
    def test_returns_agent_config_with_defaults(self, tmp_path):
        cfg = load_config(config_path=tmp_path / "nonexistent.yaml")
        assert isinstance(cfg, AgentConfig)

    def test_yaml_file_overrides_default(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("language: hi\n")
        cfg = load_config(config_path=yaml_file)
        assert cfg.language == "hi"

    def test_cli_overrides_yaml(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("language: hi\n")
        cfg = load_config(config_path=yaml_file, cli_overrides={"language": "es"})
        assert cfg.language == "es"

    def test_nested_cli_override(self, tmp_path):
        cfg = load_config(
            config_path=tmp_path / "none.yaml",
            cli_overrides={"llm": {"model": "mistral:7b"}},
        )
        assert cfg.llm.model == "mistral:7b"

    def test_env_var_overrides_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VAAK_LANGUAGE", "fr")
        cfg = load_config(config_path=tmp_path / "none.yaml")
        assert cfg.language == "fr"

    def test_system_prompt_file_loaded(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("You are a custom assistant.")
        cfg = load_config(
            cli_overrides={"system_prompt_file": str(prompt_file)},
            config_path=tmp_path / "none.yaml",
        )
        assert cfg.system_prompt == "You are a custom assistant."

    def test_missing_system_prompt_file_falls_back(self, tmp_path):
        cfg = load_config(
            cli_overrides={"system_prompt_file": str(tmp_path / "missing.txt")},
            config_path=tmp_path / "none.yaml",
        )
        # Should not raise; falls back to inline default
        assert cfg.system_prompt  # not empty
