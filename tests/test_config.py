"""Tests for configuration management (src/config.py)."""

import os
import pytest
from unittest.mock import patch

from src.config import AgentConfig, validate_config, load_config


@pytest.fixture(autouse=True)
def isolate_env_file(monkeypatch, tmp_path):
    """Prevent tests from reading the project .env file."""
    monkeypatch.chdir(tmp_path)


class TestAgentConfig:
    """Tests for AgentConfig class."""

    def test_loads_from_env_with_prefix(self, monkeypatch):
        """Config loads values from PROPHET_ prefixed env vars."""
        monkeypatch.setenv("PROPHET_ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("PROPHET_TAVILY_API_KEY", "tvly-test")
        monkeypatch.setenv("PROPHET_PORT", "9090")
        monkeypatch.setenv("PROPHET_MAX_CONCURRENCY", "5")

        config = AgentConfig()

        assert config.anthropic_api_key == "sk-ant-test"
        assert config.tavily_api_key == "tvly-test"
        assert config.port == 9090
        assert config.max_concurrency == 5

    def test_default_values(self, monkeypatch):
        """Config uses correct defaults when env vars are not set."""
        # Clear any existing PROPHET_ vars
        for key in list(os.environ.keys()):
            if key.startswith("PROPHET_"):
                monkeypatch.delenv(key, raising=False)

        config = AgentConfig()

        assert config.port == 8080
        assert config.host == "0.0.0.0"
        assert config.primary_model == "claude-sonnet-4-20250514"
        assert config.budget_model == "gpt-4o-mini"
        assert config.max_concurrency == 10
        assert config.max_searches_per_event == 3
        assert config.max_llm_calls_per_event == 5
        assert config.response_timeout_seconds == 570
        assert config.per_event_timeout_seconds == 480
        assert config.total_budget_usd == 50.0
        assert config.budget_alert_threshold == 0.90
        assert config.cache_ttl_hours == 6
        assert config.shrinkage_factor == 0.15
        assert config.platt_coefficient == 1.732
        assert config.evidence_recency_days == 90
        assert config.min_corroboration == 1

    def test_openai_key_optional(self, monkeypatch):
        """OpenAI API key is optional and defaults to None."""
        for key in list(os.environ.keys()):
            if key.startswith("PROPHET_"):
                monkeypatch.delenv(key, raising=False)

        config = AgentConfig()
        assert config.openai_api_key is None

    def test_openai_key_loaded_when_set(self, monkeypatch):
        """OpenAI API key is loaded when provided."""
        monkeypatch.setenv("PROPHET_OPENAI_API_KEY", "sk-openai-test")

        config = AgentConfig()
        assert config.openai_api_key == "sk-openai-test"


class TestValidateConfig:
    """Tests for startup validation."""

    def test_passes_with_valid_keys(self, monkeypatch):
        """Validation passes when both required keys are present."""
        monkeypatch.setenv("PROPHET_ANTHROPIC_API_KEY", "sk-ant-valid")
        monkeypatch.setenv("PROPHET_TAVILY_API_KEY", "tvly-valid")

        config = AgentConfig()
        # Should not raise or exit
        validate_config(config)

    def test_exits_when_anthropic_key_missing(self, monkeypatch):
        """Exits with code 1 when anthropic_api_key is empty."""
        for key in list(os.environ.keys()):
            if key.startswith("PROPHET_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("PROPHET_TAVILY_API_KEY", "tvly-valid")

        config = AgentConfig()
        with pytest.raises(SystemExit) as exc_info:
            validate_config(config)
        assert exc_info.value.code == 1

    def test_exits_when_tavily_key_missing(self, monkeypatch):
        """Exits with code 1 when tavily_api_key is empty."""
        for key in list(os.environ.keys()):
            if key.startswith("PROPHET_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("PROPHET_ANTHROPIC_API_KEY", "sk-ant-valid")

        config = AgentConfig()
        with pytest.raises(SystemExit) as exc_info:
            validate_config(config)
        assert exc_info.value.code == 1

    def test_exits_when_both_keys_missing(self, monkeypatch):
        """Exits with code 1 when both required keys are empty."""
        for key in list(os.environ.keys()):
            if key.startswith("PROPHET_"):
                monkeypatch.delenv(key, raising=False)

        config = AgentConfig()
        with pytest.raises(SystemExit) as exc_info:
            validate_config(config)
        assert exc_info.value.code == 1

    def test_logs_missing_variable_names(self, monkeypatch, caplog):
        """Logs error messages indicating which variables are missing."""
        for key in list(os.environ.keys()):
            if key.startswith("PROPHET_"):
                monkeypatch.delenv(key, raising=False)

        config = AgentConfig()

        import logging
        with caplog.at_level(logging.ERROR):
            with pytest.raises(SystemExit):
                validate_config(config)

        assert "PROPHET_ANTHROPIC_API_KEY" in caplog.text
        assert "PROPHET_TAVILY_API_KEY" in caplog.text


class TestLoadConfig:
    """Tests for the load_config convenience function."""

    def test_returns_config_when_valid(self, monkeypatch):
        """Returns AgentConfig when all required keys are present."""
        monkeypatch.setenv("PROPHET_ANTHROPIC_API_KEY", "sk-ant-valid")
        monkeypatch.setenv("PROPHET_TAVILY_API_KEY", "tvly-valid")

        config = load_config()
        assert isinstance(config, AgentConfig)
        assert config.anthropic_api_key == "sk-ant-valid"
        assert config.tavily_api_key == "tvly-valid"

    def test_exits_when_keys_missing(self, monkeypatch):
        """Exits when required keys are missing."""
        for key in list(os.environ.keys()):
            if key.startswith("PROPHET_"):
                monkeypatch.delenv(key, raising=False)

        with pytest.raises(SystemExit) as exc_info:
            load_config()
        assert exc_info.value.code == 1
