"""Configuration management for the Prophet Forecasting Agent.

Loads configuration from environment variables with the PROPHET_ prefix.
Validates required API keys at startup and exits with non-zero code if missing.
"""

import logging
import sys
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class AgentConfig(BaseSettings):
    """Configuration via environment variables with PROPHET_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="PROPHET_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    anthropic_api_key: str = ""
    tavily_api_key: str = ""
    openai_api_key: Optional[str] = None

    # Server
    port: int = 8080
    host: str = "0.0.0.0"

    # Model Selection
    primary_model: str = "anthropic/claude-sonnet-4"
    budget_model: str = "openai/gpt-4o-mini"

    # Processing Limits
    max_concurrency: int = 10
    max_searches_per_event: int = 3
    max_llm_calls_per_event: int = 5
    response_timeout_seconds: int = 570  # 9.5 minutes (30s safety margin)
    per_event_timeout_seconds: int = 480  # 8 minutes

    # Cost Management
    total_budget_usd: float = 50.0
    budget_alert_threshold: float = 0.90

    # Cache
    cache_ttl_hours: int = 6

    # Calibration
    shrinkage_factor: float = 0.15
    platt_coefficient: float = 1.732  # √3

    # Research
    evidence_recency_days: int = 90
    min_corroboration: int = 1

    # Market anchoring
    market_anchor_weight: float = 0.3  # How much to trust market prices (0-1)
    use_kalshi_prices: bool = True  # Fetch live Kalshi prices when not in input
    confidence_threshold: float = 0.05  # Min deviation from market to submit own prediction

    # Ensemble
    featherless_api_key: Optional[str] = None
    featherless_model: str = "Qwen/Qwen2.5-72B-Instruct"
    use_ensemble: bool = True  # Use multi-model ensemble

    # Ensemble models (all via OpenRouter for simplicity)
    ensemble_model_1: str = "anthropic/claude-sonnet-4"
    ensemble_model_2: str = "google/gemini-3.1-pro-preview"
    ensemble_model_3: str = "openai/gpt-5"

    # Search fallbacks
    serper_api_key: Optional[str] = None  # Serper.dev Google search (2500 free)

    # Tavily advanced features
    use_advanced_search: bool = True  # Use advanced search depth for HIGH complexity
    use_news_topic: bool = True  # Use news topic for better date filtering


def validate_config(config: AgentConfig) -> None:
    """Validate that required API keys are present and non-empty.

    Logs an error for each missing key and exits with sys.exit(1) if any are missing.
    """
    missing = []

    if not config.anthropic_api_key:
        missing.append("PROPHET_ANTHROPIC_API_KEY")

    if not config.tavily_api_key:
        missing.append("PROPHET_TAVILY_API_KEY")

    if missing:
        for var in missing:
            logger.error(f"Required environment variable is missing or empty: {var}")
        sys.exit(1)


def load_config() -> AgentConfig:
    """Load and validate configuration. Exits if required keys are missing."""
    config = AgentConfig()
    validate_config(config)
    return config
