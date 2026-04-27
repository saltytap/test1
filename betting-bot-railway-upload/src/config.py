from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RECOMMENDATIONS_DIR = DATA_DIR / "recommendations"
HISTORICAL_DIR = DATA_DIR / "historical"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    nt_app_name: str = Field(default="betting-analysis-bot", alias="NT_APP_NAME")
    nt_base_url: str = Field(default="https://api.norsk-tipping.no/OddsenGameInfo/v1/api", alias="NT_BASE_URL")
    nt_odds_endpoint: str = Field(
        default="/events/{sportId}",
        alias="NT_ODDS_ENDPOINT",
    )
    nt_sport_types_endpoint: str = Field(default="/sportTypes", alias="NT_SPORT_TYPES_ENDPOINT")
    nt_markets_endpoint: str = Field(default="/markets/{eventId}", alias="NT_MARKETS_ENDPOINT")
    nt_default_football_sport_id: str = Field(default="FBL", alias="NT_DEFAULT_FOOTBALL_SPORT_ID")
    nt_timeout_seconds: float = Field(default=10.0, alias="NT_TIMEOUT_SECONDS")
    nt_max_retries: int = Field(default=3, alias="NT_MAX_RETRIES")
    nt_rate_limit_sleep_seconds: float = Field(default=0.5, alias="NT_RATE_LIMIT_SLEEP_SECONDS")

    football_stats_provider: str = Field(default="footystats", alias="FOOTBALL_STATS_PROVIDER")
    require_external_stats: bool = Field(default=True, alias="REQUIRE_EXTERNAL_STATS")
    market_only_data_quality: float = Field(default=0.60, alias="MARKET_ONLY_DATA_QUALITY")
    market_only_confidence_cap: float = Field(default=0.63, alias="MARKET_ONLY_CONFIDENCE_CAP")
    external_stats_file: str = Field(default="data/external_stats.json", alias="EXTERNAL_STATS_FILE")
    min_team_mapping_score: float = Field(default=0.82, alias="MIN_TEAM_MAPPING_SCORE")
    footystats_api_key: str | None = Field(default=None, alias="FOOTYSTATS_API_KEY")
    footystats_base_url: str = Field(default="https://api.football-data-api.com", alias="FOOTYSTATS_BASE_URL")
    footystats_cache_dir: str = Field(default="data/cache/footystats", alias="FOOTYSTATS_CACHE_DIR")
    footystats_cache_ttl_seconds: int = Field(default=21_600, alias="FOOTYSTATS_CACHE_TTL_SECONDS")
    footystats_season_map_json: str | None = Field(default=None, alias="FOOTYSTATS_SEASON_MAP_JSON")
    api_football_key: str | None = Field(default=None, alias="API_FOOTBALL_KEY")
    api_football_base_url: str = Field(default="https://v3.football.api-sports.io", alias="API_FOOTBALL_BASE_URL")
    api_football_recent_matches: int = Field(default=10, alias="API_FOOTBALL_RECENT_MATCHES")
    stats_timeout_seconds: float = Field(default=10.0, alias="STATS_TIMEOUT_SECONDS")
    stats_rate_limit_sleep_seconds: float = Field(default=0.25, alias="STATS_RATE_LIMIT_SLEEP_SECONDS")

    edge_threshold: float = Field(default=0.0075, alias="EDGE_THRESHOLD")
    expected_value_threshold: float = Field(default=0.015, alias="EXPECTED_VALUE_THRESHOLD")
    lean_expected_value_threshold: float = Field(default=0.01, alias="LEAN_EXPECTED_VALUE_THRESHOLD")
    lean_expected_value_ceiling: float = Field(default=0.015, alias="LEAN_EXPECTED_VALUE_CEILING")
    lean_confidence_threshold: float = Field(default=0.60, alias="LEAN_CONFIDENCE_THRESHOLD")
    max_expected_value: float = Field(default=0.10, alias="MAX_EXPECTED_VALUE")
    min_odds: float = Field(default=1.4, alias="MIN_ODDS")
    max_odds: float = Field(default=4.5, alias="MAX_ODDS")
    confidence_threshold: float = Field(default=0.52, alias="CONFIDENCE_THRESHOLD")
    data_quality_threshold: float = Field(default=0.60, alias="DATA_QUALITY_THRESHOLD")
    max_results: int = Field(default=20, alias="MAX_RESULTS")
    recommendation_history_file: str = Field(
        default="data/recommendations/historical_picks.jsonl",
        alias="RECOMMENDATION_HISTORY_FILE",
    )
    backtest_history_file: str = Field(default="data/historical/predictions.csv", alias="BACKTEST_HISTORY_FILE")

    bankroll: float = Field(default=10_000.0, alias="BANKROLL")
    staking_strategy: str = Field(default="fractional_kelly", alias="STAKING_STRATEGY")
    kelly_fraction: float = Field(default=0.25, alias="KELLY_FRACTION")
    flat_stake_pct: float = Field(default=0.01, alias="FLAT_STAKE_PCT")
    min_stake_pct: float = Field(default=0.001, alias="MIN_STAKE_PCT")
    max_stake_pct: float = Field(default=0.01, alias="MAX_STAKE_PCT")


def get_settings() -> Settings:
    load_dotenv(PROJECT_ROOT / ".env")
    return Settings()


def ensure_data_dirs() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, RECOMMENDATIONS_DIR, HISTORICAL_DIR):
        directory.mkdir(parents=True, exist_ok=True)
