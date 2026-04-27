from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import DATA_DIR, Settings, ensure_data_dirs, get_settings
from src.data_models import Prediction
from src.footystats_source import FootyStatsMatchSource
from src.main import DEFAULT_MARKETS, analyze_matches


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

LATEST_PICKS_PATH = DATA_DIR / "latest_picks.json"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


def _format_ev(expected_value: float) -> str:
    prefix = "+" if expected_value > 0 else ""
    return f"{prefix}{expected_value * 100:.1f}%"


def _bet_type(prediction: Prediction) -> str:
    market = prediction.market.lower()
    if "over/under" in market or "total goals" in market or "totalt antall" in market:
        return prediction.selection
    if "btts" in market or "both teams" in market or "begge lag" in market:
        return "BTTS"
    return prediction.market


def clean_pick(prediction: Prediction, now: datetime | None = None) -> dict[str, Any] | None:
    current_time = now or datetime.now(UTC)
    kickoff = prediction.kickoff_time
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=UTC)
    if kickoff <= current_time or prediction.expected_value <= 0:
        return None
    return {
        "match": prediction.match,
        "bet_type": _bet_type(prediction),
        "pick": prediction.selection,
        "odds": round(prediction.odds, 2),
        "ev": _format_ev(prediction.expected_value),
    }


def export_latest_picks(
    recommendations: list[Prediction],
    output_path: Path = LATEST_PICKS_PATH,
    now: datetime | None = None,
) -> dict[str, Any]:
    ensure_data_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    picks = [pick for item in recommendations if (pick := clean_pick(item, now=now)) is not None]
    payload = {
        "last_updated": datetime.now(UTC).isoformat(),
        "picks": picks,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_daily(settings: Settings | None = None, output_path: Path = LATEST_PICKS_PATH) -> dict[str, Any]:
    settings = settings or get_settings()
    if not settings.footystats_api_key:
        raise RuntimeError("FOOTYSTATS_API_KEY is required")

    window_hours = _env_int("BETTING_BOT_WINDOW_HOURS", 24)
    max_leagues = _env_int("BETTING_BOT_MAX_LEAGUES", 50)
    max_results = _env_int("BETTING_BOT_MAX_RESULTS", 30)

    source = FootyStatsMatchSource(settings)
    matches, source_summary = source.fetch_matches(window_hours=window_hours, max_leagues=max_leagues)
    if not matches:
        raise RuntimeError("FootyStats returned no upcoming matches with usable odds")

    recommendations, summary = analyze_matches(
        matches=matches,
        markets=DEFAULT_MARKETS,
        max_results=max_results,
        country=None,
        league=None,
        sport="football",
        min_ev=settings.expected_value_threshold,
