from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import DATA_DIR
from src.parlays import build_fun_parlays


DEFAULT_ALLOWED_ORIGINS = "https://bettrack.org,http://localhost:3000"
DEFAULT_PICKS_PATH = DATA_DIR / "latest_picks.json"
APP_VERSION = "2026-04-28-fun-parlays-v1"
LAST_REFRESH_ERROR: str | None = None


def _allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _load_picks(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"last_updated": None, "picks": [], "parlays": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"last_updated": None, "picks": [], "parlays": []}
    picks = payload.get("picks")
    if not isinstance(picks, list):
        picks = []
    parlays = payload.get("parlays")
    if not isinstance(parlays, list):
        parlays = []
    return {
        "last_updated": payload.get("last_updated"),
        "picks": picks,
        "parlays": parlays,
    }


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _format_ev(expected_value: float) -> str:
    prefix = "+" if expected_value > 0 else ""
    return f"{prefix}{expected_value * 100:.1f}%"


def _bet_type(market: str, selection: str) -> str:
    normalized = market.lower()
    if "over/under" in normalized or "total goals" in normalized or "totalt antall" in normalized:
        return selection
    if "btts" in normalized or "both teams" in normalized or "begge lag" in normalized:
        return "BTTS"
    return market


def _generate_latest_picks(path: Path) -> None:
    from src.config import get_settings
    from src.footystats_source import FootyStatsMatchSource
    from src.main import DEFAULT_MARKETS, analyze_matches

    settings = get_settings()
    if not settings.footystats_api_key:
        raise RuntimeError("FOOTYSTATS_API_KEY is required")

    window_hours = _env_int("BETTING_BOT_WINDOW_HOURS", 24)
    max_leagues = _env_int("BETTING_BOT_MAX_LEAGUES", 50)
    max_results = _env_int("BETTING_BOT_MAX_RESULTS", 30)

    source = FootyStatsMatchSource(settings)
    matches, _summary = source.fetch_matches(window_hours=window_hours, max_leagues=max_leagues)
    if not matches:
        raise RuntimeError("FootyStats returned no upcoming matches with usable odds")

    recommendations, _analysis_summary = analyze_matches(
        matches=matches,
        markets=DEFAULT_MARKETS,
        max_results=max_results,
        country=None,
        league=None,
        sport="football",
        min_ev=settings.expected_value_threshold,
        settings=settings,
        production_mode=True,
        debug=False,
        window_hours=window_hours,
        max_window_hours=window_hours,
        include_secondary_window=False,
    )
    now = datetime.now(UTC)
    picks: list[dict[str, Any]] = []
    for item in recommendations:
        kickoff = item.kickoff_time
        if kickoff.tzinfo is None:
            kickoff = kickoff.replace(tzinfo=UTC)
        if kickoff <= now or item.expected_value <= 0:
            continue
        picks.append(
            {
                "match": item.match,
                "bet_type": _bet_type(item.market, item.selection),
                "pick": item.selection,
                "odds": round(item.odds, 2),
                "ev": _format_ev(item.expected_value),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "last_updated": datetime.now(UTC).isoformat(),
                "picks": picks,
                "parlays": build_fun_parlays(recommendations),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _should_refresh(path: Path) -> bool:
    if not path.exists():
        return True
    payload = _load_picks(path)
    return not payload.get("picks")


def _refresh_if_needed(path: Path) -> None:
    global LAST_REFRESH_ERROR
    if not _should_refresh(path) or not os.getenv("FOOTYSTATS_API_KEY"):
        return
    try:
        _generate_latest_picks(path)
        LAST_REFRESH_ERROR = None
    except Exception as exc:
        LAST_REFRESH_ERROR = f"{type(exc).__name__}: {exc}"
        # Keep the public API safe for BetTrack even if the data provider is down.
        return


def create_app(picks_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="BetTrack Picks API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins(),
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    path = picks_path or Path(os.getenv("LATEST_PICKS_PATH", DEFAULT_PICKS_PATH))

    @app.get("/api/picks")
    def get_picks() -> dict[str, Any]:
        _refresh_if_needed(path)
        return _load_picks(path)

    @app.get("/api/status")
    def get_status() -> dict[str, Any]:
        payload = _load_picks(path)
        return {
            "version": APP_VERSION,
            "footystats_key_configured": bool(os.getenv("FOOTYSTATS_API_KEY")),
            "window_hours": os.getenv("BETTING_BOT_WINDOW_HOURS", "24"),
            "max_leagues": os.getenv("BETTING_BOT_MAX_LEAGUES", "50"),
            "max_results": os.getenv("BETTING_BOT_MAX_RESULTS", "30"),
            "picks_file_exists": path.exists(),
            "last_updated": payload.get("last_updated"),
            "pick_count": len(payload.get("picks", [])),
            "last_refresh_error": LAST_REFRESH_ERROR,
        }

    return app


app = create_app()
