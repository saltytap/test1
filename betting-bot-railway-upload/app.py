from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import DATA_DIR


DEFAULT_ALLOWED_ORIGINS = "https://bettrack.org,http://localhost:3000"
DEFAULT_PICKS_PATH = DATA_DIR / "latest_picks.json"
APP_VERSION = "2026-04-27-btts-refresh-v2"


def _allowed_origins() -> list[str]:
    raw = os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _load_picks(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"last_updated": None, "picks": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"last_updated": None, "picks": []}
    picks = payload.get("picks")
    if not isinstance(picks, list):
        picks = []
    return {
        "last_updated": payload.get("last_updated"),
        "picks": picks,
    }


def _should_refresh(path: Path) -> bool:
    if not path.exists():
        return True
    payload = _load_picks(path)
    return not payload.get("picks")


def _refresh_if_needed(path: Path) -> None:
    if not _should_refresh(path) or not os.getenv("FOOTYSTATS_API_KEY"):
        return
    try:
        from run_daily import run_daily

        run_daily(output_path=path)
    except Exception:
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
        }

    return app


app = create_app()
