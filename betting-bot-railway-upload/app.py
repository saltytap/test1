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
        return _load_picks(path)

    return app


app = create_app()
