from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from src.config import RAW_DIR, Settings, ensure_data_dirs

logger = logging.getLogger(__name__)


def mocked_odds_response() -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "events": [
            _event(
                "match-001",
                "Bodo/Glimt",
                "Tromso",
                now + timedelta(hours=2),
                {
                    "match_winner": [1.55, 4.25, 5.80],
                    "ou_0_5": [1.06, 9.50],
                    "ou_1_5": [1.26, 4.10],
                    "ou_2_5": [1.92, 1.92],
                    "ou_3_5": [3.20, 1.38],
                    "btts": [1.95, 1.85],
                },
                league="Eliteserien",
                country="Norway",
            ),
            _event(
                "match-002",
                "Arsenal",
                "Chelsea",
                now + timedelta(hours=6),
                {
                    "match_winner": [2.10, 3.55, 3.35],
                    "ou_0_5": [1.08, 8.80],
                    "ou_1_5": [1.32, 3.55],
                    "ou_2_5": [2.05, 1.78],
                    "ou_3_5": [3.65, 1.30],
                    "btts": [2.02, 1.78],
                },
                league="Premier League",
                country="England",
            ),
            _event(
                "match-003",
                "Barcelona",
                "Sevilla",
                now + timedelta(hours=12),
                {
                    "match_winner": [2.85, 3.45, 2.62],
                    "ou_0_5": [1.07, 9.20],
                    "ou_1_5": [1.35, 3.35],
                    "ou_2_5": [2.18, 1.70],
                    "ou_3_5": [4.10, 1.25],
                    "btts": [1.98, 1.82],
                },
                league="La Liga",
                country="Spain",
            ),
            _event(
                "match-004",
                "Brann",
                "Viking",
                now + timedelta(hours=20),
                {
                    "match_winner": [2.28, 3.40, 3.10],
                    "ou_0_5": [1.10, 7.80],
                    "ou_1_5": [1.42, 3.00],
                    "ou_2_5": [2.30, 1.62],
                    "ou_3_5": [4.20, 1.23],
                    "btts": [2.12, 1.72],
                },
                league="Eliteserien",
                country="Norway",
            ),
            _event(
                "match-past",
                "Bodo/Glimt",
                "Viking",
                now - timedelta(hours=4),
                {"match_winner": [1.70, 3.90, 4.60]},
                status="finished",
            ),
            _event(
                "match-live",
                "Molde",
                "Brann",
                now - timedelta(minutes=10),
                {"match_winner": [2.20, 3.35, 3.30]},
                status="live",
            ),
        ]
    }


class NorskTippingClient:
    """Read-only client for documented Norsk Tipping odds endpoints."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.headers = {
            "Accept": "application/json",
            "X-Client-Id": settings.nt_app_name,
        }

    def fetch_odds(self, use_mock: bool = False, sport: str = "football") -> dict[str, Any]:
        if use_mock:
            logger.info("Using mocked Norsk Tipping odds response.")
            return mocked_odds_response()
        sport_id = self.resolve_sport_id(sport)
        events_payload = self._get_json(self._url(self.settings.nt_odds_endpoint.format(sportId=sport_id)))
        events = events_payload.get("eventList", events_payload.get("events", []))
        if not isinstance(events, list):
            raise RuntimeError("Norsk Tipping events response did not contain an event list.")

        enriched_events: list[dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            event_id = event.get("eventId") or event.get("id")
            if event_id:
                try:
                    markets_payload = self._get_json(
                        self._url(self.settings.nt_markets_endpoint.format(eventId=event_id))
                    )
                    markets = markets_payload.get("markets", [])
                    if isinstance(markets, list) and markets:
                        event["markets"] = markets
                except RuntimeError as exc:
                    logger.warning("Could not fetch markets for event %s: %s", event_id, exc)
            if "markets" not in event and event.get("mainMarket"):
                event["markets"] = [event["mainMarket"]]
            enriched_events.append(event)
            time.sleep(self.settings.nt_rate_limit_sleep_seconds)
        return {"eventList": enriched_events, "sportId": sport_id}

    def resolve_sport_id(self, sport: str) -> str:
        normalized = sport.strip().lower()
        if len(normalized) == 3 and normalized.isalpha():
            return normalized.upper()
        if normalized in {"football", "soccer", "fotball"}:
            return self.settings.nt_default_football_sport_id

        payload = self._get_json(self._url(self.settings.nt_sport_types_endpoint))
        sports = payload.get("sportTypeList", payload.get("sports", []))
        for item in sports:
            if not isinstance(item, dict):
                continue
            sport_id = str(item.get("sportId", ""))
            sport_name = str(item.get("sportName", "")).lower()
            if normalized in {sport_id.lower(), sport_name}:
                return sport_id
        raise RuntimeError(f"Could not resolve sport '{sport}' from Norsk Tipping /sportTypes.")

    def save_raw_response(self, payload: dict[str, Any], prefix: str = "nt_odds") -> Path:
        ensure_data_dirs()
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = RAW_DIR / f"{prefix}_{timestamp}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved raw response to %s", path)
        return path

    def _endpoint_is_placeholder(self) -> bool:
        return "replace-with-documented" in self.settings.nt_odds_endpoint

    def _url(self, endpoint: str) -> str:
        return self.settings.nt_base_url.rstrip("/") + "/" + endpoint.lstrip("/")

    def _get_json(self, url: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.settings.nt_max_retries + 1):
            try:
                with httpx.Client(timeout=self.settings.nt_timeout_seconds, headers=self.headers) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    if response.status_code == 204 or not response.content:
                        return {}
                    return response.json()
            except httpx.HTTPError as exc:
                last_error = exc
                wait_seconds = self.settings.nt_rate_limit_sleep_seconds * attempt
                logger.warning("GET %s attempt %s failed: %s", url, attempt, exc)
                time.sleep(wait_seconds)
        raise RuntimeError(f"Failed GET {url}: {last_error}")


def _event(
    event_id: str,
    home_team: str,
    away_team: str,
    kickoff_time: datetime,
    odds: dict[str, list[float]],
    status: str = "upcoming",
    league: str = "unknown",
    country: str | None = None,
) -> dict[str, Any]:
    markets: list[dict[str, Any]] = []
    if "match_winner" in odds:
        markets.append(
            {
                "id": f"{event_id}-1x2",
                "name": "Match Winner",
                "selections": [
                    {"id": f"{event_id}-home", "name": home_team, "odds": odds["match_winner"][0]},
                    {"id": f"{event_id}-draw", "name": "Draw", "odds": odds["match_winner"][1]},
                    {"id": f"{event_id}-away", "name": away_team, "odds": odds["match_winner"][2]},
                ],
            }
        )
    for key, line in (("ou_0_5", 0.5), ("ou_1_5", 1.5), ("ou_2_5", 2.5), ("ou_3_5", 3.5)):
        if key in odds:
            markets.append(
                {
                    "id": f"{event_id}-ou-{str(line).replace('.', '-')}",
                    "name": f"Over/Under {line:g}",
                    "selections": [
                        {"id": f"{event_id}-over-{line:g}", "name": f"Over {line:g}", "odds": odds[key][0]},
                        {"id": f"{event_id}-under-{line:g}", "name": f"Under {line:g}", "odds": odds[key][1]},
                    ],
                }
            )
    if "btts" in odds:
        markets.append(
            {
                "id": f"{event_id}-btts",
                "name": "BTTS",
                "selections": [
                    {"id": f"{event_id}-btts-yes", "name": "Yes", "odds": odds["btts"][0]},
                    {"id": f"{event_id}-btts-no", "name": "No", "odds": odds["btts"][1]},
                ],
            }
        )
    return {
        "id": event_id,
        "sport": "football",
        "league": league,
        "country": country,
        "homeTeam": home_team,
        "awayTeam": away_team,
        "kickoffTime": kickoff_time.isoformat(),
        "status": status,
        "markets": markets,
    }
