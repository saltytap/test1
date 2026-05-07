from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.config import RAW_DIR, Settings, ensure_data_dirs
from src.data_models import Market, Match, Selection
from src.footystats_api import FootyStatsClient
from src.value_engine import implied_probability, normalize_market_margin


logger = logging.getLogger(__name__)

DEFAULT_FOOTYSTATS_LEAGUES: tuple[tuple[str, str], ...] = (
    ("England", "Premier League"),
    ("England", "Championship"),
    ("England", "League One"),
    ("England", "League Two"),
    ("Germany", "Bundesliga"),
    ("Germany", "2. Bundesliga"),
    ("Italy", "Serie A"),
    ("Italy", "Serie B"),
    ("Spain", "La Liga"),
    ("Spain", "Segunda Division"),
    ("France", "Ligue 1"),
    ("France", "Ligue 2"),
    ("Netherlands", "Eredivisie"),
    ("USA", "MLS"),
    ("Brazil", "Serie A"),
    ("Brazil", "Serie B"),
    ("Argentina", "Primera Division"),
    ("Norway", "Eliteserien"),
    ("Norway", "First Division"),
    ("Sweden", "Allsvenskan"),
    ("Sweden", "Superettan"),
    ("Denmark", "Superliga"),
    ("Finland", "Veikkausliiga"),
    ("Poland", "Ekstraklasa"),
    ("Czech Republic", "First League"),
    ("Romania", "Liga I"),
    ("Portugal", "Liga Portugal"),
    ("Portugal", "LigaPro"),
    ("Belgium", "First Division A"),
)


class FootyStatsMatchSource:
    """Read-only FootyStats match and odds source."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = FootyStatsClient(settings)

    def fetch_matches(
        self,
        *,
        window_hours: int,
        country: str | None = None,
        league: str | None = None,
        max_leagues: int | None = None,
    ) -> tuple[list[Match], dict[str, Any]]:
        if not self.client.configured():
            raise RuntimeError("FootyStats API key is missing; cannot run FootyStats-only analysis.")

        now = datetime.now(UTC)
        window_end = now + timedelta(hours=window_hours)
        selected_leagues = _league_targets(country, league)
        if max_leagues is not None:
            selected_leagues = selected_leagues[:max_leagues]

        raw_payload: dict[str, Any] = {
            "source": "footystats",
            "primary_endpoint": "todays-matches",
            "window_start": now.isoformat(),
            "window_end": window_end.isoformat(),
            "daily_matches": [],
            "leagues": [],
        }
        normalized_matches: list[Match] = []
        skipped_leagues: list[str] = []

        season_lookup = _league_lookup_by_season_id(self.client.league_list())
        daily_matches = self._fetch_daily_matches(now, window_end)
        seen_match_ids: set[str] = set()
        for raw_match in daily_matches:
            country_name, league_name = _daily_match_league(raw_match, season_lookup)
            match = _match_from_footystats(raw_match, country_name, league_name, now, window_end)
            if match is None:
                continue
            if country and match.country.lower() != country.lower():
                continue
            if league and league.lower() not in match.league.lower():
                continue
            if match.match_id in seen_match_ids:
                continue
            normalized_matches.append(match)
            seen_match_ids.add(match.match_id)

        raw_payload["daily_matches"] = daily_matches
        if normalized_matches:
            summary = {
                "leagues_requested": len(selected_leagues),
                "leagues_loaded": None,
                "leagues_skipped": skipped_leagues,
                "matches_loaded": len(daily_matches),
                "future_matches_in_window": len(normalized_matches),
                "window_start": now.isoformat(),
                "window_end": window_end.isoformat(),
                "source_endpoint": "todays-matches",
            }
            self.save_raw_response(raw_payload)
            return normalized_matches, summary

        for country_name, league_name in selected_leagues:
            league_payload = self.client._find_league(country_name, league_name) or self.client._find_league_from_nt_name(
                f"{country_name} - {league_name}"
            )
            season_id = _latest_season_id_from_payload(league_payload)
            if season_id is None:
                skipped_leagues.append(f"{country_name} - {league_name}")
                continue
            raw_matches = self.client.league_matches(season_id)
            raw_payload["leagues"].append(
                {
                    "country": country_name,
                    "league": league_name,
                    "season_id": season_id,
                    "matches_loaded": len(raw_matches),
                    "matches": raw_matches,
                }
            )
            for raw_match in raw_matches:
                match = _match_from_footystats(raw_match, country_name, league_name, now, window_end)
                if match is not None:
                    normalized_matches.append(match)

        summary = {
            "leagues_requested": len(selected_leagues),
            "leagues_loaded": len(raw_payload["leagues"]),
            "leagues_skipped": skipped_leagues,
            "matches_loaded": sum(item["matches_loaded"] for item in raw_payload["leagues"]),
            "future_matches_in_window": len(normalized_matches),
            "window_start": now.isoformat(),
            "window_end": window_end.isoformat(),
            "source_endpoint": "league-matches",
        }
        self.save_raw_response(raw_payload)
        return normalized_matches, summary

    def _fetch_daily_matches(self, now: datetime, window_end: datetime) -> list[dict[str, Any]]:
        raw_matches: list[dict[str, Any]] = []
        current_date = now.date()
        end_date = window_end.date()
        while current_date <= end_date:
            raw_matches.extend(self.client.todays_matches(date=current_date.isoformat(), timezone="Etc/UTC"))
            current_date += timedelta(days=1)
        return raw_matches

    def save_raw_response(self, payload: dict[str, Any]) -> Path:
        ensure_data_dirs()
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = RAW_DIR / f"footystats_matches_{timestamp}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Saved FootyStats raw response to %s", path)
        return path


def _league_targets(country: str | None, league: str | None) -> list[tuple[str, str]]:
    targets = list(DEFAULT_FOOTYSTATS_LEAGUES)
    if country:
        targets = [item for item in targets if item[0].lower() == country.lower()]
    if league:
        targets = [item for item in targets if item[1].lower() == league.lower() or league.lower() in item[1].lower()]
    return targets


def _latest_season_id_from_payload(league_payload: dict[str, Any] | None) -> int | None:
    if not league_payload:
        return None
    seasons = league_payload.get("season")
    if not isinstance(seasons, list) or not seasons:
        return None
    latest = max((item for item in seasons if isinstance(item, dict)), key=lambda item: int(item.get("year", 0)))
    try:
        return int(latest["id"])
    except (KeyError, TypeError, ValueError):
        return None


def _league_lookup_by_season_id(leagues: list[dict[str, Any]]) -> dict[int, tuple[str, str]]:
    lookup: dict[int, tuple[str, str]] = {}
    for league in leagues:
        country = str(league.get("country") or "FootyStats").strip() or "FootyStats"
        league_name = str(league.get("league_name") or league.get("name") or "Daily Matches").strip() or "Daily Matches"
        seasons = league.get("season")
        if not isinstance(seasons, list):
            continue
        for season in seasons:
            if not isinstance(season, dict):
                continue
            try:
                season_id = int(season["id"])
            except (KeyError, TypeError, ValueError):
                continue
            lookup[season_id] = (country, league_name)
    return lookup


def _daily_match_league(raw_match: dict[str, Any], season_lookup: dict[int, tuple[str, str]]) -> tuple[str, str]:
    competition_id = _int(raw_match.get("competition_id") or raw_match.get("season"))
    if competition_id is not None and competition_id in season_lookup:
        return season_lookup[competition_id]
    country = _text(raw_match, "country", "country_name", "competition_country") or "FootyStats"
    league = _text(raw_match, "league_name", "competition_name", "competition", "league", "season_name") or "Daily Matches"
    return country, league


def _match_from_footystats(
    raw_match: dict[str, Any],
    country: str,
    league: str,
    now: datetime,
    window_end: datetime,
) -> Match | None:
    kickoff_unix = _float(raw_match.get("date_unix"))
    if kickoff_unix is None:
        return None
    kickoff = datetime.fromtimestamp(kickoff_unix, UTC)
    if kickoff < now or kickoff > window_end:
        return None
    if str(raw_match.get("status", "")).lower() not in {"incomplete", "scheduled", "upcoming", ""}:
        return None

    home_team = str(raw_match.get("home_name") or "").strip()
    away_team = str(raw_match.get("away_name") or "").strip()
    match_id = str(raw_match.get("id") or "").strip()
    if not home_team or not away_team or not match_id:
        return None

    markets = _markets_from_footystats(raw_match, match_id, home_team, away_team)
    if not markets:
        return None

    return Match(
        match_id=f"footystats-{match_id}",
        sport="football",
        league=f"{country} - {league}",
        country=country,
        home_team=home_team,
        away_team=away_team,
        kickoff_time=kickoff,
        status="upcoming",
        data_source="footystats",
        markets=markets,
    )


def _markets_from_footystats(raw_match: dict[str, Any], match_id: str, home_team: str, away_team: str) -> list[Market]:
    markets: list[Market] = []
    markets.extend(
        _market(
            f"{match_id}-1x2",
            "Match Winner",
            "match_winner",
            [
                (f"{match_id}-home", home_team, "H", raw_match.get("odds_ft_1")),
                (f"{match_id}-draw", "Draw", "D", _first_value(raw_match, "odds_ft_x", "odds_ft_X")),
                (f"{match_id}-away", away_team, "A", raw_match.get("odds_ft_2")),
            ],
        )
    )
    for line, over_key, under_key in (
        (1.5, "odds_ft_over15", "odds_ft_under15"),
        (2.5, "odds_ft_over25", "odds_ft_under25"),
        (3.5, "odds_ft_over35", "odds_ft_under35"),
    ):
        markets.extend(
            _market(
                f"{match_id}-ou-{line:g}",
                f"Over/Under {line:g}",
                "over_under",
                [
                    (f"{match_id}-over-{line:g}", f"Over {line:g}", "O", raw_match.get(over_key)),
                    (f"{match_id}-under-{line:g}", f"Under {line:g}", "U", raw_match.get(under_key)),
                ],
            )
        )
    markets.extend(
        _market(
            f"{match_id}-btts",
            "BTTS",
            "btts",
            [
                (f"{match_id}-btts-yes", "Yes", "Y", raw_match.get("odds_btts_yes")),
                (f"{match_id}-btts-no", "No", "N", raw_match.get("odds_btts_no")),
            ],
        )
    )
    return markets


def _market(
    market_id: str,
    name: str,
    market_type: str,
    selections: list[tuple[str, str, str, Any]],
) -> list[Market]:
    parsed: list[Selection] = []
    for selection_id, selection_name, selection_value, odds_value in selections:
        odds = _float(odds_value)
        if odds is None or odds <= 1.01:
            return []
        parsed.append(
            Selection(
                selection_id=selection_id,
                name=selection_name,
                selection_value=selection_value,
                decimal_odds=odds,
                raw_implied_probability=implied_probability(odds),
            )
        )
    return [normalize_market_margin(Market(market_id=market_id, name=name, market_type=market_type, selections=parsed))]


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_value(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _text(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        cleaned = str(value).strip()
        if cleaned:
            return cleaned
    return None
