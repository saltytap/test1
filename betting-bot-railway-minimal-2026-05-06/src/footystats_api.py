from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from src.config import PROJECT_ROOT, Settings


logger = logging.getLogger(__name__)


HIGH_VALUE_LEAGUE_ALIASES = {
    "premier league": ("England", "Premier League"),
    "england premier league": ("England", "Premier League"),
    "england championship": ("England", "Championship"),
    "championship": ("England", "Championship"),
    "bundesliga": ("Germany", "Bundesliga"),
    "tyskland bundesliga": ("Germany", "Bundesliga"),
    "2 bundesliga": ("Germany", "2. Bundesliga"),
    "serie a italy": ("Italy", "Serie A"),
    "italy serie a": ("Italy", "Serie A"),
    "serie b italy": ("Italy", "Serie B"),
    "italy serie b": ("Italy", "Serie B"),
    "spain la liga": ("Spain", "La Liga"),
    "spania primera division": ("Spain", "La Liga"),
    "la liga": ("Spain", "La Liga"),
    "la liga 2": ("Spain", "Segunda División"),
    "spania segunda division": ("Spain", "Segunda División"),
    "ligue 1": ("France", "Ligue 1"),
    "frankrike ligue 1": ("France", "Ligue 1"),
    "ligue 2": ("France", "Ligue 2"),
    "frankrike ligue 2": ("France", "Ligue 2"),
    "eredivisie": ("Netherlands", "Eredivisie"),
    "nederland eredivisie": ("Netherlands", "Eredivisie"),
    "mls usa": ("USA", "MLS"),
    "usa mls": ("USA", "MLS"),
    "brasil serie a": ("Brazil", "Serie A"),
    "brazil serie a": ("Brazil", "Serie A"),
    "primera lpf argentina": ("Argentina", "Primera División"),
    "argentina primera": ("Argentina", "Primera División"),
    "norway eliteserien": ("Norway", "Eliteserien"),
    "eliteserien": ("Norway", "Eliteserien"),
    "norge obos ligaen": ("Norway", "First Division"),
    "norway obos": ("Norway", "First Division"),
    "sweden allsvenskan": ("Sweden", "Allsvenskan"),
    "sverige allsvenskan": ("Sweden", "Allsvenskan"),
    "sweden superettan": ("Sweden", "Superettan"),
    "sverige superettan": ("Sweden", "Superettan"),
    "denmark superliga": ("Denmark", "Superliga"),
    "superliga denmark": ("Denmark", "Superliga"),
    "finland veikkausliiga": ("Finland", "Veikkausliiga"),
    "polen ekstraklasa": ("Poland", "Ekstraklasa"),
    "poland ekstraklasa": ("Poland", "Ekstraklasa"),
    "czech first league": ("Czech Republic", "First League"),
    "romania liga 1": ("Romania", "Liga I"),
    "romania liga i": ("Romania", "Liga I"),
    "besta deild iceland": ("Iceland", "Úrvalsdeild"),
    "island besta deild": ("Iceland", "Úrvalsdeild"),
    "liga mx clausura mexico": ("Mexico", "Liga MX"),
    "mexico liga mx clausura": ("Mexico", "Liga MX"),
    "ecuador primera a": ("Ecuador", "Primera Categoria Serie A"),
    "chile primera division": ("Chile", "Primera División"),
    "uruguay primera division": ("Uruguay", "Primera División"),
    "liga dimayor colombia": ("Colombia", "Categoria Primera A"),
    "brasil serie b": ("Brazil", "Serie B"),
}


class FootyStatsClient:
    """Small cached client for the FootyStats / Football Data API."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.footystats_base_url.rstrip("/")
        self.cache_dir = _resolve_path(settings.footystats_cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._league_list: list[dict[str, Any]] | None = None
        self._season_cache: dict[str, int | None] = {}

    def configured(self) -> bool:
        return bool(self.settings.footystats_api_key)

    def league_is_supported(self, nt_league: str) -> bool:
        return self._league_alias(nt_league) is not None or self.resolve_season_id(nt_league) is not None

    def resolve_season_id(self, nt_league: str) -> int | None:
        if nt_league in self._season_cache:
            return self._season_cache[nt_league]

        configured = self._configured_season_map().get(nt_league)
        if configured is not None:
            self._season_cache[nt_league] = configured
            return configured

        alias = self._league_alias(nt_league)
        if alias is None:
            best = self._find_league_from_nt_name(nt_league)
        else:
            country, league_name = alias
            best = self._find_league(country, league_name)
        season_id = _latest_season_id(best) if best else None
        self._season_cache[nt_league] = season_id
        return season_id

    def league_teams(self, season_id: int) -> list[dict[str, Any]]:
        teams: list[dict[str, Any]] = []
        page = 1
        while True:
            payload = self.get(
                "/league-teams",
                {"season_id": season_id, "include": "stats", "page": page},
            )
            page_teams = _extract_data_array(payload)
            teams.extend(page_teams)
            if len(page_teams) < 50:
                break
            page += 1
            time.sleep(self.settings.stats_rate_limit_sleep_seconds)
        return teams

    def league_matches(self, season_id: int) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        page = 1
        while True:
            payload = self.get(
                "/league-matches",
                {"season_id": season_id, "page": page, "max_per_page": 500},
            )
            page_matches = _extract_data_array(payload)
            matches.extend(page_matches)
            pager = payload.get("pager") if isinstance(payload, dict) else None
            max_page = int(pager.get("max_page", page)) if isinstance(pager, dict) else page
            if page >= max_page or len(page_matches) == 0:
                break
            page += 1
            time.sleep(self.settings.stats_rate_limit_sleep_seconds)
        return matches

    def todays_matches(self, *, date: str, timezone: str = "Etc/UTC") -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        page = 1
        while True:
            payload = self.get(
                "/todays-matches",
                {"date": date, "timezone": timezone, "page": page},
            )
            page_matches = _extract_data_array(payload)
            matches.extend(page_matches)
            pager = payload.get("pager") if isinstance(payload, dict) else None
            max_page = int(pager.get("max_page", page)) if isinstance(pager, dict) else page
            if page >= max_page or len(page_matches) == 0:
                break
            page += 1
            time.sleep(self.settings.stats_rate_limit_sleep_seconds)
        return matches

    def get(self, endpoint: str, params: dict[str, object] | None = None) -> dict[str, Any]:
        if not self.settings.footystats_api_key:
            return {}
        request_params = {"key": self.settings.footystats_api_key, **(params or {})}
        cache_path = self._cache_path(endpoint, request_params)
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_error: Exception | None = None
        for attempt in range(1, 4):
            try:
                with httpx.Client(timeout=self.settings.stats_timeout_seconds) as client:
                    response = client.get(url, params=request_params)
                    response.raise_for_status()
                    payload = response.json()
                    self._write_cache(cache_path, payload)
                    return payload
            except (httpx.HTTPError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(self.settings.stats_rate_limit_sleep_seconds * attempt)
        error_message = str(last_error)
        if self.settings.footystats_api_key:
            error_message = error_message.replace(self.settings.footystats_api_key, "<redacted>")
        logger.warning("FootyStats request failed for %s: %s", endpoint, error_message)
        return {}

    def league_list(self) -> list[dict[str, Any]]:
        if self._league_list is None:
            self._league_list = _extract_data_array(self.get("/league-list"))
        return self._league_list

    def _find_league(self, country: str, league_name: str) -> dict[str, Any] | None:
        normalized_country = _normalize(country)
        normalized_league = _normalize(league_name)
        for league in self.league_list():
            if _normalize(str(league.get("country", ""))) == normalized_country and _normalize(
                str(league.get("league_name", ""))
            ) == normalized_league:
                return league
        for league in self.league_list():
            if _normalize(str(league.get("country", ""))) == normalized_country and normalized_league in _normalize(
                str(league.get("name", ""))
            ):
                return league
        return None

    def _find_league_from_nt_name(self, nt_league: str) -> dict[str, Any] | None:
        normalized = _normalize(nt_league)
        if "internasjonal" in normalized or "world cup" in normalized or "vm" in normalized:
            return None
        parts = [part.strip() for part in nt_league.split(" - ") if part.strip()]
        league_hint = _normalize(parts[0]) if parts else normalized
        country_hint = _normalize(parts[-1]) if len(parts) > 1 else ""
        best: dict[str, Any] | None = None
        best_score = 0.0
        for league in self.league_list():
            country = _normalize(str(league.get("country", "")))
            league_names = [
                _normalize(str(league.get("league_name", ""))),
                _normalize(str(league.get("name", ""))),
            ]
            country_score = 0.12 if country_hint and (country_hint == country or country_hint in country or country in country_hint) else 0.0
            for league_name in league_names:
                if not league_name:
                    continue
                league_score = max(_similarity(league_hint, league_name), _similarity(normalized, league_name))
                score = min(league_score + country_score, 1.0)
                if score > best_score:
                    best = league
                    best_score = score
        if best_score >= 0.78:
            logger.info("Resolved FootyStats league by fuzzy match: %s score=%.2f", nt_league, best_score)
            return best
        return None

    def _league_alias(self, nt_league: str) -> tuple[str, str] | None:
        normalized = _normalize(nt_league)
        if normalized in HIGH_VALUE_LEAGUE_ALIASES:
            return HIGH_VALUE_LEAGUE_ALIASES[normalized]
        for key, alias in HIGH_VALUE_LEAGUE_ALIASES.items():
            if key in normalized or normalized in key:
                return alias
        return None

    def _configured_season_map(self) -> dict[str, int]:
        raw = self.settings.footystats_season_map_json
        if not raw:
            return {}
        try:
            return {str(key): int(value) for key, value in json.loads(raw).items()}
        except (TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Invalid FOOTYSTATS_SEASON_MAP_JSON; ignoring configured map.")
            return {}

    def _cache_path(self, endpoint: str, params: dict[str, object]) -> Path:
        redacted_params = {key: value for key, value in params.items() if key != "key"}
        cache_key = json.dumps({"endpoint": endpoint, "params": redacted_params}, sort_keys=True)
        digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        if time.time() - path.stat().st_mtime > self.settings.footystats_cache_ttl_seconds:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _write_cache(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")


def _extract_data_array(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("data", "teams", "response"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _latest_season_id(league: dict[str, Any] | None) -> int | None:
    if not league:
        return None
    seasons = league.get("season")
    if not isinstance(seasons, list) or not seasons:
        return None
    latest = max((item for item in seasons if isinstance(item, dict)), key=lambda item: int(item.get("year", 0)))
    try:
        return int(latest["id"])
    except (KeyError, TypeError, ValueError):
        return None


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _normalize(value: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in value)
    stopwords = {"fc", "cf", "fk", "bk", "sk", "ac", "sc", "club", "de", "the"}
    return " ".join(part for part in cleaned.split() if part not in stopwords)


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    overlap = len(left_tokens & right_tokens) / max(len(left_tokens), len(right_tokens), 1)
    containment = 0.92 if len(left) >= 4 and len(right) >= 4 and (left in right or right in left) else 0.0
    try:
        from difflib import SequenceMatcher

        sequence = SequenceMatcher(None, left, right).ratio()
    except Exception:
        sequence = 0.0
    return max(sequence, containment, overlap * 0.95)
