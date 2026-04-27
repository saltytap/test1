from __future__ import annotations

import json
import logging
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import httpx

from src.config import PROJECT_ROOT, Settings
from src.data_models import Match, TeamStats
from src.footystats_api import FootyStatsClient


logger = logging.getLogger(__name__)

PARTIAL_MAPPING_FLOOR = 0.68
try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - exercised when optional package is absent
    fuzz = None

MANUAL_TEAM_ALIASES = {
    "ac milan": "milan",
    "milan": "milan",
    "inter milan": "internazionale",
    "inter": "internazionale",
    "as roma": "roma",
    "ssc napoli": "napoli",
    "atalanta bc": "atalanta",
    "ss lazio": "lazio",
    "us sassuolo calcio": "sassuolo",
    "sassuolo calcio": "sassuolo",
    "us lecce": "lecce",
    "genoa cfc": "genoa",
    "ogc nice": "nice",
    "lille osc": "lille",
    "fsv mainz": "mainz 05",
    "1 fsv mainz 05": "mainz 05",
    "rb leipzig": "r b leipzig",
    "rasenballsport leipzig": "r b leipzig",
    "fc koln": "koln",
    "1 fc koln": "koln",
    "cologne": "koln",
    "west bromwich": "west bromwich albion",
    "west bromwich albion fc": "west bromwich albion",
    "hull city": "hull city",
    "coventry": "coventry city",
    "coventry city fc": "coventry city",
    "red bull bragantino sp": "bragantino",
    "red bull bragantino": "bragantino",
    "atletico mineiro mg": "atletico mineiro",
    "fluminense fc rj": "fluminense",
    "atletico tucuman": "atletico tucuman",
    "los angeles galaxy": "la galaxy",
    "la galaxy": "la galaxy",
    "real salt lake": "real salt lake",
    "aik": "aik",
    "ik sirius": "sirius",
    "vejle bk": "vejle",
    "hjk helsinki": "hjk",
    "mks arka gdynia": "arka gdynia",
    "espanyol": "rcd espanyol",
    "alaves": "deportivo alaves",
    "getafe": "getafe cf",
    "real betis": "real betis",
    "cagliari": "cagliari",
    "udinese": "udinese",
    "bologna": "bologna",
}


@dataclass(frozen=True)
class MatchStatsResult:
    stats: dict[str, TeamStats]
    has_full_data: bool
    matched_teams: int
    required_teams: int = 2
    missing_reasons: list[str] | None = None
    unmatched_teams: list[str] | None = None


class FootballStatsProvider:
    """External football stats facade.

    Supported providers:
    - footystats: FootyStats / Football Data API.
    - api_football: API-Football / API-Sports read-only API.
    - json: provider-neutral local JSON stats.
    - none: disabled.

    If no external stats are available for both teams, analysis skips the match
    instead of inventing data.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.provider = settings.football_stats_provider.lower()
        self.footystats = FootyStatsProvider(settings)
        self.local_json = LocalJsonStatsProvider(settings)
        self.api_football = ApiFootballStatsProvider(settings)

    def get_match_stats(self, match: Match) -> MatchStatsResult:
        if self.provider in {"footystats", "football-data-api"}:
            return self.footystats.get_match_stats(match)
        if self.provider in {"api_football", "apifootball", "api-sports"}:
            return self.api_football.get_match_stats(match)
        if self.provider in {"json", "local_json"}:
            return self.local_json.get_match_stats(match)
        if self.provider in {"none", "disabled", ""}:
            return _missing_result(match)
        logger.warning("Unknown FOOTBALL_STATS_PROVIDER=%s; no external stats used.", self.provider)
        return _missing_result(match)


class FootyStatsProvider:
    """Maps Norsk Tipping matches to FootyStats league/team season stats."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = FootyStatsClient(settings)
        self._league_team_cache: dict[int, list[dict[str, Any]]] = {}
        self._mapping_cache_path = _resolve_stats_path(settings.footystats_cache_dir) / "team_mappings.json"
        self._mapping_cache = self._load_mapping_cache()

    def get_match_stats(self, match: Match) -> MatchStatsResult:
        if not self.client.configured():
            return _missing_result(match, ["footystats_key_missing"])
        if not self.client.league_is_supported(match.league):
            return _missing_result(match, [f"unsupported_league:{match.league}"])

        season_id = self.client.resolve_season_id(match.league)
        if season_id is None:
            return _missing_result(match, [f"no_footystats_season_mapping:{match.league}"])

        teams = self._league_teams(season_id)
        home_raw, home_score = self._map_team(match.home_team, match.league, season_id, teams)
        away_raw, away_score = self._map_team(match.away_team, match.league, season_id, teams)
        unmatched: list[str] = []
        if home_raw is None or home_score < PARTIAL_MAPPING_FLOOR:
            unmatched.append(match.home_team)
        if home_raw is not None and home_score < self.settings.min_team_mapping_score:
            logger.info(
                "Using partial FootyStats mapping for %s: %s score=%.2f",
                match.match_id,
                match.home_team,
                home_score,
            )
        if away_raw is None or away_score < PARTIAL_MAPPING_FLOOR:
            unmatched.append(match.away_team)
        if away_raw is not None and away_score < self.settings.min_team_mapping_score:
            logger.info(
                "Using partial FootyStats mapping for %s: %s score=%.2f",
                match.match_id,
                match.away_team,
                away_score,
            )
        league_profile = _league_profile_from_teams(teams)
        if unmatched:
            logger.info("Unmatched FootyStats teams for %s: %s", match.match_id, ", ".join(unmatched))

        stats = {
            match.home_team: _team_stats_from_footystats(match.home_team, home_raw, home_score)
            if home_raw is not None and home_score >= PARTIAL_MAPPING_FLOOR
            else _fallback_team_stats(match.home_team, league_profile, "mapping_failure"),
            match.away_team: _team_stats_from_footystats(match.away_team, away_raw, away_score)
            if away_raw is not None and away_score >= PARTIAL_MAPPING_FLOOR
            else _fallback_team_stats(match.away_team, league_profile, "mapping_failure"),
        }
        result = _validate_match_stats(match, stats)
        if unmatched:
            return MatchStatsResult(
                stats=result.stats,
                has_full_data=False,
                matched_teams=result.matched_teams,
                missing_reasons=["mapping_failure"],
                unmatched_teams=unmatched,
            )
        return result

    def _league_teams(self, season_id: int) -> list[dict[str, Any]]:
        if season_id not in self._league_team_cache:
            self._league_team_cache[season_id] = self.client.league_teams(season_id)
        return self._league_team_cache[season_id]

    def _map_team(
        self,
        team_name: str,
        league_name: str,
        season_id: int,
        teams: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, float]:
        cache_key = f"{season_id}:{_normalize_name(team_name)}"
        cached_id = self._mapping_cache.get(cache_key)
        if cached_id is not None:
            for team in teams:
                if str(team.get("id")) == str(cached_id):
                    return team, 1.0
        raw, score = _best_team_match(team_name, teams)
        if raw is not None and score >= PARTIAL_MAPPING_FLOOR and raw.get("id") is not None:
            self._mapping_cache[cache_key] = str(raw["id"])
            self._save_mapping_cache()
        elif raw is None or score < PARTIAL_MAPPING_FLOOR:
            logger.info("Mapping failure in %s: %s score=%.2f", league_name, team_name, score)
        return raw, score

    def _load_mapping_cache(self) -> dict[str, str]:
        if not self._mapping_cache_path.exists():
            return {}
        try:
            payload = json.loads(self._mapping_cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {str(key): str(value) for key, value in payload.items()} if isinstance(payload, dict) else {}

    def _save_mapping_cache(self) -> None:
        self._mapping_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._mapping_cache_path.write_text(json.dumps(self._mapping_cache, indent=2), encoding="utf-8")


class LocalJsonStatsProvider:
    """Loads provider-neutral football stats from JSON.

    Supported input shapes:
    - {"teams": [{...}, {...}]}
    - [{...}, {...}]
    - {"leagues": {"England - Premier League": {"teams": [{...}]}}}

    Each team row should contain at least a team/name field, plus model inputs
    such as xg_avg, xga_avg, goals_for_avg, goals_against_avg, elo, form, and
    sample_size. Unknown fields are ignored.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = _resolve_stats_path(settings.external_stats_file)
        self._records: list[dict[str, Any]] | None = None

    def get_match_stats(self, match: Match) -> MatchStatsResult:
        records = self._load_records()
        if not records:
            return _missing_result(match)

        candidates = _filter_candidates(records, match)
        home_raw, home_score = _best_team_match(match.home_team, candidates)
        away_raw, away_score = _best_team_match(match.away_team, candidates)
        min_score = self.settings.min_team_mapping_score
        if home_raw is None or away_raw is None or home_score < min_score or away_score < min_score:
            logger.info(
                "External stats mapping miss for %s vs %s in %s (home=%.2f away=%.2f)",
                match.home_team,
                match.away_team,
                match.league,
                home_score,
                away_score,
            )
            return _missing_result(match)

        stats = {
            match.home_team: _team_stats_from_json(match.home_team, home_raw, home_score),
            match.away_team: _team_stats_from_json(match.away_team, away_raw, away_score),
        }
        return _validate_match_stats(match, stats)

    def _load_records(self) -> list[dict[str, Any]]:
        if self._records is not None:
            return self._records
        if not self.path.exists():
            logger.info("External stats file not found: %s", self.path)
            self._records = []
            return self._records
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load external stats file %s: %s", self.path, exc)
            self._records = []
            return self._records
        self._records = _extract_team_records(payload)
        return self._records


class ApiFootballStatsProvider:
    """Read-only API-Football integration.

    API-Football provides fixtures, fixture statistics, teams, standings, and
    other football data. This provider uses only read endpoints and requires an
    API key. It treats xG as critical: if the provider/league does not expose an
    expected-goals statistic for the recent fixtures, the match is rejected.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._team_cache: dict[str, tuple[dict[str, Any] | None, float]] = {}
        self._stats_cache: dict[int, TeamStats | None] = {}
        self._fixture_stats_cache: dict[int, list[dict[str, Any]]] = {}

    def get_match_stats(self, match: Match) -> MatchStatsResult:
        if not self.settings.api_football_key:
            return _missing_result(match, ["api_football_key_missing"])

        home_raw, home_score = self._map_team(match.home_team, match.country)
        away_raw, away_score = self._map_team(match.away_team, match.country)
        unmatched: list[str] = []
        if home_raw is None or home_score < self.settings.min_team_mapping_score:
            unmatched.append(match.home_team)
        if away_raw is None or away_score < self.settings.min_team_mapping_score:
            unmatched.append(match.away_team)
        if unmatched:
            logger.info("Unmatched API-Football teams for %s: %s", match.match_id, ", ".join(unmatched))
            return _missing_result(match, ["unresolved_team_mapping"], unmatched)

        home_id = int(home_raw["team"]["id"])
        away_id = int(away_raw["team"]["id"])
        home_stats = self._team_stats_from_recent_fixtures(match.home_team, home_id, home_score)
        away_stats = self._team_stats_from_recent_fixtures(match.away_team, away_id, away_score)
        if home_stats is None or away_stats is None:
            missing = []
            if home_stats is None:
                missing.append(f"{match.home_team}: no_recent_fixture_stats")
            if away_stats is None:
                missing.append(f"{match.away_team}: no_recent_fixture_stats")
            return _missing_result(match, missing)

        return _validate_match_stats(match, {match.home_team: home_stats, match.away_team: away_stats})

    def _map_team(self, team_name: str, country: str | None) -> tuple[dict[str, Any] | None, float]:
        cache_key = f"{_normalize_name(team_name)}::{_normalize_name(country or '')}"
        if cache_key in self._team_cache:
            return self._team_cache[cache_key]

        payload = self._get_json("/teams", {"search": team_name})
        candidates = [item for item in payload.get("response", []) if isinstance(item, dict) and item.get("team")]
        best: dict[str, Any] | None = None
        best_score = 0.0
        target = _normalize_name(team_name)
        country_target = _normalize_name(country or "")
        for candidate in candidates:
            team = candidate.get("team", {})
            names = [team.get("name"), team.get("code")]
            base_score = max(
                (SequenceMatcher(None, target, _normalize_name(str(name))).ratio() for name in names if name),
                default=0.0,
            )
            if country_target and _normalize_name(str(team.get("country", ""))) == country_target:
                base_score = min(base_score + 0.05, 1.0)
            if base_score > best_score:
                best = candidate
                best_score = base_score

        result = (best, round(best_score, 4))
        self._team_cache[cache_key] = result
        time.sleep(self.settings.stats_rate_limit_sleep_seconds)
        return result

    def _team_stats_from_recent_fixtures(self, team_name: str, team_id: int, mapping_score: float) -> TeamStats | None:
        if team_id in self._stats_cache:
            return self._stats_cache[team_id]

        payload = self._get_json(
            "/fixtures",
            {
                "team": team_id,
                "last": self.settings.api_football_recent_matches,
                "status": "FT",
            },
        )
        fixtures = [item for item in payload.get("response", []) if isinstance(item, dict)]
        records: list[dict[str, Any]] = []
        for fixture in fixtures:
            record = self._fixture_record(team_id, fixture)
            if record:
                records.append(record)
            time.sleep(self.settings.stats_rate_limit_sleep_seconds)

        if not records:
            self._stats_cache[team_id] = None
            return None

        stats = _team_stats_from_fixture_records(team_name, team_id, records, mapping_score)
        self._stats_cache[team_id] = stats
        return stats

    def _fixture_record(self, team_id: int, fixture: dict[str, Any]) -> dict[str, Any] | None:
        fixture_id = _nested_int(fixture, "fixture", "id")
        home_id = _nested_int(fixture, "teams", "home", "id")
        away_id = _nested_int(fixture, "teams", "away", "id")
        home_goals = _nested_float(fixture, "goals", "home")
        away_goals = _nested_float(fixture, "goals", "away")
        if fixture_id is None or home_id is None or away_id is None or home_goals is None or away_goals is None:
            return None
        is_home = team_id == home_id
        goals_for = home_goals if is_home else away_goals
        goals_against = away_goals if is_home else home_goals
        fixture_stats = self._fixture_statistics(fixture_id)
        team_stats = _stats_for_team(fixture_stats, team_id)
        opponent_stats = _stats_for_team(fixture_stats, away_id if is_home else home_id)
        return {
            "is_home": is_home,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "points": 3.0 if goals_for > goals_against else 1.0 if goals_for == goals_against else 0.0,
            "xg": _stat_value(team_stats, "expected goals", "expected_goals", "xg"),
            "xga": _stat_value(opponent_stats, "expected goals", "expected_goals", "xg"),
            "shots": _stat_value(team_stats, "total shots", "shots total", "shots"),
            "shots_on_target": _stat_value(team_stats, "shots on goal", "shots on target", "shots on"),
            "opponent_strength": _nested_float(fixture, "league", "id"),
        }

    def _fixture_statistics(self, fixture_id: int) -> list[dict[str, Any]]:
        if fixture_id in self._fixture_stats_cache:
            return self._fixture_stats_cache[fixture_id]
        payload = self._get_json("/fixtures/statistics", {"fixture": fixture_id})
        stats = [item for item in payload.get("response", []) if isinstance(item, dict)]
        self._fixture_stats_cache[fixture_id] = stats
        return stats

    def _get_json(self, endpoint: str, params: dict[str, object]) -> dict[str, Any]:
        url = self.settings.api_football_base_url.rstrip("/") + "/" + endpoint.lstrip("/")
        headers = {"x-apisports-key": self.settings.api_football_key or "", "Accept": "application/json"}
        try:
            with httpx.Client(timeout=self.settings.stats_timeout_seconds, headers=headers) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
                if response.status_code == 204 or not response.content:
                    return {"response": []}
                return response.json()
        except httpx.HTTPError as exc:
            logger.warning("API-Football request failed for %s: %s", endpoint, exc)
            return {"response": []}


def _missing_result(
    match: Match,
    reasons: list[str] | None = None,
    unmatched_teams: list[str] | None = None,
) -> MatchStatsResult:
    return MatchStatsResult(
        stats={
            match.home_team: TeamStats(team=match.home_team, stats_source="none", has_external_data=False),
            match.away_team: TeamStats(team=match.away_team, stats_source="none", has_external_data=False),
        },
        has_full_data=False,
        matched_teams=0,
        missing_reasons=reasons or ["missing_external_stats"],
        unmatched_teams=unmatched_teams or [],
    )


def _validate_match_stats(match: Match, stats: dict[str, TeamStats]) -> MatchStatsResult:
    reasons: list[str] = []
    matched = 0
    for team_name in (match.home_team, match.away_team):
        team = stats[team_name]
        missing = sorted(set(team.missing_critical_stats + _critical_missing(team)))
        team.missing_critical_stats = missing
        if team.has_external_data and not missing:
            matched += 1
        elif missing:
            reasons.append(f"{team_name}: missing_{','.join(missing)}")
    return MatchStatsResult(
        stats=stats,
        has_full_data=matched == 2,
        matched_teams=matched,
        missing_reasons=reasons,
        unmatched_teams=[],
    )


def _critical_missing(team: TeamStats) -> list[str]:
    missing: list[str] = []
    has_attack_signal = team.xg_avg is not None or team.goals_for_avg is not None or team.shots_avg is not None
    has_defense_signal = team.xga_avg is not None or team.goals_against_avg is not None
    if not has_attack_signal:
        missing.append("attack_signal")
    if not has_defense_signal:
        missing.append("defense_signal")
    if team.sample_size < 4:
        missing.append("sample_size")
    return missing


def _resolve_stats_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _extract_team_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    teams = payload.get("teams")
    if isinstance(teams, list):
        return [item for item in teams if isinstance(item, dict)]

    leagues = payload.get("leagues")
    records: list[dict[str, Any]] = []
    if isinstance(leagues, dict):
        for league_name, league_payload in leagues.items():
            if isinstance(league_payload, dict):
                league_teams = league_payload.get("teams", [])
            elif isinstance(league_payload, list):
                league_teams = league_payload
            else:
                league_teams = []
            if isinstance(league_teams, list):
                for item in league_teams:
                    if isinstance(item, dict):
                        records.append({"league": str(league_name), **item})
    return records


def _filter_candidates(records: list[dict[str, Any]], match: Match) -> list[dict[str, Any]]:
    league_target = _normalize_name(match.league)
    country_target = _normalize_name(match.country or "")
    scoped = [
        record
        for record in records
        if _field_matches(record.get("league"), league_target)
        or _field_matches(record.get("competition"), league_target)
        or _field_matches(record.get("country"), country_target)
    ]
    return scoped or records


def _field_matches(value: Any, target: str) -> bool:
    if not value or not target:
        return False
    return _normalize_name(str(value)) == target


def _best_team_match(team_name: str, candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    best: dict[str, Any] | None = None
    best_score = 0.0
    target = _normalize_name(team_name)
    target_alias = _alias_name(target)
    for candidate in candidates:
        names = [
            candidate.get("team"),
            candidate.get("name"),
            candidate.get("full_name"),
            candidate.get("display_name"),
            candidate.get("short_name"),
            candidate.get("english_name"),
        ]
        for name in names:
            if not name:
                continue
            candidate_name = _normalize_name(str(name))
            candidate_alias = _alias_name(candidate_name)
            score = _name_similarity(target_alias, candidate_alias)
            if score > best_score:
                best = candidate
                best_score = score
    return best, round(best_score, 4)


def _team_stats_from_json(team_name: str, raw: dict[str, Any], mapping_score: float) -> TeamStats:
    missing_signals = _missing_json_signals(raw)
    return TeamStats(
        team=team_name,
        external_team_id=_str_or_none(raw.get("external_team_id") or raw.get("id")),
        stats_source=str(raw.get("stats_source") or "json"),
        has_external_data=True,
        mapping_confidence=mapping_score,
        elo=_float(raw, "elo", "rating", default=1500.0) or 1500.0,
        league_strength=_float(raw, "league_strength", default=1.0) or 1.0,
        form_last_5=_clip01(_float(raw, "form_last_5", default=0.5)),
        form_last_10=_clip01(_float(raw, "form_last_10", default=0.5)),
        points_per_game_last_5=_float(raw, "points_per_game_last_5", "ppg_last_5", default=1.5) or 1.5,
        points_per_game_last_10=_float(raw, "points_per_game_last_10", "ppg_last_10", default=1.5) or 1.5,
        home_strength=_clip01(_float(raw, "home_strength", default=0.5)),
        away_strength=_clip01(_float(raw, "away_strength", default=0.5)),
        goals_for_avg=_float(raw, "goals_for_avg", "goals_scored_avg", default=1.3) or 1.3,
        goals_against_avg=_float(raw, "goals_against_avg", "goals_conceded_avg", default=1.3) or 1.3,
        goal_difference_trend=_float(raw, "goal_difference_trend", "goal_diff_trend", default=0.0) or 0.0,
        clean_sheet_rate=_clip01(_float(raw, "clean_sheet_rate", default=0.25)),
        failed_to_score_rate=_clip01(_float(raw, "failed_to_score_rate", default=0.25)),
        opponent_strength=_clip01(_float(raw, "opponent_strength", default=0.5)),
        opponent_adjusted_form=_clip01(_float(raw, "opponent_adjusted_form", default=0.5)),
        rest_days=int(_float(raw, "rest_days", default=5.0) or 5),
        xg_avg=_float(raw, "xg_avg", "xg_for_avg", default=None),
        xga_avg=_float(raw, "xga_avg", "xg_against_avg", default=None),
        xg_home_avg=_float(raw, "xg_home_avg", "xg_for_avg_home", default=None),
        xga_home_avg=_float(raw, "xga_home_avg", "xg_against_avg_home", default=None),
        xg_away_avg=_float(raw, "xg_away_avg", "xg_for_avg_away", default=None),
        xga_away_avg=_float(raw, "xga_away_avg", "xg_against_avg_away", default=None),
        xg_trend=_float(raw, "xg_trend", default=None),
        shots_avg=_float(raw, "shots_avg", default=None),
        shots_on_target_avg=_float(raw, "shots_on_target_avg", default=None),
        league_goals_avg=_float(raw, "league_goals_avg", "league_avg_goals", default=None),
        btts_rate=_percentage(raw, "btts_rate"),
        over25_rate=_percentage(raw, "over25_rate"),
        winning_streak=int(_float(raw, "winning_streak", default=0.0) or 0),
        unbeaten_streak=int(_float(raw, "unbeaten_streak", default=0.0) or 0),
        scoring_streak=int(_float(raw, "scoring_streak", default=0.0) or 0),
        clean_sheet_streak=int(_float(raw, "clean_sheet_streak", default=0.0) or 0),
        home_away_streak=int(_float(raw, "home_away_streak", default=0.0) or 0),
        sample_size=int(_float(raw, "sample_size", "matches_played", default=10.0) or 10),
        data_recency_days=int(_float(raw, "data_recency_days", default=3.0) or 3),
        league_reliability=_clip01(_float(raw, "league_reliability", default=0.85)),
        missing_critical_stats=missing_signals,
    )


def _missing_json_signals(raw: dict[str, Any]) -> list[str]:
    attack_keys = ("xg_avg", "xg_for_avg", "goals_for_avg", "goals_scored_avg", "shots_avg")
    defense_keys = ("xga_avg", "xg_against_avg", "goals_against_avg", "goals_conceded_avg")
    missing: list[str] = []
    if all(raw.get(key) in (None, "") for key in attack_keys):
        missing.append("attack_signal")
    if all(raw.get(key) in (None, "") for key in defense_keys):
        missing.append("defense_signal")
    return missing


def _team_stats_from_footystats(team_name: str, raw: dict[str, Any], mapping_score: float) -> TeamStats:
    stats = _stats_dict(raw)
    matches = _float(stats, "seasonMatchesPlayed_overall", default=0.0) or 0.0
    ppg = _float(stats, "seasonPPG_overall", default=None)
    if ppg is None and matches:
        wins = _float(stats, "seasonWinsNum_overall", default=0.0) or 0.0
        draws = _float(stats, "seasonDrawsNum_overall", default=0.0) or 0.0
        ppg = ((wins * 3.0) + draws) / matches
    ppg = ppg if ppg is not None else 1.4
    goals_for = _float(stats, "seasonScoredAVG_overall", "seasonGoalsAVG_overall", default=None)
    goals_against = _float(stats, "seasonConcededAVG_overall", default=None)
    xg_for = _float(stats, "xg_for_avg_overall", default=None)
    xga = _float(stats, "xg_against_avg_overall", default=None)
    xg_home = _float(stats, "xg_for_avg_home", default=xg_for)
    xga_home = _float(stats, "xg_against_avg_home", default=xga)
    xg_away = _float(stats, "xg_for_avg_away", default=xg_for)
    xga_away = _float(stats, "xg_against_avg_away", default=xga)
    league_goals = _float(stats, "seasonAVG_overall", default=None)

    form5 = _form_from_recent(raw, ppg / 3.0)
    form10 = _clip01(ppg / 3.0)
    return TeamStats(
        team=team_name,
        external_team_id=_str_or_none(raw.get("id")),
        stats_source="footystats",
        has_external_data=True,
        mapping_confidence=mapping_score,
        elo=round(1500.0 + ((ppg - 1.35) * 120.0) + ((_float(stats, "seasonGoalDifference_overall", default=0.0) or 0.0) * 2.2), 2),
        form_last_5=form5,
        form_last_10=form10,
        points_per_game_last_5=round(form5 * 3.0, 3),
        points_per_game_last_10=round(form10 * 3.0, 3),
        home_strength=_clip01((_float(stats, "seasonPPG_home", default=ppg) or ppg) / 3.0),
        away_strength=_clip01((_float(stats, "seasonPPG_away", default=ppg) or ppg) / 3.0),
        goals_for_avg=goals_for if goals_for is not None else 1.25,
        goals_against_avg=goals_against if goals_against is not None else 1.25,
        goal_difference_trend=_float(stats, "seasonGoalDifference_overall", default=0.0) or 0.0,
        clean_sheet_rate=_percentage(stats, "seasonCSPercentage_overall"),
        failed_to_score_rate=_percentage(stats, "seasonFTSPercentage_overall"),
        opponent_strength=0.55,
        opponent_adjusted_form=form10,
        xg_avg=xg_for,
        xga_avg=xga,
        xg_home_avg=xg_home,
        xga_home_avg=xga_home,
        xg_away_avg=xg_away,
        xga_away_avg=xga_away,
        xg_trend=(xg_for - goals_for) if xg_for is not None and goals_for is not None else None,
        shots_avg=_float(stats, "shotsAVG_overall", "team_shots_avg_overall", default=None),
        shots_on_target_avg=_float(stats, "shotsOnTargetAVG_overall", default=None),
        league_goals_avg=league_goals,
        btts_rate=_percentage(stats, "seasonBTTSPercentage_overall"),
        over25_rate=_percentage(stats, "seasonOver25Percentage_overall"),
        sample_size=int(matches),
        data_recency_days=3,
        league_reliability=0.88,
    )


def _league_profile_from_teams(teams: list[dict[str, Any]]) -> dict[str, float]:
    team_stats = [_team_stats_from_footystats("league_average", item, 0.80) for item in teams]
    if not team_stats:
        return {
            "goals_for_avg": 1.25,
            "goals_against_avg": 1.25,
            "league_goals_avg": 2.50,
            "btts_rate": 0.50,
            "over25_rate": 0.50,
        }
    return {
        "goals_for_avg": _mean([team.goals_for_avg for team in team_stats]),
        "goals_against_avg": _mean([team.goals_against_avg for team in team_stats]),
        "league_goals_avg": _mean([team.league_goals_avg or 2.50 for team in team_stats]),
        "btts_rate": _mean([team.btts_rate or 0.50 for team in team_stats]),
        "over25_rate": _mean([team.over25_rate or 0.50 for team in team_stats]),
    }


def _fallback_team_stats(team_name: str, league_profile: dict[str, float], reason: str) -> TeamStats:
    goals_for = league_profile.get("goals_for_avg", 1.25)
    goals_against = league_profile.get("goals_against_avg", 1.25)
    return TeamStats(
        team=team_name,
        stats_source="footystats_league_fallback",
        has_external_data=True,
        mapping_confidence=0.55,
        elo=1500.0,
        form_last_5=0.50,
        form_last_10=0.50,
        points_per_game_last_5=1.50,
        points_per_game_last_10=1.50,
        home_strength=0.50,
        away_strength=0.50,
        goals_for_avg=goals_for,
        goals_against_avg=goals_against,
        clean_sheet_rate=0.25,
        failed_to_score_rate=0.25,
        xg_avg=None,
        xga_avg=None,
        shots_avg=None,
        shots_on_target_avg=None,
        league_goals_avg=league_profile.get("league_goals_avg", 2.50),
        btts_rate=league_profile.get("btts_rate", 0.50),
        over25_rate=league_profile.get("over25_rate", 0.50),
        sample_size=6,
        league_reliability=0.68,
        missing_critical_stats=[reason, "missing_xg"],
    )


def _stats_dict(raw: dict[str, Any]) -> dict[str, Any]:
    stats = raw.get("stats")
    if isinstance(stats, list) and stats and isinstance(stats[0], dict):
        return stats[0]
    if isinstance(stats, dict):
        return stats
    return raw


def _form_from_recent(raw: dict[str, Any], fallback: float) -> float:
    value = raw.get("recent_form") or raw.get("formRun_overall")
    if isinstance(value, str) and value:
        points = 0
        games = 0
        for char in value.upper()[:5]:
            if char == "W":
                points += 3
                games += 1
            elif char == "D":
                points += 1
                games += 1
            elif char == "L":
                games += 1
        if games:
            return _clip01(points / (games * 3.0))
    return _clip01(fallback)


def _team_stats_from_fixture_records(
    team_name: str,
    team_id: int,
    records: list[dict[str, Any]],
    mapping_score: float,
) -> TeamStats:
    latest = records[:10]
    last5 = latest[:5]
    goals_for = [_as_float(item["goals_for"]) for item in latest]
    goals_against = [_as_float(item["goals_against"]) for item in latest]
    points = [_as_float(item["points"]) for item in latest]
    home_records = [item for item in latest if item["is_home"]]
    away_records = [item for item in latest if not item["is_home"]]
    xg_values = [_as_float(item["xg"]) for item in latest if item.get("xg") is not None]
    xga_values = [_as_float(item["xga"]) for item in latest if item.get("xga") is not None]
    shots = [_as_float(item["shots"]) for item in latest if item.get("shots") is not None]
    shots_on_target = [_as_float(item["shots_on_target"]) for item in latest if item.get("shots_on_target") is not None]
    ppg10 = _mean(points)
    ppg5 = _mean([_as_float(item["points"]) for item in last5])
    goals_for_avg = _mean(goals_for)
    goals_against_avg = _mean(goals_against)
    gd_trend = _mean([_as_float(item["goals_for"]) - _as_float(item["goals_against"]) for item in last5]) - (
        goals_for_avg - goals_against_avg
    )
    league_goals_avg = _mean([_as_float(item["goals_for"]) + _as_float(item["goals_against"]) for item in latest])
    xg_avg = _mean_or_none(xg_values)
    xga_avg = _mean_or_none(xga_values)
    return TeamStats(
        team=team_name,
        external_team_id=str(team_id),
        stats_source="api_football",
        has_external_data=True,
        mapping_confidence=mapping_score,
        elo=round(1500.0 + ((ppg10 - 1.35) * 115.0) + ((goals_for_avg - goals_against_avg) * 42.0), 2),
        form_last_5=_clip01(ppg5 / 3.0),
        form_last_10=_clip01(ppg10 / 3.0),
        points_per_game_last_5=ppg5,
        points_per_game_last_10=ppg10,
        home_strength=_clip01(_points_per_game(home_records) / 3.0),
        away_strength=_clip01(_points_per_game(away_records) / 3.0),
        goals_for_avg=goals_for_avg,
        goals_against_avg=goals_against_avg,
        goal_difference_trend=gd_trend,
        clean_sheet_rate=sum(_as_float(item["goals_against"]) == 0 for item in latest) / len(latest),
        failed_to_score_rate=sum(_as_float(item["goals_for"]) == 0 for item in latest) / len(latest),
        opponent_strength=0.55,
        opponent_adjusted_form=_clip01(ppg10 / 3.0),
        xg_avg=xg_avg,
        xga_avg=xga_avg,
        xg_trend=(_mean(xg_values[:5]) - _mean(xg_values)) if len(xg_values) >= 5 else None,
        shots_avg=_mean_or_none(shots),
        shots_on_target_avg=_mean_or_none(shots_on_target),
        league_goals_avg=league_goals_avg,
        winning_streak=_streak(latest, lambda item: _as_float(item["points"]) == 3.0),
        unbeaten_streak=_streak(latest, lambda item: _as_float(item["points"]) > 0.0),
        scoring_streak=_streak(latest, lambda item: _as_float(item["goals_for"]) > 0.0),
        clean_sheet_streak=_streak(latest, lambda item: _as_float(item["goals_against"]) == 0.0),
        sample_size=len(latest),
        data_recency_days=3,
        league_reliability=0.82,
    )


def _stats_for_team(fixture_stats: list[dict[str, Any]], team_id: int) -> dict[str, float]:
    for block in fixture_stats:
        if _nested_int(block, "team", "id") == team_id:
            stats = block.get("statistics", [])
            if isinstance(stats, list):
                parsed: dict[str, float] = {}
                for item in stats:
                    if isinstance(item, dict):
                        value = _parse_stat_value(item.get("value"))
                        if value is not None and item.get("type"):
                            parsed[_normalize_stat_type(str(item["type"]))] = value
                return parsed
    return {}


def _stat_value(stats: dict[str, float], *keys: str) -> float | None:
    for key in keys:
        value = stats.get(_normalize_stat_type(key))
        if value is not None:
            return value
    return None


def _normalize_stat_type(value: str) -> str:
    return _normalize_name(value).replace(" ", "_")


def _parse_stat_value(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, str):
        value = value.replace("%", "").strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested_int(container: dict[str, Any], *keys: str) -> int | None:
    value: Any = container
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _nested_float(container: dict[str, Any], *keys: str) -> float | None:
    value: Any = container
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _parse_stat_value(value)


def _as_float(value: Any) -> float:
    parsed = _parse_stat_value(value)
    return parsed if parsed is not None else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_or_none(values: list[float]) -> float | None:
    return round(_mean(values), 4) if values else None


def _points_per_game(records: list[dict[str, Any]]) -> float:
    return _mean([_as_float(item["points"]) for item in records]) if records else 1.5


def _streak(records: list[dict[str, Any]], predicate: Any) -> int:
    count = 0
    for item in records:
        if not predicate(item):
            break
        count += 1
    return count


def _float(container: dict[str, Any], *keys: str, default: float | None = 0.0) -> float | None:
    for key in keys:
        value = container.get(key)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _percentage(container: dict[str, Any], key: str) -> float:
    value = _float(container, key, default=None)
    if value is None:
        return 0.5
    return _clip01(value / 100.0 if value > 1.0 else value)


def _str_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = "".join(char for char in normalized if not unicodedata.combining(char))
    cleaned = "".join(char.lower() if char.isalnum() else " " for char in ascii_value)
    stopwords = {
        "fc",
        "cf",
        "fk",
        "bk",
        "sk",
        "ac",
        "sc",
        "afc",
        "calcio",
        "football",
        "soccer",
        "club",
        "de",
        "the",
        "cd",
        "ec",
        "sp",
        "rj",
        "mg",
        "go",
        "pr",
        "ba",
        "ce",
    }
    return " ".join(part for part in cleaned.split() if part not in stopwords)


def _alias_name(normalized_name: str) -> str:
    return MANUAL_TEAM_ALIASES.get(normalized_name, normalized_name)


def _name_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    left_tokens = left.split()
    right_tokens = right.split()
    token_sort_left = " ".join(sorted(left_tokens))
    token_sort_right = " ".join(sorted(right_tokens))
    base = max(
        SequenceMatcher(None, left, right).ratio(),
        SequenceMatcher(None, token_sort_left, token_sort_right).ratio(),
    )
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    overlap = len(left_set & right_set) / max(len(left_set), len(right_set), 1)
    containment = 0.0
    if len(left) >= 4 and len(right) >= 4 and (left in right or right in left):
        containment = 0.92
    acronym_score = 0.0
    if len(left_tokens) > 1:
        acronym_score = max(acronym_score, 0.90 if "".join(token[0] for token in left_tokens) == right else 0.0)
    if len(right_tokens) > 1:
        acronym_score = max(acronym_score, 0.90 if "".join(token[0] for token in right_tokens) == left else 0.0)
    rapidfuzz_score = 0.0
    if fuzz is not None:
        rapidfuzz_score = max(fuzz.WRatio(left, right), fuzz.token_sort_ratio(left, right)) / 100.0
    return max(base, containment, acronym_score, overlap * 0.95, rapidfuzz_score)


def _clip01(value: float | None) -> float:
    if value is None:
        return 0.5
    return max(0.0, min(1.0, value))
