from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from src.config import Settings
from src.data_models import Match
from src.football_stats import ApiFootballStatsProvider, FootballStatsProvider, LocalJsonStatsProvider


def test_json_provider_maps_teams_and_extracts_xg(tmp_path) -> None:
    stats_path = tmp_path / "external_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "teams": [
                    {
                        "team": "Arsenal FC",
                        "league": "England - Premier League",
                        "country": "England",
                        "elo": 1710,
                        "xg_avg": 2.05,
                        "xga_avg": 0.95,
                        "goals_for_avg": 2.1,
                        "goals_against_avg": 0.9,
                        "shots_avg": 14.2,
                        "shots_on_target_avg": 5.4,
                        "league_goals_avg": 2.8,
                        "sample_size": 10,
                    },
                    {
                        "team": "Chelsea FC",
                        "league": "England - Premier League",
                        "country": "England",
                        "elo": 1615,
                        "xg_avg": 1.65,
                        "xga_avg": 1.25,
                        "goals_for_avg": 1.7,
                        "goals_against_avg": 1.3,
                        "shots_avg": 12.1,
                        "shots_on_target_avg": 4.6,
                        "league_goals_avg": 2.8,
                        "sample_size": 10,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    provider = LocalJsonStatsProvider(Settings(EXTERNAL_STATS_FILE=str(stats_path)))
    result = provider.get_match_stats(_match())

    assert result.has_full_data
    assert result.matched_teams == 2
    assert result.stats["Arsenal"].has_external_data
    assert result.stats["Arsenal"].xg_avg == 2.05
    assert result.stats["Chelsea"].xga_avg == 1.25


def test_missing_json_file_returns_no_external_data(tmp_path) -> None:
    provider = FootballStatsProvider(
        Settings(FOOTBALL_STATS_PROVIDER="json", EXTERNAL_STATS_FILE=str(tmp_path / "missing.json"))
    )
    result = provider.get_match_stats(_match())

    assert not result.has_full_data
    assert result.matched_teams == 0


def test_json_provider_rejects_missing_critical_stats(tmp_path) -> None:
    stats_path = tmp_path / "external_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "teams": [
                    {"team": "Arsenal", "league": "England - Premier League", "xg_avg": 2.0},
                    {"team": "Chelsea", "league": "England - Premier League", "xg_avg": 1.4},
                ]
            }
        ),
        encoding="utf-8",
    )
    provider = LocalJsonStatsProvider(Settings(EXTERNAL_STATS_FILE=str(stats_path)))
    result = provider.get_match_stats(_match())

    assert not result.has_full_data
    assert result.matched_teams == 0
    assert result.missing_reasons


def test_disabled_provider_returns_no_external_data() -> None:
    provider = FootballStatsProvider(Settings(FOOTBALL_STATS_PROVIDER="none"))
    result = provider.get_match_stats(_match())

    assert not result.has_full_data
    assert result.matched_teams == 0


def test_api_football_provider_maps_and_extracts_recent_stats(monkeypatch) -> None:
    provider = ApiFootballStatsProvider(Settings(API_FOOTBALL_KEY="key", API_FOOTBALL_RECENT_MATCHES=5))

    def fake_get_json(endpoint: str, params: dict[str, object]) -> dict[str, object]:
        if endpoint == "/teams" and params["search"] == "Arsenal":
            return {"response": [{"team": {"id": 1, "name": "Arsenal", "country": "England"}}]}
        if endpoint == "/teams" and params["search"] == "Chelsea":
            return {"response": [{"team": {"id": 2, "name": "Chelsea", "country": "England"}}]}
        if endpoint == "/fixtures" and params["team"] == 1:
            return {
                "response": [
                    {
                        "fixture": {"id": fixture_id},
                        "league": {"id": 39},
                        "teams": {"home": {"id": 1}, "away": {"id": 2}},
                        "goals": {"home": 2, "away": 1},
                    }
                    for fixture_id in range(10, 15)
                ]
            }
        if endpoint == "/fixtures" and params["team"] == 2:
            return {
                "response": [
                    {
                        "fixture": {"id": fixture_id},
                        "league": {"id": 39},
                        "teams": {"home": {"id": 1}, "away": {"id": 2}},
                        "goals": {"home": 2, "away": 1},
                    }
                    for fixture_id in range(10, 15)
                ]
            }
        if endpoint == "/fixtures/statistics":
            return {
                "response": [
                    {
                        "team": {"id": 1},
                        "statistics": [
                            {"type": "Expected Goals", "value": "1.9"},
                            {"type": "Total Shots", "value": 14},
                            {"type": "Shots on Goal", "value": 6},
                        ],
                    },
                    {
                        "team": {"id": 2},
                        "statistics": [
                            {"type": "Expected Goals", "value": "1.1"},
                            {"type": "Total Shots", "value": 9},
                            {"type": "Shots on Goal", "value": 3},
                        ],
                    },
                ]
            }
        return {"response": []}

    monkeypatch.setattr(provider, "_get_json", fake_get_json)
    result = provider.get_match_stats(_match())

    assert result.has_full_data
    assert result.stats["Arsenal"].stats_source == "api_football"
    assert result.stats["Arsenal"].xg_avg == 1.9
    assert result.stats["Arsenal"].xga_avg == 1.1
    assert result.stats["Chelsea"].shots_on_target_avg == 3.0


def _match() -> Match:
    return Match(
        match_id="real-1",
        sport="football",
        league="England - Premier League",
        country="England",
        home_team="Arsenal",
        away_team="Chelsea",
        kickoff_time=datetime.now(UTC) + timedelta(days=1),
        markets=[],
    )
