from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.footystats_source import _match_from_footystats


def test_footystats_match_source_builds_odds_markets() -> None:
    now = datetime.now(UTC)
    raw = {
        "id": 123,
        "status": "incomplete",
        "date_unix": int((now + timedelta(hours=12)).timestamp()),
        "home_name": "Arsenal",
        "away_name": "Chelsea",
        "odds_ft_1": 2.1,
        "odds_ft_x": 3.4,
        "odds_ft_2": 3.2,
        "odds_ft_over15": 1.35,
        "odds_ft_under15": 3.2,
        "odds_ft_over25": 1.9,
        "odds_ft_under25": 1.9,
        "odds_ft_over35": 3.1,
        "odds_ft_under35": 1.35,
        "odds_btts_yes": 1.8,
        "odds_btts_no": 2.0,
    }

    match = _match_from_footystats(raw, "England", "Premier League", now, now + timedelta(hours=24))

    assert match is not None
    assert match.data_source == "footystats"
    assert match.match_id == "footystats-123"
    assert [market.market_type for market in match.markets] == [
        "match_winner",
        "over_under",
        "over_under",
        "over_under",
        "btts",
    ]


def test_footystats_match_source_rejects_missing_odds_and_past_matches() -> None:
    now = datetime.now(UTC)
    raw = {
        "id": 123,
        "status": "incomplete",
        "date_unix": int((now - timedelta(hours=1)).timestamp()),
        "home_name": "Arsenal",
        "away_name": "Chelsea",
        "odds_ft_1": 2.1,
        "odds_ft_x": 3.4,
        "odds_ft_2": 3.2,
    }

    assert _match_from_footystats(raw, "England", "Premier League", now, now + timedelta(hours=24)) is None
