from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.config import Settings
from src.data_models import Market, Match, Prediction, Selection
from src.value_engine import (
    calculate_edge,
    calculate_expected_value,
    implied_probability,
    is_upcoming_match,
    normalize_market_margin,
    passes_recommendation_filters,
    rank_recommendations,
    select_best_market_recommendation,
)


def test_implied_probability_from_decimal_odds() -> None:
    assert implied_probability(2.0) == 0.5
    assert implied_probability(1.95) == 0.5128


def test_edge_and_expected_value_calculation() -> None:
    assert calculate_edge(0.58, 0.5128) == 0.0672
    assert calculate_expected_value(0.58, 1.95) == 0.131


def test_market_margin_normalization() -> None:
    market = normalize_market_margin(
        Market(
            market_id="m1",
            name="BTTS",
            market_type="btts",
            selections=[
                Selection(selection_id="yes", name="Yes", decimal_odds=1.90),
                Selection(selection_id="no", name="No", decimal_odds=1.90),
            ],
        )
    )

    assert sum(selection.normalized_implied_probability or 0 for selection in market.selections) == 1.0
    assert market.selections[0].normalized_implied_probability == 0.5


def test_exclude_past_and_live_matches() -> None:
    future = _match(datetime.now(UTC) + timedelta(hours=2), "upcoming")
    past = _match(datetime.now(UTC) - timedelta(hours=2), "upcoming")
    live = _match(datetime.now(UTC) + timedelta(hours=2), "live")

    assert is_upcoming_match(future)
    assert not is_upcoming_match(past)
    assert not is_upcoming_match(live)


def test_recommendation_filtering() -> None:
    settings = Settings(
        EXPECTED_VALUE_THRESHOLD=0.03,
        EDGE_THRESHOLD=0.025,
        MIN_ODDS=1.5,
        MAX_ODDS=4.5,
        CONFIDENCE_THRESHOLD=0.65,
        DATA_QUALITY_THRESHOLD=0.70,
    )
    assert passes_recommendation_filters(0.08, 0.04, 1.95, 0.72, 0.82, settings)
    assert not passes_recommendation_filters(0.02, 0.04, 1.95, 0.72, 0.82, settings)
    assert not passes_recommendation_filters(0.08, 0.02, 1.95, 0.72, 0.82, settings)
    assert not passes_recommendation_filters(0.08, 0.04, 4.9, 0.72, 0.82, settings)
    assert not passes_recommendation_filters(0.08, 0.04, 1.95, 0.60, 0.82, settings)
    assert not passes_recommendation_filters(0.08, 0.04, 1.95, 0.72, 0.65, settings)


def test_best_market_pick_and_ranking() -> None:
    weaker = _prediction("Over 2.5", expected_value=0.06, confidence=0.80)
    stronger = _prediction("BTTS Yes", expected_value=0.12, confidence=0.76)
    best = select_best_market_recommendation([weaker, stronger])

    assert best is not None
    assert best.selection == "BTTS Yes"
    assert rank_recommendations([weaker, stronger], max_results=1) == [stronger]


def _match(kickoff_time: datetime, status: str) -> Match:
    return Match(
        match_id="match-001",
        sport="football",
        league="Eliteserien",
        home_team="A",
        away_team="B",
        kickoff_time=kickoff_time,
        status=status,
        markets=[],
    )


def _prediction(selection: str, expected_value: float, confidence: float) -> Prediction:
    return Prediction(
        kickoff_time=datetime.now(UTC) + timedelta(hours=2),
        league="Eliteserien",
        data_source="mock",
        match_id="match-001",
        market_id="market-001",
        selection_id=selection,
        match="A vs B",
        market="Example",
        selection=selection,
        bet_on=selection,
        odds=2.0,
        model_probability=0.55,
        raw_implied_probability=0.50,
        normalized_implied_probability=0.49,
        edge=0.06,
        expected_value=expected_value,
        confidence_score=confidence,
        data_quality_score=0.82,
        recommended_stake_pct=1.0,
        recommended_stake_amount=100.0,
        reason_codes=["Positive expected value after removing market margin"],
    )
