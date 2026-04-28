from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.data_models import Market, Prediction, Selection, TeamStats
from src.main import _candidate_tier_from_metrics, _is_target_selection, _prioritize_rolling_windows, _risk_adjusted_stake, _risk_tier
from src.config import Settings
from src.probability_model import single_team_data_quality


def test_watchlist_requires_positive_ev_and_product_odds() -> None:
    assert _candidate_tier_from_metrics(0.006, 0.004, 2.1, 0.60, 0.72, strong_support=True) == "watchlist"
    assert _candidate_tier_from_metrics(0.0, 0.02, 2.1, 0.70, 0.82, strong_support=True) is None
    assert _candidate_tier_from_metrics(-0.001, 0.02, 2.1, 0.70, 0.82, strong_support=True) is None
    assert _candidate_tier_from_metrics(0.02, 0.02, 4.6, 0.70, 0.82, strong_support=True) is None


def test_expanded_main_requires_strong_support() -> None:
    assert _candidate_tier_from_metrics(0.013, 0.012, 2.2, 0.66, 0.78, strong_support=True) == "main"
    assert _candidate_tier_from_metrics(0.013, 0.012, 2.2, 0.66, 0.78, strong_support=False) is None


def test_goal_and_btts_markets_can_use_lower_supported_edges() -> None:
    assert (
        _candidate_tier_from_metrics(
            0.009,
            0.005,
            1.9,
            0.58,
            0.74,
            market_type="over_under",
            strong_support=True,
        )
        == "main"
    )
    assert _candidate_tier_from_metrics(
        0.005,
        0.003,
        1.9,
        0.58,
        0.74,
        market_type="match_winner",
        strong_support=True,
    ) != "main"


def test_supported_risk_bets_can_be_main_with_positive_ev() -> None:
    assert (
        _candidate_tier_from_metrics(
            0.008,
            0.005,
            2.8,
            0.54,
            0.66,
            market_type="btts",
            strong_support=True,
        )
        == "main"
    )
    assert (
        _candidate_tier_from_metrics(
            0.008,
            0.005,
            2.8,
            0.54,
            0.66,
            market_type="btts",
            strong_support=False,
        )
        is None
    )
    assert (
        _candidate_tier_from_metrics(
            -0.001,
            0.005,
            2.8,
            0.54,
            0.66,
            market_type="btts",
            strong_support=True,
        )
        is None
    )


def test_over_under_target_selection_checks_both_sides() -> None:
    market = Market(
        market_id="ou25",
        name="Over/Under 2.5",
        market_type="over_under",
        selections=[
            Selection(selection_id="over", name="Over 2.5", decimal_odds=1.9),
            Selection(selection_id="under", name="Under 2.5", decimal_odds=1.9),
        ],
    )

    assert _is_target_selection(market, "Over 2.5")
    assert _is_target_selection(market, "Under 2.5")


def test_default_window_is_strictly_next_24h() -> None:
    now = datetime.now(UTC)
    primary = _prediction(now + timedelta(hours=4), "primary")
    secondary = _prediction(now + timedelta(hours=30), "secondary")

    assert _prioritize_rolling_windows([primary, secondary], 24, 48) == [primary]
    assert _prioritize_rolling_windows([primary, secondary], 24, 48, include_secondary_window=True) == [
        primary,
        secondary,
    ]


def test_data_quality_penalizes_partial_and_fuzzy_data() -> None:
    complete = TeamStats(
        team="A",
        stats_source="footystats",
        has_external_data=True,
        mapping_confidence=1.0,
        xg_avg=1.5,
        xga_avg=1.2,
        xg_home_avg=1.6,
        xga_home_avg=1.1,
        xg_away_avg=1.4,
        xga_away_avg=1.3,
        xg_trend=0.1,
        shots_avg=12.0,
        shots_on_target_avg=4.5,
        league_goals_avg=2.7,
        btts_rate=0.55,
        over25_rate=0.54,
        sample_size=18,
        league_reliability=0.88,
    )
    fuzzy = complete.model_copy(update={"mapping_confidence": 0.80})
    fallback = complete.model_copy(
        update={
            "stats_source": "footystats_league_fallback",
            "mapping_confidence": 0.55,
            "xg_avg": None,
            "xga_avg": None,
            "missing_critical_stats": ["mapping_failure", "missing_xg"],
        }
    )

    assert single_team_data_quality(complete) <= 0.92
    assert single_team_data_quality(fuzzy) <= 0.78
    assert single_team_data_quality(fallback) <= 0.66


def test_risk_tier_classification_and_stake_caps() -> None:
    low = _risk_tier(
        recommendation_type="main",
        market=None,
        selection_name="Over 2.5",
        selection_value=None,
        odds=1.85,
        expected_value=0.025,
        edge=0.018,
        confidence_score=0.72,
        data_quality_score=0.86,
    )
    medium = _risk_tier(
        recommendation_type="main",
        market=None,
        selection_name="Home",
        selection_value=None,
        odds=2.65,
        expected_value=0.03,
        edge=0.02,
        confidence_score=0.68,
        data_quality_score=0.78,
    )
    medium_draw = _risk_tier(
        recommendation_type="main",
        market=None,
        selection_name="Draw",
        selection_value="D",
        odds=3.20,
        expected_value=0.03,
        edge=0.02,
        confidence_score=0.70,
        data_quality_score=0.82,
    )
    high = _risk_tier(
        recommendation_type="main",
        market=None,
        selection_name="Away",
        selection_value="A",
        odds=3.80,
        expected_value=0.03,
        edge=0.02,
        confidence_score=0.68,
        data_quality_score=0.82,
    )

    assert low == "low"
    assert medium == "medium"
    assert medium_draw == "medium"
    assert high == "high"
    assert _risk_adjusted_stake(0.02, "low", Settings(MAX_STAKE_PCT=0.01)) == 0.01
    assert _risk_adjusted_stake(0.02, "medium", Settings(MAX_STAKE_PCT=0.01)) == 0.006
    assert _risk_adjusted_stake(0.02, "high", Settings(MAX_STAKE_PCT=0.01)) == 0.003


def _prediction(kickoff: datetime, selection: str) -> Prediction:
    return Prediction(
        kickoff_time=kickoff,
        league="Example League",
        country="Example",
        data_source="real_api",
        match_id=f"real-{selection}",
        market_id="m1",
        selection_id=selection,
        match=f"A {selection} vs B {selection}",
        market="Match Winner",
        selection="A",
        bet_on="A",
        odds=2.0,
        model_probability=0.52,
        raw_implied_probability=0.50,
        normalized_implied_probability=0.49,
        edge=0.03,
        expected_value=0.04,
        confidence_score=0.66,
        data_quality_score=0.78,
        recommended_stake_pct=0.2,
        recommended_stake_amount=20.0,
        recommendation_type="main",
        reason_codes=["Win model 52% vs implied 49%"],
    )
