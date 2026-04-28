from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.data_models import Prediction
from src.parlays import build_fun_parlays


def test_build_fun_parlays_prefers_distinct_high_probability_main_legs() -> None:
    now = datetime.now(UTC)
    recommendations = [
        _prediction("a", now + timedelta(hours=2), 1.55, 0.68, 0.72, 0.86, "low"),
        _prediction("b", now + timedelta(hours=3), 1.60, 0.66, 0.70, 0.84, "low"),
        _prediction("c", now + timedelta(hours=4), 1.70, 0.62, 0.69, 0.82, "medium"),
        _prediction("a-second-market", now + timedelta(hours=2), 1.65, 0.65, 0.73, 0.86, "low", match_id="a"),
    ]

    parlays = build_fun_parlays(recommendations)

    assert len(parlays) == 2
    assert parlays[0]["leg_count"] == 2
    assert parlays[0]["estimated_hit_probability"] >= 0.34
    assert len({leg["match"] for leg in parlays[0]["legs"]}) == 2
    assert parlays[1]["leg_count"] == 3


def test_build_fun_parlays_excludes_watchlist_high_risk_and_low_probability_legs() -> None:
    now = datetime.now(UTC)
    recommendations = [
        _prediction("a", now + timedelta(hours=2), 1.55, 0.53, 0.72, 0.86, "low"),
        _prediction("b", now + timedelta(hours=3), 1.60, 0.66, 0.70, 0.84, "high"),
        _prediction("c", now + timedelta(hours=4), 1.70, 0.62, 0.69, 0.82, "medium", recommendation_type="watchlist"),
    ]

    assert build_fun_parlays(recommendations) == []


def _prediction(
    suffix: str,
    kickoff: datetime,
    odds: float,
    model_probability: float,
    confidence: float,
    data_quality: float,
    risk: str,
    recommendation_type: str = "main",
    match_id: str | None = None,
) -> Prediction:
    actual_match_id = match_id or suffix
    return Prediction(
        kickoff_time=kickoff,
        league="Example League",
        country="Example",
        data_source="real_api",
        match_id=actual_match_id,
        market_id=f"market-{suffix}",
        selection_id=f"selection-{suffix}",
        match=f"Team {suffix} vs Opponent {suffix}",
        market="Over/Under 2.5",
        selection="Over 2.5",
        bet_on="Over 2.5",
        odds=odds,
        model_probability=model_probability,
        raw_implied_probability=1.0 / odds,
        normalized_implied_probability=(1.0 / odds) - 0.02,
        edge=0.03,
        expected_value=(model_probability * odds) - 1.0,
        confidence_score=confidence,
        data_quality_score=data_quality,
        recommended_stake_pct=0.3,
        recommended_stake_amount=30.0,
        recommendation_type=recommendation_type,
        risk_tier=risk,
        reason_codes=["Test recommendation"],
    )
