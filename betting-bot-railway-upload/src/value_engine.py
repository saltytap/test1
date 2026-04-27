from __future__ import annotations

from datetime import UTC, datetime

from src.config import Settings
from src.data_models import Market, Match, Prediction


ACTIVE_STATUSES = {"upcoming", "scheduled", "not_started"}


def implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("decimal_odds must be greater than 1.0")
    return round(1.0 / decimal_odds, 4)


def normalize_market_margin(market: Market) -> Market:
    implied = [implied_probability(selection.decimal_odds) for selection in market.selections]
    total = sum(implied)
    normalized = [round(value / total, 4) for value in implied] if total else [0.0 for _ in implied]
    rounding_gap = round(1.0 - sum(normalized), 4) if normalized else 0.0
    if normalized:
        normalized[-1] = round(normalized[-1] + rounding_gap, 4)
    for selection, raw, no_vig in zip(market.selections, implied, normalized, strict=True):
        selection.raw_implied_probability = raw
        selection.normalized_implied_probability = no_vig
    return market


def calculate_edge(model_probability: float, normalized_implied_probability: float) -> float:
    return round(model_probability - normalized_implied_probability, 4)


def calculate_expected_value(model_probability: float, decimal_odds: float) -> float:
    return round((model_probability * decimal_odds) - 1.0, 4)


def is_upcoming_match(match: Match, now: datetime | None = None) -> bool:
    current_time = now or datetime.now(UTC)
    kickoff = match.kickoff_time
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=UTC)
    return match.status.lower() in ACTIVE_STATUSES and kickoff > current_time


def market_has_complete_odds(market: Market) -> bool:
    return bool(market.selections) and all(selection.decimal_odds > 1.0 for selection in market.selections)


def passes_recommendation_filters(
    expected_value: float,
    edge: float,
    odds: float,
    confidence_score: float,
    data_quality_score: float,
    settings: Settings,
    min_ev: float | None = None,
) -> bool:
    return not recommendation_filter_failures(
        expected_value,
        edge,
        odds,
        confidence_score,
        data_quality_score,
        settings,
        min_ev=min_ev,
    )


def recommendation_filter_failures(
    expected_value: float,
    edge: float,
    odds: float,
    confidence_score: float,
    data_quality_score: float,
    settings: Settings,
    min_ev: float | None = None,
) -> list[str]:
    ev_threshold = settings.expected_value_threshold if min_ev is None else min_ev
    confidence_floor = settings.confidence_threshold
    if expected_value > 0.08:
        confidence_floor = max(confidence_floor, 0.65)
    failures: list[str] = []
    if expected_value <= ev_threshold:
        failures.append("low_ev")
    if expected_value > settings.max_expected_value:
        failures.append("unrealistic_ev")
    if edge <= settings.edge_threshold:
        failures.append("low_edge")
    if not settings.min_odds <= odds <= settings.max_odds:
        failures.append("odds_out_of_range")
    if confidence_score <= confidence_floor:
        failures.append("low_confidence")
    if data_quality_score <= settings.data_quality_threshold:
        failures.append("low_data_quality")
    return failures


def select_best_market_recommendation(candidates: list[Prediction]) -> Prediction | None:
    """Return the strongest non-conflicting bet for a single market."""

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: (
            item.expected_value,
            item.edge,
            item.confidence_score,
            item.data_quality_score,
        ),
    )


def rank_recommendations(recommendations: list[Prediction], max_results: int = 20) -> list[Prediction]:
    ranked = sorted(
        recommendations,
        key=lambda item: (
            -item.expected_value,
            -item.confidence_score,
            -item.data_quality_score,
            item.kickoff_time,
        ),
    )
    return ranked[:max_results]


def reason_codes(
    market_type: str,
    selection_name: str,
    expected_value: float,
    edge: float,
    confidence_score: float,
) -> list[str]:
    reasons = ["Positive expected value after removing market margin"]
    if edge > 0.05:
        reasons.append("Model probability clearly above no-vig implied probability")
    if confidence_score > 0.78:
        reasons.append("High confidence from consistent team and market signals")
    if market_type == "over_under" and "Over" in selection_name:
        reasons.append("Poisson goal model projects higher scoring than market")
    elif market_type == "over_under" and "Under" in selection_name:
        reasons.append("Poisson goal model projects lower scoring than market")
    elif market_type == "btts":
        reasons.append("BTTS probability supported by scoring and conceding profile")
    elif market_type == "match_winner":
        reasons.append("Elo and expected-goal difference support match outcome")
    if expected_value > 0.05:
        reasons.append("Moderate EV cushion versus configured threshold")
    if expected_value > 0.20:
        reasons.append("High-risk EV outlier: verify odds, team mapping, and source data")
    return reasons


def recommendation_to_dict(prediction: Prediction) -> dict[str, object]:
    return prediction.model_dump(mode="json")
