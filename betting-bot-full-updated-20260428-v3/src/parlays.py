from __future__ import annotations

from math import prod
from typing import Any

from src.data_models import Prediction


MIN_LEG_PROBABILITY = 0.54
MIN_LEG_CONFIDENCE = 0.64
MIN_LEG_DATA_QUALITY = 0.72
MIN_COMBINED_PROBABILITY = {
    2: 0.34,
    3: 0.20,
}
MAX_COMBINED_ODDS = {
    2: 3.80,
    3: 6.50,
}


def build_fun_parlays(recommendations: list[Prediction], max_cards: int = 2) -> list[dict[str, Any]]:
    """Build small, higher-hit-rate parlay cards from already-approved singles.

    The probability assumes independent legs, so the output is a ranking aid rather than a
    guarantee. Correlated legs from the same match are excluded.
    """

    eligible = _eligible_legs(recommendations)
    cards: list[dict[str, Any]] = []
    used_signatures: set[tuple[str, ...]] = set()
    for leg_count, label in ((2, "high_probability_fun"), (3, "balanced_fun")):
        if len(eligible) < leg_count:
            continue
        legs = _distinct_match_legs(eligible, leg_count)
        if len(legs) != leg_count:
            continue
        signature = tuple(sorted(item.match_id for item in legs))
        if signature in used_signatures:
            continue
        card = _parlay_card(label, legs)
        if (
            card["estimated_hit_probability"] >= MIN_COMBINED_PROBABILITY[leg_count]
            and card["combined_odds"] <= MAX_COMBINED_ODDS[leg_count]
        ):
            cards.append(card)
            used_signatures.add(signature)
        if len(cards) >= max_cards:
            break
    return cards


def _eligible_legs(recommendations: list[Prediction]) -> list[Prediction]:
    legs = [
        item
        for item in recommendations
        if item.recommendation_type == "main"
        and item.risk_tier in {"low", "medium"}
        and item.expected_value > 0
        and item.model_probability >= MIN_LEG_PROBABILITY
        and item.confidence_score >= MIN_LEG_CONFIDENCE
        and item.data_quality_score >= MIN_LEG_DATA_QUALITY
    ]
    return sorted(
        legs,
        key=lambda item: (
            -_leg_score(item),
            item.kickoff_time,
            item.match,
            item.market,
        ),
    )


def _distinct_match_legs(eligible: list[Prediction], leg_count: int) -> list[Prediction]:
    selected: list[Prediction] = []
    match_ids: set[str] = set()
    for item in eligible:
        if item.match_id in match_ids:
            continue
        selected.append(item)
        match_ids.add(item.match_id)
        if len(selected) == leg_count:
            break
    return selected


def _leg_score(item: Prediction) -> float:
    probability_score = item.model_probability * 0.50
    confidence_score = item.confidence_score * 0.25
    quality_score = item.data_quality_score * 0.20
    value_score = min(item.expected_value, 0.05) * 1.0
    return probability_score + confidence_score + quality_score + value_score


def _parlay_card(label: str, legs: list[Prediction]) -> dict[str, Any]:
    combined_odds = prod(item.odds for item in legs)
    combined_probability = prod(item.model_probability for item in legs)
    combined_ev = (combined_probability * combined_odds) - 1.0
    return {
        "type": label,
        "leg_count": len(legs),
        "combined_odds": round(combined_odds, 2),
        "estimated_hit_probability": round(combined_probability, 4),
        "estimated_ev": _format_ev(combined_ev),
        "risk_note": "Estimated hit rate assumes independent legs; keep stakes small.",
        "legs": [
            {
                "match": item.match,
                "market": item.market,
                "pick": item.selection,
                "odds": round(item.odds, 2),
                "model_probability": round(item.model_probability, 4),
                "confidence": round(item.confidence_score, 4),
                "risk": item.risk_tier,
            }
            for item in legs
        ],
    }


def _format_ev(expected_value: float) -> str:
    prefix = "+" if expected_value > 0 else ""
    return f"{prefix}{expected_value * 100:.1f}%"
