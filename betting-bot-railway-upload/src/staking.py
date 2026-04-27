from __future__ import annotations

from src.config import Settings


def flat_stake_pct(settings: Settings) -> float:
    return round(min(settings.flat_stake_pct, settings.max_stake_pct), 4)


def fractional_kelly_stake_pct(decimal_odds: float, model_probability: float, settings: Settings) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("decimal_odds must be greater than 1.0")
    b = decimal_odds - 1.0
    q = 1.0 - model_probability
    full_kelly = ((b * model_probability) - q) / b
    if full_kelly <= 0:
        return 0.0
    stake = full_kelly * settings.kelly_fraction
    return round(max(settings.min_stake_pct, min(stake, settings.max_stake_pct)), 4)


def calculate_stake_pct(decimal_odds: float, model_probability: float, settings: Settings) -> float:
    if settings.staking_strategy.lower() == "flat":
        return flat_stake_pct(settings)
    return fractional_kelly_stake_pct(decimal_odds, model_probability, settings)
