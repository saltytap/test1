from __future__ import annotations

from src.config import Settings
from src.staking import calculate_stake_pct, flat_stake_pct, fractional_kelly_stake_pct


def test_fractional_kelly_positive_edge_is_capped() -> None:
    settings = Settings(KELLY_FRACTION=0.25, MAX_STAKE_PCT=0.03)
    assert fractional_kelly_stake_pct(2.0, 0.60, settings) == 0.03


def test_fractional_kelly_negative_edge_returns_zero() -> None:
    settings = Settings(KELLY_FRACTION=0.25, MAX_STAKE_PCT=0.03)
    assert fractional_kelly_stake_pct(2.0, 0.45, settings) == 0.0


def test_flat_staking() -> None:
    settings = Settings(STAKING_STRATEGY="flat", FLAT_STAKE_PCT=0.01, MAX_STAKE_PCT=0.03)
    assert flat_stake_pct(settings) == 0.01
    assert calculate_stake_pct(2.0, 0.45, settings) == 0.01

