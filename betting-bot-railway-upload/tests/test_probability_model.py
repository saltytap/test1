from __future__ import annotations

from src.probability_model import (
    poisson_btts_probability,
    poisson_match_outcome_probabilities,
    poisson_over_probability,
)


def test_poisson_over_probability() -> None:
    assert poisson_over_probability(2.8, 2.5) > 0.50
    assert poisson_over_probability(1.4, 2.5) < 0.25


def test_btts_probability() -> None:
    assert poisson_btts_probability(1.7, 1.5) > 0.50
    assert poisson_btts_probability(0.7, 0.8) < 0.30


def test_match_outcome_probabilities_sum_to_one() -> None:
    probabilities = poisson_match_outcome_probabilities(1.6, 1.2)

    assert round(sum(probabilities.values()), 4) == 1.0
    assert probabilities["home"] > probabilities["away"]

