from __future__ import annotations

import math
from dataclasses import dataclass

from src.data_models import Match, TeamStats


HOME_ADVANTAGE_ELO = 65.0


@dataclass(frozen=True)
class GoalProjection:
    home_goals: float
    away_goals: float

    @property
    def total_goals(self) -> float:
        return self.home_goals + self.away_goals


class FootballProbabilityModel:
    """Transparent football model using Elo, Poisson goals, and bounded momentum."""

    def expected_goals(self, match: Match, stats: dict[str, TeamStats]) -> GoalProjection:
        home = stats[match.home_team]
        away = stats[match.away_team]
        league_goal_avg = _league_goal_average(home, away)
        home_attack = _blend(home.xg_home_avg or home.xg_avg, home.goals_for_avg, 0.70)
        away_defense = _blend(away.xga_away_avg or away.xga_avg, away.goals_against_avg, 0.70)
        away_attack = _blend(away.xg_away_avg or away.xg_avg, away.goals_for_avg, 0.70)
        home_defense = _blend(home.xga_home_avg or home.xga_avg, home.goals_against_avg, 0.70)

        home_momentum = momentum_score(home) - 0.5
        away_momentum = momentum_score(away) - 0.5
        home_base = ((home_attack * 0.52) + (away_defense * 0.34) + (league_goal_avg * 0.14)) * 1.07
        away_base = ((away_attack * 0.52) + (home_defense * 0.34) + (league_goal_avg * 0.14)) * 0.93

        home_goals = home_base * (1.0 + (home_momentum * 0.04)) * home.league_strength
        away_goals = away_base * (1.0 + (away_momentum * 0.04)) * away.league_strength
        return GoalProjection(
            home_goals=round(_clamp(home_goals, 0.25, 3.6), 3),
            away_goals=round(_clamp(away_goals, 0.25, 3.6), 3),
        )

    def match_winner_probabilities(self, match: Match, stats: dict[str, TeamStats]) -> dict[str, float]:
        projection = self.expected_goals(match, stats)
        poisson_probs = poisson_match_outcome_probabilities(projection.home_goals, projection.away_goals)
        home = stats[match.home_team]
        away = stats[match.away_team]
        elo_home = elo_win_probability(
            (home.elo * home.league_strength) + HOME_ADVANTAGE_ELO,
            away.elo * away.league_strength,
        )
        elo_away = 1.0 - elo_home
        draw = _clamp(poisson_probs["Draw"] * 0.80 + draw_calibration(projection) * 0.20, 0.16, 0.34)
        non_draw = 1.0 - draw
        home_share = _clamp((poisson_probs["home"] * 0.62) + (elo_home * 0.38), 0.05, 0.90)
        away_share = _clamp((poisson_probs["away"] * 0.62) + (elo_away * 0.38), 0.05, 0.90)
        share_total = home_share + away_share
        return _normalize(
            {
                match.home_team: non_draw * (home_share / share_total),
                "Draw": draw,
                match.away_team: non_draw * (away_share / share_total),
            }
        )

    def over_under_probabilities(self, match: Match, stats: dict[str, TeamStats], line: float) -> dict[str, float]:
        projection = self.expected_goals(match, stats)
        over = poisson_over_probability(projection.total_goals, line)
        return _normalize({f"Over {line:g}": over, f"Under {line:g}": 1.0 - over})

    def btts_probabilities(self, match: Match, stats: dict[str, TeamStats]) -> dict[str, float]:
        projection = self.expected_goals(match, stats)
        home = stats[match.home_team]
        away = stats[match.away_team]
        poisson_yes = poisson_btts_probability(projection.home_goals, projection.away_goals)
        profile_yes = (
            (1.0 - home.failed_to_score_rate)
            + (1.0 - away.failed_to_score_rate)
            + (1.0 - home.clean_sheet_rate)
            + (1.0 - away.clean_sheet_rate)
        ) / 4.0
        btts_history = _mean_optional(home.btts_rate, away.btts_rate)
        if btts_history is not None:
            yes = _clamp((poisson_yes * 0.42) + (profile_yes * 0.35) + (btts_history * 0.23), 0.03, 0.97)
        else:
            yes = _clamp((poisson_yes * 0.60) + (profile_yes * 0.40), 0.03, 0.97)
        return _normalize({"Yes": yes, "No": 1.0 - yes})

    def confidence_score(
        self,
        match: Match,
        stats: dict[str, TeamStats],
        market_type: str,
        probabilities: dict[str, float] | None = None,
    ) -> float:
        home = stats[match.home_team]
        away = stats[match.away_team]
