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
        data_quality = data_quality_score(home, away)
        consistency = 1.0 - min(abs(home.form_last_5 - home.form_last_10) + abs(away.form_last_5 - away.form_last_10), 1.0) * 0.25
        xg_adjustment = 0.04 if home.xg_avg is not None and away.xg_avg is not None else -0.10
        model_agreement = _model_agreement(probabilities or {})
        market_adjustment = 0.02 if market_type in {"over_under", "btts"} else 0.0
        score = (
            data_quality * 0.42
            + consistency * 0.18
            + model_agreement * 0.26
            + 0.08
            + xg_adjustment
            + market_adjustment
        )
        if data_quality < 0.80:
            score -= (0.80 - data_quality) * 0.35
        return round(_clamp(score, 0.05, 0.80), 4)


def elo_win_probability(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def poisson_probability(mean: float, goals: int) -> float:
    return math.exp(-mean) * math.pow(mean, goals) / math.factorial(goals)


def poisson_over_probability(mean_total_goals: float, line: float) -> float:
    max_under_goals = math.floor(line)
    under_or_equal = sum(poisson_probability(mean_total_goals, goals) for goals in range(max_under_goals + 1))
    return round(_clamp(1.0 - under_or_equal, 0.0, 1.0), 4)


def poisson_btts_probability(home_goals: float, away_goals: float) -> float:
    home_scores = 1.0 - poisson_probability(home_goals, 0)
    away_scores = 1.0 - poisson_probability(away_goals, 0)
    return round(_clamp(home_scores * away_scores, 0.0, 1.0), 4)


def poisson_match_outcome_probabilities(home_goals: float, away_goals: float, max_goals: int = 10) -> dict[str, float]:
    home_win = 0.0
    draw = 0.0
    away_win = 0.0
    for home_score in range(max_goals + 1):
        home_prob = poisson_probability(home_goals, home_score)
        for away_score in range(max_goals + 1):
            probability = home_prob * poisson_probability(away_goals, away_score)
            if home_score > away_score:
                home_win += probability
            elif home_score == away_score:
                draw += probability
            else:
                away_win += probability
    total = home_win + draw + away_win
    return {
        "home": home_win / total,
        "Draw": draw / total,
        "away": away_win / total,
    }


def momentum_score(team: TeamStats) -> float:
    base = (
        _clamp(team.winning_streak / 5.0, 0.0, 1.0) * 0.18
        + _clamp(team.unbeaten_streak / 8.0, 0.0, 1.0) * 0.16
        + _clamp(team.scoring_streak / 8.0, 0.0, 1.0) * 0.14
        + _clamp(team.points_per_game_last_5 / 3.0, 0.0, 1.0) * 0.16
        + _clamp((team.goal_difference_trend + 2.0) / 4.0, 0.0, 1.0) * 0.14
        + _clamp(((team.xg_trend if team.xg_trend is not None else 0.0) + 1.0) / 2.0, 0.0, 1.0) * 0.12
        + team.opponent_adjusted_form * 0.10
    )
    support = 1.0
    if team.xg_avg is not None and team.xg_avg < 1.0 and team.winning_streak >= 3:
        support -= 0.20
    if team.opponent_strength < 0.45 and team.winning_streak >= 3:
        support -= 0.15
    if team.xg_trend is not None and team.xg_trend < -0.15 and team.winning_streak >= 2:
        support -= 0.10
    return round(_clamp((base * max(support, 0.55)) + 0.15, 0.05, 0.85), 4)


def data_quality_score(home: TeamStats, away: TeamStats) -> float:
    return round((single_team_data_quality(home) + single_team_data_quality(away)) / 2.0, 4)


def single_team_data_quality(team: TeamStats) -> float:
    if not team.has_external_data:
        return 0.0

    xg_values = [team.xg_avg, team.xga_avg]
    core_values = [team.goals_for_avg, team.goals_against_avg, team.league_goals_avg]
    optional_values = [
        team.xg_home_avg,
        team.xga_home_avg,
        team.xg_away_avg,
        team.xga_away_avg,
        team.xg_trend,
        team.league_goals_avg,
        team.shots_avg,
        team.shots_on_target_avg,
        team.btts_rate,
        team.over25_rate,
    ]
    xg_completeness = sum(value is not None for value in xg_values) / len(xg_values)
    core_completeness = sum(value is not None for value in core_values) / len(core_values)
    optional_completeness = sum(value is not None for value in optional_values) / len(optional_values)
    sample = _clamp(team.sample_size / 10.0, 0.0, 1.0)
    recency = _clamp(1.0 - (team.data_recency_days / 30.0), 0.0, 1.0)
    mapping = _clamp(team.mapping_confidence, 0.0, 1.0)
    form_stability = 1.0 - min(abs(team.form_last_5 - team.form_last_10), 0.55) / 0.55
    if team.xg_avg is None:
        xg_goal_alignment = 0.0
    else:
        xg_goal_alignment = 1.0 - min(abs(team.xg_avg - team.goals_for_avg), 1.25) / 1.25
    score = (
        0.46
        + xg_completeness * 0.105
        + core_completeness * 0.08
        + optional_completeness * 0.07
        + sample * 0.045
        + recency * 0.02
        + team.league_reliability * 0.045
        + mapping * 0.055
        + form_stability * 0.025
        + xg_goal_alignment * 0.025
    )
    if team.missing_critical_stats:
        score -= min(len(team.missing_critical_stats) * 0.09, 0.36)
    if xg_completeness < 1.0:
        score -= (1.0 - xg_completeness) * 0.10
    if mapping < 0.95:
        score -= (0.95 - mapping) * 0.12
    if team.league_reliability < 0.82:
        score -= (0.82 - team.league_reliability) * 0.10

    complete_data = (
        xg_completeness == 1.0
        and core_completeness == 1.0
        and optional_completeness >= 0.80
        and mapping >= 0.95
        and team.sample_size >= 10
        and not team.missing_critical_stats
        and team.stats_source != "footystats_league_fallback"
    )
    dynamic_ceiling = 0.92 if complete_data else 0.90
    if xg_completeness < 1.0:
        dynamic_ceiling = min(dynamic_ceiling, 0.78)
    if mapping < 0.82:
        dynamic_ceiling = min(dynamic_ceiling, 0.78)
    elif mapping < 0.95:
        dynamic_ceiling = min(dynamic_ceiling, 0.85)
    if team.stats_source == "footystats_league_fallback":
        dynamic_ceiling = min(dynamic_ceiling, 0.66)
    if team.league_reliability < 0.75:
        dynamic_ceiling = min(dynamic_ceiling, 0.80)
    return _clamp(score, 0.60, dynamic_ceiling)


def draw_calibration(projection: GoalProjection) -> float:
    goal_gap = abs(projection.home_goals - projection.away_goals)
    total = projection.total_goals
    return _clamp(0.30 - (goal_gap * 0.05) - max(total - 2.4, 0.0) * 0.03, 0.16, 0.34)


def _blend(primary: float | None, fallback: float, primary_weight: float) -> float:
    if primary is None:
        return fallback
    return (primary * primary_weight) + (fallback * (1.0 - primary_weight))


def _normalize(probabilities: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0.0) for value in probabilities.values())
    if total <= 0:
        even = round(1.0 / len(probabilities), 4)
        return {key: even for key in probabilities}
    normalized = {key: round(max(value, 0.0) / total, 4) for key, value in probabilities.items()}
    keys = list(normalized)
    rounding_gap = round(1.0 - sum(normalized.values()), 4)
    normalized[keys[-1]] = round(normalized[keys[-1]] + rounding_gap, 4)
    return normalized


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _model_agreement(probabilities: dict[str, float]) -> float:
    if not probabilities:
        return 0.45
    top_probability = max(probabilities.values())
    # A model that barely separates outcomes or asserts near-certainty should be
    # less trusted than a moderate, well-separated forecast.
    if top_probability > 0.78:
        return 0.55
    return _clamp(0.45 + abs(top_probability - 0.50), 0.35, 0.80)


def _league_goal_average(home: TeamStats, away: TeamStats) -> float:
    values = [value for value in (home.league_goals_avg, away.league_goals_avg) if value is not None]
    if values:
        return _clamp(sum(values) / len(values) / 2.0, 0.7, 1.9)
    return 1.30


def _mean_optional(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None
