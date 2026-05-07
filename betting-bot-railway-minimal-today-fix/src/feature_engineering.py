from __future__ import annotations

from src.data_models import FeatureSet, Match, Selection, TeamStats


def build_features(
    match: Match,
    selection: Selection,
    stats: dict[str, TeamStats],
    market_name: str = "",
) -> FeatureSet:
    market_lower = market_name.lower()
    selection_lower = selection.name.lower()
    if "over/under" in market_lower or "over" in selection_lower or "under" in selection_lower:
        return _total_goals_features(match, selection, stats)
    if selection_lower == "draw":
        return _draw_features(match, selection, stats)

    subject = _subject_team(match, selection)
    team_stats = stats.get(subject, stats[match.home_team])
    opponent = match.away_team if subject == match.home_team else match.home_team
    opponent_stats = stats.get(opponent, stats[match.away_team])

    home_away_strength = team_stats.home_strength if subject == match.home_team else team_stats.away_strength
    goal_difference_trend = _clip01((team_stats.goals_for_avg - team_stats.goals_against_avg + 3.0) / 6.0)
    opponent_adjustment = _clip01(1.0 - opponent_stats.opponent_strength)
    rest_days = _clip01(team_stats.rest_days / 10.0)

    return FeatureSet(
        match_id=match.match_id,
        selection_id=selection.selection_id,
        team_form_last_5=team_stats.form_last_5,
        team_form_last_10=team_stats.form_last_10,
        home_away_strength=home_away_strength,
        goal_difference_trend=goal_difference_trend,
        opponent_strength_adjustment=opponent_adjustment,
        market_movement=0.5,
        rest_days=rest_days,
        injury_news_adjustment=0.5,
    )


def _draw_features(match: Match, selection: Selection, stats: dict[str, TeamStats]) -> FeatureSet:
    home = stats[match.home_team]
    away = stats[match.away_team]
    team_similarity = 1.0 - abs(home.form_last_10 - away.form_last_10)
    goal_balance = 1.0 - min(abs(home.goals_for_avg - away.goals_for_avg) / 3.0, 1.0)
    defensive_balance = 1.0 - min(abs(home.goals_against_avg - away.goals_against_avg) / 3.0, 1.0)
    return FeatureSet(
        match_id=match.match_id,
        selection_id=selection.selection_id,
        team_form_last_5=team_similarity * 0.55,
        team_form_last_10=team_similarity * 0.55,
        home_away_strength=0.42,
        goal_difference_trend=goal_balance * 0.50,
        opponent_strength_adjustment=defensive_balance * 0.50,
        market_movement=0.5,
        rest_days=0.5,
        injury_news_adjustment=0.5,
    )


def _total_goals_features(match: Match, selection: Selection, stats: dict[str, TeamStats]) -> FeatureSet:
    home = stats[match.home_team]
    away = stats[match.away_team]
    expected_goal_pressure = _clip01(
        ((home.goals_for_avg + away.goals_for_avg + home.goals_against_avg + away.goals_against_avg) / 4.0) / 2.5
    )
    if "under" in selection.name.lower():
        expected_goal_pressure = 1.0 - expected_goal_pressure
    return FeatureSet(
        match_id=match.match_id,
        selection_id=selection.selection_id,
        team_form_last_5=expected_goal_pressure,
        team_form_last_10=expected_goal_pressure,
        home_away_strength=expected_goal_pressure,
        goal_difference_trend=expected_goal_pressure,
        opponent_strength_adjustment=expected_goal_pressure,
        market_movement=0.5,
        rest_days=0.5,
        injury_news_adjustment=0.5,
    )


def _subject_team(match: Match, selection: Selection) -> str:
    if selection.name == match.away_team:
        return match.away_team
    return match.home_team


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))
