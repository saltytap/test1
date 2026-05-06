from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Selection(BaseModel):
    selection_id: str
    name: str
    decimal_odds: float
    selection_value: str | None = None
    raw_implied_probability: float | None = None
    normalized_implied_probability: float | None = None

    @field_validator("decimal_odds")
    @classmethod
    def odds_must_be_positive(cls, value: float) -> float:
        if value <= 1.0:
            raise ValueError("decimal_odds must be greater than 1.0")
        return value


class Market(BaseModel):
    market_id: str
    name: str
    market_type: str
    selections: list[Selection]


class Match(BaseModel):
    match_id: str
    sport: str
    league: str
    country: str | None = None
    home_team: str
    away_team: str
    kickoff_time: datetime
    status: str = "upcoming"
    data_source: str = "real_api"
    markets: list[Market]


class TeamStats(BaseModel):
    team: str
    external_team_id: str | None = None
    stats_source: str = "none"
    has_external_data: bool = False
    mapping_confidence: float = 0.0
    elo: float = 1500.0
    league_strength: float = 1.0
    form_last_5: float = 0.5
    form_last_10: float = 0.5
    points_per_game_last_5: float = 1.5
    points_per_game_last_10: float = 1.5
    home_strength: float = 0.5
    away_strength: float = 0.5
    goals_for_avg: float = 1.3
    goals_against_avg: float = 1.3
    goal_difference_trend: float = 0.0
    clean_sheet_rate: float = 0.25
    failed_to_score_rate: float = 0.25
    opponent_strength: float = 0.5
    opponent_adjusted_form: float = 0.5
    rest_days: int = 5
    xg_avg: float | None = None
    xga_avg: float | None = None
    xg_home_avg: float | None = None
    xga_home_avg: float | None = None
    xg_away_avg: float | None = None
    xga_away_avg: float | None = None
    xg_trend: float | None = None
    shots_avg: float | None = None
    shots_on_target_avg: float | None = None
    league_goals_avg: float | None = None
    btts_rate: float | None = None
    over25_rate: float | None = None
    missing_critical_stats: list[str] = Field(default_factory=list)
    winning_streak: int = 0
    unbeaten_streak: int = 0
    scoring_streak: int = 0
    clean_sheet_streak: int = 0
    home_away_streak: int = 0
    sample_size: int = 10
    data_recency_days: int = 3
    league_reliability: float = 0.85


class FeatureSet(BaseModel):
    match_id: str
    selection_id: str
    team_form_last_5: float
    team_form_last_10: float
    home_away_strength: float
    goal_difference_trend: float
    opponent_strength_adjustment: float
    market_movement: float
    rest_days: float
    injury_news_adjustment: float = 0.5


class Prediction(BaseModel):
    kickoff_time: datetime
    league: str
    country: str | None = None
    data_source: str
    match_id: str
    market_id: str
    selection_id: str
    match: str
    market: str
    selection: str
    bet_on: str
    odds: float
    model_probability: float
    raw_implied_probability: float
    normalized_implied_probability: float
    edge: float
    expected_value: float
    confidence_score: float
    data_quality_score: float
    recommended_stake_pct: float
    recommended_stake_amount: float
    recommendation_type: str = "main"
    risk_tier: str = "medium"
    reason_codes: list[str] = Field(default_factory=list)


class HistoricalPrediction(BaseModel):
    match_id: str
    selection_id: str
    odds: float
    closing_odds: float
    model_probability: float
    actual_result: bool
    stake_pct: float = 0.01
    metadata: dict[str, Any] = Field(default_factory=dict)


class RejectionDetail(BaseModel):
    match_id: str
    match: str
    league: str
    market: str | None = None
    selection: str | None = None
    reasons: list[str] = Field(default_factory=list)
    expected_value: float | None = None
    edge: float | None = None
    confidence_score: float | None = None
    data_quality_score: float | None = None
