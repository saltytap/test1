from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.backtest import run_backtest, save_backtest_report
from src.config import PROCESSED_DIR, RECOMMENDATIONS_DIR, Settings, ensure_data_dirs, get_settings
from src.data_models import Match, Market, Prediction, RejectionDetail, TeamStats
from src.football_stats import FootballStatsProvider
from src.footystats_source import FootyStatsMatchSource
from src.normalizer import normalize_odds_response
from src.nt_api import NorskTippingClient
from src.probability_model import FootballProbabilityModel, data_quality_score
from src.staking import calculate_stake_pct
from src.value_engine import (
    calculate_edge,
    calculate_expected_value,
    is_upcoming_match,
    market_has_complete_odds,
    rank_recommendations,
    recommendation_filter_failures,
    select_best_market_recommendation,
)


logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("src.football_stats").setLevel(logging.WARNING)
logging.getLogger("src.footystats_api").setLevel(logging.WARNING)

DEFAULT_MARKETS = ["match_winner", "over_under", "btts"]
FAKE_MATCH_ID_RE = re.compile(r"(^match-\d+$|demo|sample|test)", re.IGNORECASE)
MAIN_EV_THRESHOLD = 0.015
MAIN_EXPANDED_EV_THRESHOLD = 0.012
MAIN_EDGE_THRESHOLD = 0.01
MAIN_CONFIDENCE_THRESHOLD = 0.55
MIN_DATA_QUALITY = 0.60
WATCHLIST_EV_FLOOR = 0.0
WATCHLIST_EV_CEILING = 0.01
WATCHLIST_CONFIDENCE_THRESHOLD = 0.55
MIN_PRODUCT_ODDS = 1.40
MAX_PRODUCT_ODDS = 4.50

MARKET_THRESHOLDS = {
    "match_winner": {
        "main_ev": 0.015,
        "main_edge": 0.010,
        "expanded_ev": 0.012,
        "expanded_edge": 0.010,
        "risk_ev": 0.006,
        "risk_edge": 0.0035,
        "watchlist_ceiling": 0.010,
    },
    "over_under": {
        "main_ev": 0.010,
        "main_edge": 0.0075,
        "expanded_ev": 0.008,
        "expanded_edge": 0.005,
        "risk_ev": 0.0025,
        "risk_edge": 0.0015,
        "watchlist_ceiling": 0.010,
    },
    "btts": {
        "main_ev": 0.010,
        "main_edge": 0.0075,
        "expanded_ev": 0.008,
        "expanded_edge": 0.005,
        "risk_ev": 0.0025,
        "risk_edge": 0.0015,
        "watchlist_ceiling": 0.010,
    },
}


def fetch_command(use_mock: bool, sport: str = "football") -> Path:
    settings = get_settings()
    data_source = "mock" if use_mock else "real_api"
    client = NorskTippingClient(settings)
    payload = client.fetch_odds(use_mock=use_mock, sport=sport)
    raw_path = client.save_raw_response(payload, prefix=f"{data_source}_nt_odds")
    matches = normalize_odds_response(payload, data_source=data_source)
    _save_processed_matches(matches, data_source)
    return raw_path


def analyze_command(
    markets: list[str] | None = None,
    max_results: int | None = None,
    country: str | None = None,
    league: str | None = None,
    sport: str = "football",
