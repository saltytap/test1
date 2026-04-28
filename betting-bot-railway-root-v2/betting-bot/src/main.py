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
    min_ev: float | None = None,
    use_mock: bool = False,
    debug: bool = False,
    window_hours: int = 24,
    max_window_hours: int = 48,
    include_secondary_window: bool = False,
) -> list[Prediction]:
    settings = get_settings()
    data_source = "mock" if use_mock else "real_api"
    client = NorskTippingClient(settings)
    payload = client.fetch_odds(use_mock=use_mock, sport=sport)
    matches = normalize_odds_response(payload, data_source=data_source)
    if not use_mock and not matches:
        raise RuntimeError("Norsk Tipping returned no normalized matches; refusing to analyze empty production data.")
    _save_processed_matches(matches, data_source)
    recommendations, summary = analyze_matches(
        matches=matches,
        markets=markets or DEFAULT_MARKETS,
        max_results=max_results or settings.max_results,
        country=country,
        league=league,
        sport=sport,
        min_ev=min_ev if min_ev is not None else settings.expected_value_threshold,
        settings=settings,
        production_mode=not use_mock,
        debug=debug,
        window_hours=window_hours,
        max_window_hours=max_window_hours,
        include_secondary_window=include_secondary_window,
    )
    _save_recommendations(recommendations)
    _print_summary(summary)
    _print_recommendations(recommendations, window_hours=window_hours, max_window_hours=max_window_hours)
    return recommendations


def footystats_analyze_command(
    markets: list[str] | None = None,
    max_results: int | None = None,
    country: str | None = None,
    league: str | None = None,
    window_hours: int = 24,
    max_leagues: int | None = None,
    debug: bool = False,
) -> list[Prediction]:
    settings = get_settings()
    source = FootyStatsMatchSource(settings)
    matches, source_summary = source.fetch_matches(
        window_hours=window_hours,
        country=country,
        league=league,
        max_leagues=max_leagues,
    )
    if not matches:
        raise RuntimeError("FootyStats returned no upcoming matches with usable odds in the selected window.")
    _save_processed_matches(matches, "footystats")
    recommendations, summary = analyze_matches(
        matches=matches,
        markets=markets or DEFAULT_MARKETS,
        max_results=max_results or settings.max_results,
        country=country,
        league=None,
        sport="football",
        min_ev=settings.expected_value_threshold,
        settings=settings,
        production_mode=True,
        debug=debug,
        window_hours=window_hours,
        max_window_hours=window_hours,
        include_secondary_window=False,
    )
    summary["source_summary"] = source_summary
    _save_recommendations(recommendations)
    _print_summary(summary)
    _print_recommendations(recommendations, window_hours=window_hours, max_window_hours=window_hours)
    return recommendations


def analyze_matches(
    matches: list[Match],
    markets: list[str],
    max_results: int,
    country: str | None,
    league: str | None,
    sport: str,
    min_ev: float,
    settings: Settings,
    production_mode: bool,
    debug: bool = False,
    window_hours: int = 24,
    max_window_hours: int = 48,
    include_secondary_window: bool = False,
) -> tuple[list[Prediction], dict[str, object]]:
    selected_markets = set(markets)
    filtered_matches = [
        match
        for match in matches
        if _matches_filters(match, country=country, league=league, sport=sport)
        and not (production_mode and is_fake_match_id(match.match_id))
    ]
    future_matches = [match for match in filtered_matches if is_upcoming_match(match)]
    analysis_matches = [
        match
        for match in future_matches
        if _match_window(match, window_hours=window_hours, max_window_hours=max_window_hours) == "primary"
        or (
            include_secondary_window
            and _match_window(match, window_hours=window_hours, max_window_hours=max_window_hours) == "secondary"
        )
    ]
    stats_provider = FootballStatsProvider(settings)
    model = FootballProbabilityModel()
    recommendations: list[Prediction] = []
    debug_rejections: list[RejectionDetail] = []
    stats_coverage = {
        "matches_with_full_external_data": 0,
        "matches_with_partial_external_data": 0,
        "matches_missing_external_data": 0,
        "market_only_matches": 0,
        "matches_analyzed": 0,
        "matches_with_usable_stats": 0,
        "teams_matched": 0,
        "teams_required": len(analysis_matches) * 2,
    }

    for match in analysis_matches:
        stats_result = stats_provider.get_match_stats(match)
        stats_coverage["teams_matched"] += stats_result.matched_teams
        if not stats_result.has_full_data:
            stats_coverage["matches_missing_external_data"] += 1
            stats = stats_result.stats
            quality = data_quality_score(stats[match.home_team], stats[match.away_team])
            has_any_external_signal = any(team.has_external_data for team in stats.values())
            if settings.require_external_stats and (not has_any_external_signal or quality < 0.60):
                debug_rejections.append(
                    RejectionDetail(
                        match_id=match.match_id,
                        match=f"{match.home_team} vs {match.away_team}",
                        league=match.league,
                        reasons=_map_rejection_reasons(stats_result.missing_reasons or ["missing_external_stats"]),
                        data_quality_score=quality,
                    )
                )
                continue
            if has_any_external_signal:
                stats_coverage["matches_with_partial_external_data"] += 1
            else:
                stats = _market_only_stats(match)
                quality = settings.market_only_data_quality
                stats_coverage["market_only_matches"] += 1
            debug_rejections.append(
                RejectionDetail(
                    match_id=match.match_id,
                    match=f"{match.home_team} vs {match.away_team}",
                    league=match.league,
                    reasons=_map_rejection_reasons(stats_result.missing_reasons or ["partial_external_stats"]),
                    data_quality_score=quality,
                )
            )
        else:
            stats_coverage["matches_with_full_external_data"] += 1
            stats = stats_result.stats
            quality = data_quality_score(stats[match.home_team], stats[match.away_team])
        if any(team.has_external_data for team in stats.values()) and quality >= 0.60:
            stats_coverage["matches_with_usable_stats"] += 1
        stats_coverage["matches_analyzed"] += 1
        match_recommendations_before = len(recommendations)
        for market in match.markets:
            if market.market_type not in selected_markets or not market_has_complete_odds(market):
                continue
            best_pick = _analyze_market(
                match,
                market,
                stats,
                model,
                settings,
                quality,
                min_ev,
                debug_rejections,
                market_only=stats[match.home_team].stats_source == "market_only",
            )
            if best_pick:
                recommendations.append(best_pick)
        if len(recommendations) == match_recommendations_before:
            debug_rejections.append(
                RejectionDetail(
                    match_id=match.match_id,
                    match=f"{match.home_team} vs {match.away_team}",
                    league=match.league,
                    reasons=["no_selection_passed_filters"],
                    data_quality_score=quality,
                )
            )

    calibrated_recommendations = _calibrate_recommendation_portfolio(recommendations, settings, min_ev)
    windowed_recommendations = _prioritize_rolling_windows(
        calibrated_recommendations,
        window_hours,
        max_window_hours,
        include_secondary_window=include_secondary_window,
    )
    ranked = _rank_tiered_recommendations(windowed_recommendations, max_results)
    summary = _build_summary(
        filtered_matches,
        future_matches,
        ranked,
        stats_coverage,
        debug_rejections,
        window_hours=window_hours,
        max_window_hours=max_window_hours,
        include_debug_rows=debug,
    )
    return ranked, summary


def backtest_command() -> dict[str, float | bool]:
    settings = get_settings()
    report = run_backtest(bankroll=settings.bankroll, history_path=settings.backtest_history_file)
    path = save_backtest_report(report)
    print(json.dumps(report, indent=2))
    logger.info("Saved backtest report to %s", path)
    return report


def export_command(output_format: str) -> Path:
    ensure_data_dirs()
    source = RECOMMENDATIONS_DIR / "latest_recommendations.json"
    if not source.exists():
        raise RuntimeError("No recommendations found. Run analyze first.")
    rows = json.loads(source.read_text(encoding="utf-8"))
    if output_format == "json":
        target = RECOMMENDATIONS_DIR / "latest_recommendations.json"
    else:
        target = RECOMMENDATIONS_DIR / "latest_recommendations.csv"
        pd.DataFrame(rows).to_csv(target, index=False)
    print(f"Exported recommendations to {target}")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Norsk Tipping Oddsen value analysis bot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch and store odds data")
    fetch_parser.add_argument("--mock", action="store_true", help="Use mocked demo odds response")
    fetch_parser.add_argument("--sport", default="football")

    analyze_parser = subparsers.add_parser("analyze", help="Generate value recommendations")
    analyze_parser.add_argument("--mock", action="store_true", help="Use mocked demo odds response")
    analyze_parser.add_argument("--country", default=None)
    analyze_parser.add_argument("--league", default=None)
    analyze_parser.add_argument("--sport", default="football")
    analyze_parser.add_argument("--markets", nargs="+", default=DEFAULT_MARKETS)
    analyze_parser.add_argument("--max-results", type=int, default=None)
    analyze_parser.add_argument("--min-ev", type=float, default=None)
    analyze_parser.add_argument("--window-hours", type=int, default=24)
    analyze_parser.add_argument("--max-window-hours", type=int, default=48)
    analyze_parser.add_argument(
        "--include-secondary-window",
        action="store_true",
        help="Also show qualifying recommendations from the 24-48h fallback window",
    )
    analyze_parser.add_argument("--debug", action="store_true", help="Print rejection reasons for skipped matches/selections")

    footystats_parser = subparsers.add_parser("footystats-analyze", help="Generate recommendations from FootyStats odds and stats only")
    footystats_parser.add_argument("--country", default=None)
    footystats_parser.add_argument("--league", default=None)
    footystats_parser.add_argument("--markets", nargs="+", default=DEFAULT_MARKETS)
    footystats_parser.add_argument("--max-results", type=int, default=None)
    footystats_parser.add_argument("--window-hours", type=int, default=24)
    footystats_parser.add_argument("--max-leagues", type=int, default=None)
    footystats_parser.add_argument("--debug", action="store_true", help="Print rejection reasons for skipped matches/selections")

    subparsers.add_parser("backtest", help="Run historical backtest from configured data file")

    export_parser = subparsers.add_parser("export", help="Export latest recommendations")
    export_parser.add_argument("--format", choices=["csv", "json"], default="csv")

    args = parser.parse_args()
    try:
        if args.command == "fetch":
            fetch_command(use_mock=args.mock, sport=args.sport)
        elif args.command == "analyze":
            analyze_command(
                markets=args.markets,
                max_results=args.max_results,
                country=args.country,
                league=args.league,
                sport=args.sport,
                min_ev=args.min_ev,
                use_mock=args.mock,
                debug=args.debug,
                window_hours=args.window_hours,
                max_window_hours=args.max_window_hours,
                include_secondary_window=args.include_secondary_window,
            )
        elif args.command == "footystats-analyze":
            footystats_analyze_command(
                markets=args.markets,
                max_results=args.max_results,
                country=args.country,
                league=args.league,
                window_hours=args.window_hours,
                max_leagues=args.max_leagues,
                debug=args.debug,
            )
        elif args.command == "backtest":
            backtest_command()
        elif args.command == "export":
            export_command(output_format=args.format)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _analyze_market(
    match: Match,
    market: Market,
    stats: dict[str, TeamStats],
    model: FootballProbabilityModel,
    settings: Settings,
    data_quality: float,
    min_ev: float,
    debug_rejections: list[RejectionDetail] | None = None,
    market_only: bool = False,
) -> Prediction | None:
    raw_probabilities = _market_model_probabilities(match, market, stats, model)
    if not raw_probabilities:
        return None

    probabilities = _calibrate_probabilities_to_market(match, market, raw_probabilities, data_quality)
    market_confidence = model.confidence_score(match, stats, market.market_type, probabilities)
    candidates: list[Prediction] = []
    for selection in market.selections:
        probability_key = _probability_key(match, market, selection.name, selection.selection_value)
        model_probability = probabilities.get(probability_key)
        if model_probability is None:
            continue
        normalized_implied = selection.normalized_implied_probability
        raw_implied = selection.raw_implied_probability
        if normalized_implied is None or raw_implied is None:
            continue
        if not _is_target_selection(market, selection.name):
            continue
        profile_failure = _selection_profile_failure(
            match,
            market,
            selection.name,
            selection.selection_value,
            selection.decimal_odds,
            stats,
            model,
        )
        if profile_failure and profile_failure != "underdog_control":
            if debug_rejections is not None:
                debug_rejections.append(
                    RejectionDetail(
                        match_id=match.match_id,
                        match=f"{match.home_team} vs {match.away_team}",
                        league=match.league,
                        market=market.name,
                        selection=selection.name,
                        reasons=[profile_failure],
                        data_quality_score=data_quality,
                    )
                )
            continue
        confidence = _selection_confidence_score(
            match=match,
            stats=stats,
            market_type=market.market_type,
            model_probability=model_probability,
            normalized_implied_probability=normalized_implied,
            data_quality=data_quality,
            market_confidence=market_confidence,
            odds=selection.decimal_odds,
        )
        if market_only:
            confidence = min(confidence, settings.market_only_confidence_cap)
        edge = calculate_edge(model_probability, normalized_implied)
        expected_value = calculate_expected_value(model_probability, selection.decimal_odds)
        stake_decimal = calculate_stake_pct(selection.decimal_odds, model_probability, settings)
        strong_support = _strong_selection_support(match, market, selection.name, selection.selection_value, stats, model)
        failures = _tier_filter_failures(
            expected_value,
            edge,
            selection.decimal_odds,
            confidence,
            data_quality,
            stats,
            match,
            market_type=market.market_type,
            strong_support=strong_support,
        )
        recommendation_type = _candidate_tier(
            expected_value,
            edge,
            selection.decimal_odds,
            confidence,
            data_quality,
            stats,
            match,
            market_type=market.market_type,
            strong_support=strong_support,
            profile_failure=profile_failure,
        )
        if recommendation_type:
            if recommendation_type == "watchlist":
                stake_decimal = 0.0
            risk_tier = _risk_tier(
                recommendation_type=recommendation_type,
                market=market,
                selection_name=selection.name,
                selection_value=selection.selection_value,
                odds=selection.decimal_odds,
                expected_value=expected_value,
                edge=edge,
                confidence_score=confidence,
                data_quality_score=data_quality,
            )
            stake_decimal = _risk_adjusted_stake(stake_decimal, risk_tier, settings)
            candidates.append(
                Prediction(
                    kickoff_time=match.kickoff_time,
                    league=match.league,
                    country=match.country,
                    data_source=match.data_source,
                    match_id=match.match_id,
                    market_id=market.market_id,
                    selection_id=selection.selection_id,
                    match=f"{match.home_team} vs {match.away_team}",
                    market=market.name,
                    selection=selection.name,
                    bet_on=selection.name,
                    odds=selection.decimal_odds,
                    model_probability=model_probability,
                    raw_implied_probability=raw_implied,
                    normalized_implied_probability=normalized_implied,
                    edge=edge,
                    expected_value=expected_value,
                    confidence_score=confidence,
                    data_quality_score=round(data_quality, 4),
                    recommended_stake_pct=round(stake_decimal * 100.0, 2),
                    recommended_stake_amount=round(settings.bankroll * stake_decimal, 2),
                    recommendation_type=recommendation_type,
                    risk_tier=risk_tier,
                    reason_codes=_risk_labels(selection.decimal_odds, profile_failure)
                    + _recommendation_reasons(
                        match,
                        market,
                        selection.name,
                        selection.selection_value,
                        model_probability,
                        normalized_implied,
                        stats,
                        model,
                    )
                    + (["Market-only analysis: external team stats not required"] if market_only else []),
                )
            )
        elif debug_rejections is not None:
            debug_rejections.append(
                RejectionDetail(
                    match_id=match.match_id,
                    match=f"{match.home_team} vs {match.away_team}",
                    league=match.league,
                    market=market.name,
                    selection=selection.name,
                    reasons=failures,
                    expected_value=expected_value,
                    edge=edge,
                    confidence_score=confidence,
                    data_quality_score=data_quality,
                )
            )
    return select_best_market_recommendation(candidates)


def _market_model_probabilities(
    match: Match,
    market: Market,
    stats: dict[str, TeamStats],
    model: FootballProbabilityModel,
) -> dict[str, float]:
    if market.market_type == "match_winner":
        return model.match_winner_probabilities(match, stats)
    if market.market_type == "over_under":
        line = _extract_total_line(market.name)
        if line is None:
            return {}
        return model.over_under_probabilities(match, stats, line)
    if market.market_type == "btts":
        return model.btts_probabilities(match, stats)
    return {}


def _probability_key(match: Match, market: Market, selection_name: str, selection_value: str | None) -> str:
    value = (selection_value or "").upper()
    if market.market_type == "match_winner":
        if value == "H":
            return match.home_team
        if value == "D":
            return "Draw"
        if value == "A":
            return match.away_team
        if selection_name.lower() in {"uavgjort", "draw", "d"}:
            return "Draw"
    if market.market_type == "btts":
        normalized = selection_name.lower()
        if normalized in {"ja", "yes"}:
            return "Yes"
        if normalized in {"nei", "no"}:
            return "No"
    return selection_name


def _is_target_selection(market: Market, selection_name: str) -> bool:
    if market.market_type != "over_under":
        return True
    line = _extract_total_line(market.name)
    if line is None:
        return False
    normalized = selection_name.lower()
    return line in {1.5, 2.5, 3.5} and ("over" in normalized or "under" in normalized)


def _calibrate_probabilities_to_market(
    match: Match,
    market: Market,
    model_probabilities: dict[str, float],
    data_quality: float,
) -> dict[str, float]:
    market_probabilities: dict[str, float] = {}
    for selection in market.selections:
        key = _probability_key(match, market, selection.name, selection.selection_value)
        if selection.normalized_implied_probability is not None:
            market_probabilities[key] = selection.normalized_implied_probability

    if not market_probabilities:
        return model_probabilities

    if market.market_type == "btts":
        model_weight = max(0.40, min(0.78, 0.50 + max(data_quality - 0.65, 0.0) * 0.80))
    elif market.market_type == "over_under":
        model_weight = max(0.30, min(0.66, 0.40 + max(data_quality - 0.65, 0.0) * 0.80))
    else:
        model_weight = max(0.20, min(0.50, 0.28 + max(data_quality - 0.65, 0.0) * 0.65))
    preliminary_edges = [
        abs(model_probabilities[key] - market_probability)
        for key, market_probability in market_probabilities.items()
        if key in model_probabilities
    ]
    if preliminary_edges:
        average_edge = sum(preliminary_edges) / len(preliminary_edges)
        high_edge_share = sum(edge > 0.05 for edge in preliminary_edges) / len(preliminary_edges)
        goal_market = market.market_type in {"over_under", "btts"}
        average_edge_limit = 0.075 if goal_market and data_quality >= 0.85 else 0.045
        if average_edge > average_edge_limit:
            model_weight *= 0.88 if goal_market else 0.78
        if high_edge_share > 0.50 and not goal_market:
            model_weight *= 0.70
        model_weight = max(0.12, model_weight)
    blended = {
        key: (probability * model_weight) + (market_probabilities.get(key, probability) * (1.0 - model_weight))
        for key, probability in model_probabilities.items()
    }
    total = sum(blended.values())
    if total <= 0:
        return model_probabilities
    normalized = {key: round(value / total, 4) for key, value in blended.items()}
    keys = list(normalized)
    normalized[keys[-1]] = round(normalized[keys[-1]] + (1.0 - sum(normalized.values())), 4)
    return normalized


def _calibrate_recommendation_portfolio(
    recommendations: list[Prediction],
    settings: Settings,
    min_ev: float,
) -> list[Prediction]:
    if len(recommendations) < 3:
        return recommendations

    average_edge = sum(item.edge for item in recommendations) / len(recommendations)
    high_edge_share = sum(item.edge > 0.05 for item in recommendations) / len(recommendations)
    if average_edge <= 0.04 and high_edge_share <= 0.35:
        return recommendations

    shrink_factor = 0.75
    confidence_penalty = 0.015
    if average_edge > 0.06 or high_edge_share > 0.60:
        shrink_factor = 0.62
        confidence_penalty = 0.035

    calibrated: list[Prediction] = []
    for item in recommendations:
        item_shrink_factor = shrink_factor
        item_market_type = _market_type_from_market_name(item.market)
        if item_market_type in {"over_under", "btts"} and item.confidence_score >= 0.70 and item.data_quality_score >= 0.85:
            item_shrink_factor = max(item_shrink_factor, 0.90)
        adjusted_probability = item.normalized_implied_probability + (
            (item.model_probability - item.normalized_implied_probability) * item_shrink_factor
        )
        item.model_probability = round(adjusted_probability, 4)
        item.edge = calculate_edge(item.model_probability, item.normalized_implied_probability)
        item.expected_value = calculate_expected_value(item.model_probability, item.odds)
        item.confidence_score = round(_clamp(item.confidence_score - confidence_penalty, 0.50, 0.80), 4)
        stake_decimal = calculate_stake_pct(item.odds, item.model_probability, settings)
        item.risk_tier = _risk_tier(
            recommendation_type=item.recommendation_type,
            market=None,
            selection_name=item.selection,
            selection_value=None,
            odds=item.odds,
            expected_value=item.expected_value,
            edge=item.edge,
            confidence_score=item.confidence_score,
            data_quality_score=item.data_quality_score,
        )
        stake_decimal = _risk_adjusted_stake(stake_decimal, item.risk_tier, settings)
        item.recommended_stake_pct = round(stake_decimal * 100.0, 2)
        item.recommended_stake_amount = round(settings.bankroll * stake_decimal, 2)
        strong_support = any(
            token in reason.lower()
            for reason in item.reason_codes
            for token in ("xg", "model", "btts rate", "over 2.5 rate", "conceded")
        )
        tier = _candidate_tier_from_metrics(
            item.expected_value,
            item.edge,
            item.odds,
            item.confidence_score,
            item.data_quality_score,
            market_type=item_market_type,
            strong_support=strong_support,
        )
        if tier == "main":
            item.recommendation_type = "main"
            calibrated.append(item)
        elif tier == "watchlist":
            item.recommendation_type = "watchlist"
            item.risk_tier = _risk_tier(
                recommendation_type="watchlist",
                market=None,
                selection_name=item.selection,
                selection_value=None,
                odds=item.odds,
                expected_value=item.expected_value,
                edge=item.edge,
                confidence_score=item.confidence_score,
                data_quality_score=item.data_quality_score,
            )
            item.recommended_stake_pct = 0.0
            item.recommended_stake_amount = 0.0
            calibrated.append(item)
    return calibrated


def _rank_tiered_recommendations(recommendations: list[Prediction], max_results: int) -> list[Prediction]:
    main = [item for item in recommendations if item.recommendation_type == "main"]
    watchlist = [item for item in recommendations if item.recommendation_type == "watchlist"]
    main_limit = min(6, max_results)
    watchlist_limit = 15 if len(main) < 2 else max(5, max_results - main_limit)
    main_ranked = _cap_market_share(rank_recommendations(main, main_limit), main_limit)
    watchlist_ranked = _cap_market_share(rank_recommendations(watchlist, watchlist_limit), watchlist_limit)
    return main_ranked + watchlist_ranked


def _cap_market_share(recommendations: list[Prediction], max_results: int) -> list[Prediction]:
    if len(recommendations) < 4:
        return recommendations[:max_results]
    cap = max(1, (max_results + 1) // 2)
    counts: Counter[str] = Counter()
    selected: list[Prediction] = []
    overflow: list[Prediction] = []
    for item in recommendations:
        market_group = _market_group(item.market)
        if counts[market_group] < cap:
            selected.append(item)
            counts[market_group] += 1
        else:
            overflow.append(item)
    if len(selected) < max_results:
        selected.extend(overflow[: max_results - len(selected)])
    return selected[:max_results]


def _prioritize_rolling_windows(
    recommendations: list[Prediction],
    window_hours: int,
    max_window_hours: int,
    include_secondary_window: bool = False,
) -> list[Prediction]:
    primary = [
        item
        for item in recommendations
        if _recommendation_window(item, window_hours=window_hours, max_window_hours=max_window_hours) == "primary"
    ]
    secondary = [
        item
        for item in recommendations
        if _recommendation_window(item, window_hours=window_hours, max_window_hours=max_window_hours) == "secondary"
    ]
    if include_secondary_window:
        return primary + secondary
    return primary


def _candidate_tier(
    expected_value: float,
    edge: float,
    odds: float,
    confidence_score: float,
    data_quality_score: float,
    stats: dict[str, TeamStats],
    match: Match,
    market_type: str,
    strong_support: bool,
    profile_failure: str | None = None,
) -> str | None:
    if profile_failure == "underdog_control":
        return None
    tier = _candidate_tier_from_metrics(
        expected_value,
        edge,
        odds,
        confidence_score,
        data_quality_score,
        market_type=market_type,
        strong_support=strong_support,
    )
    if tier == "watchlist" and not _supported_by_stats(stats, match):
        return None
    return tier


def _candidate_tier_from_metrics(
    expected_value: float,
    edge: float,
    odds: float,
    confidence_score: float,
    data_quality_score: float,
    market_type: str = "match_winner",
    strong_support: bool = False,
) -> str | None:
    thresholds = _market_thresholds(market_type)
    if not MIN_PRODUCT_ODDS <= odds <= MAX_PRODUCT_ODDS or data_quality_score <= MIN_DATA_QUALITY:
        return None
    if expected_value > 0.08 and confidence_score <= 0.65:
        return None
    if (
        expected_value > thresholds["main_ev"]
        and edge > thresholds["main_edge"]
        and confidence_score > MAIN_CONFIDENCE_THRESHOLD
        and (market_type == "match_winner" or strong_support)
    ):
        return "main"
    thin_edge = expected_value < 0.0075
    risk_confidence_floor = 0.70 if thin_edge else 0.52
    risk_data_floor = 0.85 if thin_edge else MIN_DATA_QUALITY
    if (
        expected_value >= thresholds["risk_ev"]
        and edge >= thresholds["risk_edge"]
        and confidence_score > risk_confidence_floor
        and data_quality_score > risk_data_floor
        and strong_support
    ):
        return "main"
    if (
        expected_value >= thresholds["expanded_ev"]
        and edge > thresholds["expanded_edge"]
        and confidence_score > 0.65
        and strong_support
    ):
        return "main"
    if (
        WATCHLIST_EV_FLOOR < expected_value < thresholds["watchlist_ceiling"]
        and edge > 0
        and confidence_score > WATCHLIST_CONFIDENCE_THRESHOLD
        and strong_support
    ):
        return "watchlist"
    return None


def _tier_filter_failures(
    expected_value: float,
    edge: float,
    odds: float,
    confidence_score: float,
    data_quality_score: float,
    stats: dict[str, TeamStats],
    match: Match,
    market_type: str = "match_winner",
    strong_support: bool = False,
) -> list[str]:
    thresholds = _market_thresholds(market_type)
    failures: list[str] = []
    if expected_value <= 0:
        failures.append("low_ev")
    elif expected_value < thresholds["risk_ev"]:
        failures.append("low_ev")
    if edge <= 0:
        failures.append("low_edge")
    elif edge < thresholds["risk_edge"]:
        failures.append("low_edge")
    if not MIN_PRODUCT_ODDS <= odds <= MAX_PRODUCT_ODDS:
        failures.append("odds_out_of_range")
    if confidence_score <= MAIN_CONFIDENCE_THRESHOLD:
        failures.append("low_confidence")
    if data_quality_score <= MIN_DATA_QUALITY:
        failures.append("low_data_quality")
    if not _supported_by_stats(stats, match) or not strong_support:
        failures.append("weak_stat_support")
    return failures


def _risk_tier(
    *,
    recommendation_type: str,
    market: Market | None,
    selection_name: str,
    selection_value: str | None,
    odds: float,
    expected_value: float,
    edge: float,
    confidence_score: float,
    data_quality_score: float,
) -> str:
    selection_lower = selection_name.lower()
    is_match_winner = market.market_type == "match_winner" if market is not None else "match winner" in selection_lower
    is_watchlist = recommendation_type == "watchlist"

    if (
        is_watchlist
        or odds > 3.40
        or confidence_score < 0.62
        or data_quality_score < 0.70
        or expected_value > 0.08
    ):
        return "high"

    if (
        1.40 <= odds <= 2.10
        and 0.010 <= expected_value <= 0.055
        and 0.0075 <= edge <= 0.045
        and confidence_score >= 0.70
        and data_quality_score >= 0.80
    ):
        return "low"

    if odds <= 3.40 and confidence_score >= 0.62 and data_quality_score >= 0.70:
        if is_match_winner and odds > 2.85 and confidence_score < 0.68:
            return "high"
        return "medium"
    return "high"


def _risk_adjusted_stake(stake_decimal: float, risk_tier: str, settings: Settings) -> float:
    if stake_decimal <= 0:
        return 0.0
    caps = {
        "low": settings.max_stake_pct,
        "medium": min(settings.max_stake_pct, 0.006),
        "high": min(settings.max_stake_pct, 0.003),
    }
    return min(stake_decimal, caps.get(risk_tier, 0.003))


def _supported_by_stats(stats: dict[str, TeamStats], match: Match) -> bool:
    home = stats[match.home_team]
    away = stats[match.away_team]
    has_xg = home.xg_avg is not None or away.xg_avg is not None
    has_shots = home.shots_avg is not None or away.shots_avg is not None
    goal_signal = abs(home.goals_for_avg - away.goals_against_avg) > 0.10 or abs(away.goals_for_avg - home.goals_against_avg) > 0.10
    form_signal = abs(home.form_last_5 - home.form_last_10) > 0.04 or abs(away.form_last_5 - away.form_last_10) > 0.04
    return has_xg or has_shots or goal_signal or form_signal


def _strong_selection_support(
    match: Match,
    market: Market,
    selection_name: str,
    selection_value: str | None,
    stats: dict[str, TeamStats],
    model: FootballProbabilityModel,
) -> bool:
    home = stats[match.home_team]
    away = stats[match.away_team]
    projection = model.expected_goals(match, stats)
    selection_lower = selection_name.lower()
    value = (selection_value or "").upper()
    has_core_signal = (home.xg_avg is not None or away.xg_avg is not None or home.shots_avg is not None or away.shots_avg is not None)
    if not has_core_signal:
        return False
    if market.market_type == "match_winner":
        if value == "H" or selection_lower == match.home_team.lower():
            return projection.home_goals - projection.away_goals >= 0.18 or away.goals_against_avg >= 1.45
        if value == "A" or selection_lower == match.away_team.lower():
            return projection.away_goals - projection.home_goals >= 0.18 or home.goals_against_avg >= 1.45
        return abs(projection.home_goals - projection.away_goals) <= 0.18
    if market.market_type == "over_under":
        line = _extract_total_line(market.name)
        if line is None:
            return False
        over25_rate = _mean_present(home.over25_rate, away.over25_rate)
        if "over" in selection_lower:
            if line == 1.5:
                return projection.total_goals >= 1.72 or min(projection.home_goals, projection.away_goals) >= 0.62
            if line == 2.5:
                return projection.total_goals >= 2.62 or (over25_rate is not None and over25_rate >= 0.52)
            if line == 3.5:
                return projection.total_goals >= 3.65 or (over25_rate is not None and over25_rate >= 0.68)
        if line == 1.5:
            return projection.total_goals <= 1.38 or max(home.failed_to_score_rate, away.failed_to_score_rate) >= 0.38
        if line == 2.5:
            return projection.total_goals <= 2.38 or (home.clean_sheet_rate + away.clean_sheet_rate) >= 0.42
        if line == 3.5:
            return projection.total_goals <= 3.30 or (over25_rate is not None and over25_rate <= 0.58)
        return False
    if market.market_type == "btts":
        btts_rate = _mean_present(home.btts_rate, away.btts_rate)
        if selection_lower in {"ja", "yes"}:
            return min(projection.home_goals, projection.away_goals) >= 1.0 or (btts_rate is not None and btts_rate >= 0.55)
        return min(projection.home_goals, projection.away_goals) <= 0.85 or (home.clean_sheet_rate + away.clean_sheet_rate) >= 0.45
    return False


def _passes_lean_filters(
    expected_value: float,
    edge: float,
    odds: float,
    confidence_score: float,
    data_quality_score: float,
    settings: Settings,
) -> bool:
    return (
        settings.lean_expected_value_threshold <= expected_value <= settings.lean_expected_value_ceiling
        and edge > max(settings.edge_threshold * 0.50, 0.005)
        and settings.min_odds <= odds <= settings.max_odds
        and confidence_score > settings.lean_confidence_threshold
        and data_quality_score > settings.data_quality_threshold
    )


def _selection_confidence_score(
    match: Match,
    stats: dict[str, TeamStats],
    market_type: str,
    model_probability: float,
    normalized_implied_probability: float,
    data_quality: float,
    market_confidence: float,
    odds: float = 2.0,
) -> float:
    home = stats[match.home_team]
    away = stats[match.away_team]
    edge_size = abs(model_probability - normalized_implied_probability)
    data_component = _clamp((data_quality - 0.60) / 0.35, 0.0, 1.0)
    agreement_component = _market_agreement_component(edge_size)
    stability_component = _input_stability_score(home, away)
    sample_component = _average(_clamp(home.sample_size / 10.0, 0.0, 1.0), _clamp(away.sample_size / 10.0, 0.0, 1.0))
    league_component = _average(home.league_reliability, away.league_reliability)
    market_component = _clamp(market_confidence / 0.80, 0.0, 1.0)
    score = (
        0.48
        + data_component * 0.09
        + agreement_component * 0.06
        + stability_component * 0.04
        + sample_component * 0.03
        + league_component * 0.02
        + market_component * 0.02
    )
    if edge_size > 0.08:
        score -= min((edge_size - 0.08) * 0.55, 0.05)
    if data_quality < 0.70:
        score -= (0.70 - data_quality) * 0.20
    if market_type == "match_winner" and model_probability > 0.62:
        score -= min((model_probability - 0.62) * 0.20, 0.035)
    if odds > 3.0:
        score -= min((odds - 3.0) * 0.025, 0.055)
    return round(_clamp(score, 0.50, 0.80), 4)


def _market_agreement_component(edge_size: float) -> float:
    if edge_size < 0.012:
        return 0.45
    if edge_size <= 0.055:
        return 0.88
    return _clamp(0.88 - ((edge_size - 0.055) / 0.10), 0.25, 0.88)


def _input_stability_score(home: TeamStats, away: TeamStats) -> float:
    form_volatility = abs(home.form_last_5 - home.form_last_10) + abs(away.form_last_5 - away.form_last_10)
    xg_goal_gaps = []
    for team in (home, away):
        if team.xg_avg is not None:
            xg_goal_gaps.append(abs(team.xg_avg - team.goals_for_avg))
        if team.xga_avg is not None:
            xg_goal_gaps.append(abs(team.xga_avg - team.goals_against_avg))
    xg_goal_volatility = _average(*xg_goal_gaps) if xg_goal_gaps else 0.75
    volatility = min(form_volatility / 0.80, 1.0) * 0.45 + min(xg_goal_volatility / 1.10, 1.0) * 0.55
    return _clamp(1.0 - volatility, 0.15, 1.0)


def _recommendation_reasons(
    match: Match,
    market: Market,
    selection_name: str,
    selection_value: str | None,
    model_probability: float,
    normalized_implied_probability: float,
    stats: dict[str, TeamStats],
    model: FootballProbabilityModel,
) -> list[str]:
    home = stats[match.home_team]
    away = stats[match.away_team]
    projection = model.expected_goals(match, stats)
    reasons: list[str] = []
    value = (selection_value or "").upper()
    selection_lower = selection_name.lower()

    if market.market_type == "match_winner":
        if value == "H" or selection_lower == match.home_team.lower():
            selected, opponent, selected_goals, opponent_goals = home, away, projection.home_goals, projection.away_goals
        elif value == "A" or selection_lower == match.away_team.lower():
            selected, opponent, selected_goals, opponent_goals = away, home, projection.away_goals, projection.home_goals
        else:
            selected, opponent, selected_goals, opponent_goals = home, away, projection.home_goals, projection.away_goals
            reasons.append(f"Draw model {model_probability * 100:.0f}% vs implied {normalized_implied_probability * 100:.0f}%")
            reasons.append(f"Projected goal gap only {abs(projection.home_goals - projection.away_goals):.2f}")
        xg_edge = selected_goals - opponent_goals
        if value != "D" and selection_lower not in {"draw", "uavgjort"}:
            reasons.append(f"Win model {model_probability * 100:.0f}% vs implied {normalized_implied_probability * 100:.0f}%")
        if xg_edge > 0.10:
            reasons.append(f"Projected goals: {selected.team} {selected_goals:.2f} vs {opponent.team} {opponent_goals:.2f}")
        selected_attack = selected.xg_avg if selected.xg_avg is not None else selected.goals_for_avg
        opponent_defense = opponent.xga_avg if opponent.xga_avg is not None else opponent.goals_against_avg
        if selected_attack > opponent_defense:
            reasons.append(f"{selected.team} xG {selected_attack:.2f} vs {opponent.team} xGA {opponent_defense:.2f}")
        if opponent.goals_against_avg >= 1.35 or (opponent.xga_avg is not None and opponent.xga_avg >= 1.35):
            reasons.append(f"{opponent.team} conceded {opponent.goals_against_avg:.2f} goals/match")
        if _momentum_supported(selected):
            reasons.append(f"{selected.team} form delta: last5 {selected.form_last_5:.2f} vs last10 {selected.form_last_10:.2f}")

    elif market.market_type == "over_under":
        line = _extract_total_line(market.name) or 2.5
        total_xg = projection.total_goals
        reasons.append(f"{selection_name} model {model_probability * 100:.0f}% vs implied {normalized_implied_probability * 100:.0f}%")
        if "over" in selection_lower:
            reasons.append(f"Projected total goals {total_xg:.2f} vs line {line:g}")
            over_rate = _mean_present(home.over25_rate, away.over25_rate)
            if over_rate is not None:
                reasons.append(f"Over 2.5 rate {over_rate * 100:.0f}% vs implied {normalized_implied_probability * 100:.0f}%")
            if projection.home_goals >= 1.1 and projection.away_goals >= 1.0:
                reasons.append(f"Projected scoring split {projection.home_goals:.2f}-{projection.away_goals:.2f}")
        else:
            reasons.append(f"Projected total goals {total_xg:.2f} vs line {line:g}")
            if home.clean_sheet_rate + away.clean_sheet_rate >= 0.45:
                reasons.append(f"Clean-sheet rates total {(home.clean_sheet_rate + away.clean_sheet_rate) * 100:.0f}%")
            if projection.home_goals < 1.25 or projection.away_goals < 1.05:
                reasons.append(f"Lower attack projection: {projection.home_goals:.2f}-{projection.away_goals:.2f}")
        if home.xga_avg is not None and away.xga_avg is not None and home.xga_avg + away.xga_avg >= 2.8:
            reasons.append(f"Combined defensive xGA {home.xga_avg + away.xga_avg:.2f}")

    elif market.market_type == "btts":
        reasons.append(f"BTTS model {model_probability * 100:.0f}% vs implied {normalized_implied_probability * 100:.0f}%")
        if selection_lower in {"ja", "yes"}:
            reasons.append(f"Projected scoring: home {projection.home_goals:.2f}, away {projection.away_goals:.2f}")
            if home.failed_to_score_rate <= 0.28 and away.failed_to_score_rate <= 0.28:
                reasons.append(f"Failed-to-score rates {home.failed_to_score_rate * 100:.0f}%/{away.failed_to_score_rate * 100:.0f}%")
            btts_rate = _mean_present(home.btts_rate, away.btts_rate)
            if btts_rate is not None:
                reasons.append(f"BTTS rate {btts_rate * 100:.0f}% vs implied {normalized_implied_probability * 100:.0f}%")
            if home.xga_avg is not None and away.xga_avg is not None and home.xga_avg + away.xga_avg >= 2.5:
                reasons.append(f"Combined defensive xGA {home.xga_avg + away.xga_avg:.2f}")
        else:
            reasons.append(f"Projected scoring suppresses BTTS at {projection.home_goals:.2f}-{projection.away_goals:.2f}")
            if home.clean_sheet_rate + away.clean_sheet_rate >= 0.45:
                reasons.append(f"Clean-sheet rates total {(home.clean_sheet_rate + away.clean_sheet_rate) * 100:.0f}%")
            if home.failed_to_score_rate >= 0.30 or away.failed_to_score_rate >= 0.30:
                reasons.append(f"Failed-to-score rates {home.failed_to_score_rate * 100:.0f}%/{away.failed_to_score_rate * 100:.0f}%")

    fallback_reasons = [
        f"Projected goals {projection.home_goals:.2f}-{projection.away_goals:.2f}",
        f"Data quality {data_quality_score(home, away):.2f} with mapping {min(home.mapping_confidence, away.mapping_confidence):.2f}",
    ]
    for reason in fallback_reasons:
        if len(reasons) >= 2:
            break
        if reason not in reasons:
            reasons.append(reason)
    return reasons[:4]


def _risk_labels(odds: float, profile_failure: str | None = None) -> list[str]:
    labels: list[str] = []
    if odds > MAX_PRODUCT_ODDS:
        labels.append("high_risk: odds above product cap")
    if profile_failure == "underdog_control":
        labels.append("underdog support incomplete")
    return labels


def _selection_profile_failure(
    match: Match,
    market: Market,
    selection_name: str,
    selection_value: str | None,
    odds: float,
    stats: dict[str, TeamStats],
    model: FootballProbabilityModel,
) -> str | None:
    projection = model.expected_goals(match, stats)
    home = stats[match.home_team]
    away = stats[match.away_team]
    value = (selection_value or "").upper()
    selection_lower = selection_name.lower()
    if market.market_type == "match_winner":
        if value == "H" or selection_lower == match.home_team.lower():
            if odds > 3.0 and not _underdog_supported(home, away, projection.home_goals, projection.away_goals, is_away=False):
                return "underdog_control"
            if projection.home_goals + 0.20 < projection.away_goals:
                return "xg_conflict"
        elif value == "A" or selection_lower == match.away_team.lower():
            if odds > 3.0 and not _underdog_supported(away, home, projection.away_goals, projection.home_goals, is_away=True):
                return "underdog_control"
            if projection.away_goals + 0.20 < projection.home_goals:
                return "xg_conflict"
        elif abs(projection.home_goals - projection.away_goals) > 0.55:
            return "xg_conflict"
    if market.market_type == "over_under":
        line = _extract_total_line(market.name)
        if line is None:
            return "missing_goal_line"
        if "over" in selection_lower and projection.total_goals < line - 0.25:
            return "xg_conflict"
        if "under" in selection_lower and projection.total_goals > line + 0.25:
            return "xg_conflict"
    if market.market_type == "btts":
        if selection_lower in {"ja", "yes"} and min(projection.home_goals, projection.away_goals) < 0.75:
            return "xg_conflict"
        if selection_lower in {"nei", "no"} and min(projection.home_goals, projection.away_goals) > 1.35:
            return "xg_conflict"
    return None


def _underdog_supported(
    selected: TeamStats,
    opponent: TeamStats,
    selected_goals: float,
    opponent_goals: float,
    is_away: bool,
) -> bool:
    xg_diff = selected_goals - opponent_goals
    opponent_weakness = opponent.goals_against_avg >= 1.30 or (opponent.xga_avg is not None and opponent.xga_avg >= 1.25)
    form_support = selected.form_last_5 >= selected.form_last_10 or selected.points_per_game_last_5 >= selected.points_per_game_last_10
    data_support = selected.mapping_confidence >= 0.68 and selected.stats_source != "footystats_league_fallback"
    away_penalty_ok = not is_away or xg_diff >= 0.12
    return xg_diff >= 0.02 and opponent_weakness and form_support and data_support and away_penalty_ok


def _momentum_supported(team: TeamStats) -> bool:
    xg_support = team.xg_trend is None or team.xg_trend >= -0.05
    return team.form_last_5 > team.form_last_10 and team.points_per_game_last_5 >= team.points_per_game_last_10 and xg_support


def _mean_present(*values: float | None) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _average(*values: float) -> float:
    return sum(values) / len(values) if values else 0.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def is_fake_match_id(match_id: str) -> bool:
    return bool(FAKE_MATCH_ID_RE.search(match_id))


def _market_only_stats(match: Match) -> dict[str, TeamStats]:
    missing = ["xg_avg", "xga_avg", "shots_avg", "shots_on_target_avg", "league_goals_avg"]
    return {
        match.home_team: TeamStats(
            team=match.home_team,
            stats_source="market_only",
            has_external_data=False,
            mapping_confidence=0.0,
            sample_size=0,
            league_reliability=0.55,
            missing_critical_stats=missing,
        ),
        match.away_team: TeamStats(
            team=match.away_team,
            stats_source="market_only",
            has_external_data=False,
            mapping_confidence=0.0,
            sample_size=0,
            league_reliability=0.55,
            missing_critical_stats=missing,
        ),
    }


def _matches_filters(match: Match, country: str | None, league: str | None, sport: str) -> bool:
    if sport and match.sport.lower() != sport.lower():
        return False
    if country and (match.country or "").lower() != country.lower():
        return False
    if league and match.league.lower() != league.lower():
        return False
    return True


def _match_window(match: Match, window_hours: int = 24, max_window_hours: int = 48) -> str:
    now = datetime.now(UTC)
    kickoff = match.kickoff_time
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=UTC)
    kickoff = kickoff.astimezone(UTC)
    primary_window_end = now + timedelta(hours=window_hours)
    secondary_window_end = now + timedelta(hours=max_window_hours)
    if now <= kickoff <= primary_window_end:
        return "primary"
    if primary_window_end < kickoff <= secondary_window_end:
        return "secondary"
    return "later"


def _recommendation_window(prediction: Prediction, window_hours: int = 24, max_window_hours: int = 48) -> str:
    return _match_window(
        Match(
            match_id=prediction.match_id,
            sport="football",
            league=prediction.league,
            country=prediction.country,
            home_team=prediction.match.split(" vs ")[0] if " vs " in prediction.match else prediction.match,
            away_team=prediction.match.split(" vs ")[1] if " vs " in prediction.match else "",
            kickoff_time=prediction.kickoff_time,
            status="upcoming",
            data_source=prediction.data_source,
            markets=[],
        ),
        window_hours=window_hours,
        max_window_hours=max_window_hours,
    )


def _map_rejection_reasons(reasons: list[str]) -> list[str]:
    mapped: list[str] = []
    for reason in reasons:
        if reason == "unresolved_team_mapping" or "mapping" in reason:
            mapped.append("mapping_failure")
        else:
            mapped.append(reason)
    return mapped


def _market_group(market_name: str) -> str:
    normalized = market_name.lower()
    if "begge lag scorer" in normalized or "btts" in normalized or "both teams" in normalized:
        return "BTTS"
    if "over/under" in normalized or "totalt antall mål" in normalized or "total goals" in normalized:
        line = _extract_total_line(market_name)
        if line is not None:
            if line == 1.5:
                return "Over/Under 1.5"
            if line == 2.5:
                return "Over/Under 2.5"
            if line == 3.5:
                return "Over/Under 3.5"
        return "Over/Under"
    if "match winner" in normalized or "hub" in normalized:
        return "Match Winner"
    return market_name


def _market_type_from_market_name(market_name: str) -> str:
    normalized = market_name.lower()
    if "begge lag scorer" in normalized or "btts" in normalized or "both teams" in normalized:
        return "btts"
    if "over/under" in normalized or "totalt antall" in normalized or "total goals" in normalized:
        return "over_under"
    return "match_winner"


def _market_thresholds(market_type: str) -> dict[str, float]:
    return MARKET_THRESHOLDS.get(market_type, MARKET_THRESHOLDS["match_winner"])


def _build_summary(
    matches: list[Match],
    future_matches: list[Match],
    recommendations: list[Prediction],
    stats_coverage: dict[str, int],
    debug_rejections: list[RejectionDetail] | None = None,
    window_hours: int = 24,
    max_window_hours: int = 48,
    include_debug_rows: bool = False,
) -> dict[str, object]:
    leagues = Counter(match.league for match in future_matches)
    markets = Counter(market.market_type for match in future_matches for market in match.markets)
    match_windows = Counter(_match_window(match, window_hours, max_window_hours) for match in future_matches)
    recommendation_windows = Counter(
        _recommendation_window(item, window_hours=window_hours, max_window_hours=max_window_hours)
        for item in recommendations
    )
    full = stats_coverage["matches_with_full_external_data"]
    future_count = len(future_matches)
    analyzed_count = stats_coverage["matches_analyzed"]
    summary: dict[str, object] = {
        "window_hours": window_hours,
        "max_window_hours": max_window_hours,
        "total_matches_loaded": len(matches),
        "future_matches": len(future_matches),
        "total_matches_in_window": match_windows.get("primary", 0),
        "matches_analyzed_in_window": min(stats_coverage["matches_analyzed"], match_windows.get("primary", 0)),
        "recommendations_in_window": recommendation_windows.get("primary", 0),
        "total_matches_in_24h": match_windows.get("primary", 0),
        "matches_in_24_48h": match_windows.get("secondary", 0),
        "matches_analyzed_24h": min(stats_coverage["matches_analyzed"], match_windows.get("primary", 0)),
        "recommendations_24h": recommendation_windows.get("primary", 0),
        "matches_with_full_data": full,
        "matches_analyzed": stats_coverage["matches_analyzed"],
        "leagues_found": dict(sorted(leagues.items())),
        "markets_found": dict(sorted(markets.items())),
        "external_stats_coverage": {
            **stats_coverage,
            "full_match_coverage_pct": round((full / analyzed_count) * 100.0, 2) if analyzed_count else 0.0,
            "usable_stats_coverage_pct": round(
                (stats_coverage["matches_with_usable_stats"] / analyzed_count) * 100.0,
                2,
            )
            if analyzed_count
            else 0.0,
            "team_mapping_coverage_pct": round(
                (stats_coverage["teams_matched"] / stats_coverage["teams_required"]) * 100.0,
                2,
            )
            if stats_coverage["teams_required"]
            else 0.0,
        },
        "recommendations_generated": sum(item.recommendation_type == "main" for item in recommendations),
        "main_bets_count": sum(item.recommendation_type == "main" for item in recommendations),
        "watchlist_count": sum(item.recommendation_type == "watchlist" for item in recommendations),
        "picks_by_market": dict(sorted(Counter(_market_group(item.market) for item in recommendations).items())),
        "picks_by_risk": dict(sorted(Counter(item.risk_tier for item in recommendations).items())),
    }
    if debug_rejections is not None:
        reason_counts = Counter(reason for item in debug_rejections for reason in item.reasons)
        rejected_by_market = Counter(_market_group(item.market) for item in debug_rejections if item.market)
        summary["rejection_reason_counts"] = dict(sorted(reason_counts.items()))
        summary["rejected_by_market"] = dict(sorted(rejected_by_market.items()))
        if include_debug_rows:
            summary["debug_rejections"] = [item.model_dump(mode="json") for item in debug_rejections]
    return summary


def _extract_total_line(market_name: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)", market_name)
    return float(match.group(1)) if match else None


def _save_processed_matches(matches: list[Match], data_source: str) -> Path:
    ensure_data_dirs()
    processed_path = PROCESSED_DIR / "latest_matches.json"
    processed_path.write_text(
        json.dumps(
            {
                "data_source": data_source,
                "matches": [match.model_dump(mode="json") for match in matches],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Saved normalized matches to %s", processed_path)
    return processed_path


def _save_recommendations(recommendations: list[Prediction]) -> None:
    ensure_data_dirs()
    settings = get_settings()
    json_path = RECOMMENDATIONS_DIR / "latest_recommendations.json"
    csv_path = RECOMMENDATIONS_DIR / "latest_recommendations.csv"
    rows = [item.model_dump(mode="json") for item in recommendations]
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    _append_recommendation_history(settings.recommendation_history_file, rows)
    logger.info("Saved recommendations to %s and %s", json_path, csv_path)


def _append_recommendation_history(path_value: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    path.parent.mkdir(parents=True, exist_ok=True)
    analysis_time = pd.Timestamp.now("UTC").isoformat()
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"analysis_time": analysis_time, **row}) + "\n")


def _print_summary(summary: dict[str, object]) -> None:
    print("\nAnalysis Summary")
    window_hours = summary.get("window_hours", 24)
    for key in (
        "total_matches_loaded",
        "future_matches",
        "matches_with_full_data",
        "matches_analyzed",
        "main_bets_count",
        "watchlist_count",
    ):
        print(f"{key}: {summary.get(key)}")
    if window_hours == 24:
        for key in ("total_matches_in_24h", "matches_in_24_48h", "matches_analyzed_24h", "recommendations_24h"):
            print(f"{key}: {summary.get(key)}")
    else:
        print(f"total_matches_in_{window_hours}h: {summary.get('total_matches_in_window')}")
        print(f"matches_analyzed_{window_hours}h: {summary.get('matches_analyzed_in_window')}")
        print(f"recommendations_{window_hours}h: {summary.get('recommendations_in_window')}")
    print(f"leagues_found: {_compact_distribution(summary.get('leagues_found', {}))}")
    print(f"markets_found: {_compact_distribution(summary.get('markets_found', {}))}")
    source_summary = summary.get("source_summary")
    if isinstance(source_summary, dict):
        print(
            "source_summary: "
            f"leagues_loaded={source_summary.get('leagues_loaded')}, "
            f"matches_loaded={source_summary.get('matches_loaded')}, "
            f"future_matches_in_window={source_summary.get('future_matches_in_window')}, "
            f"window_start={source_summary.get('window_start')}, "
            f"window_end={source_summary.get('window_end')}"
        )
    coverage = summary.get("external_stats_coverage")
    if isinstance(coverage, dict):
        print(f"external_stats_full_match_coverage: {coverage.get('full_match_coverage_pct', 0.0)}%")
        print(f"usable_stats_coverage: {coverage.get('usable_stats_coverage_pct', 0.0)}%")
        print(f"team_mapping_coverage: {coverage.get('team_mapping_coverage_pct', 0.0)}%")
    print(f"picks_by_market: {_compact_distribution(summary.get('picks_by_market', {}))}")
    print(f"picks_by_risk: {_compact_distribution(summary.get('picks_by_risk', {}))}")
    if "rejected_by_market" in summary:
        print(f"rejected_by_market: {_compact_distribution(summary.get('rejected_by_market', {}))}")
    if "rejection_reason_counts" in summary:
        print(f"rejection_reason_counts: {_compact_distribution(summary.get('rejection_reason_counts', {}))}")
        debug_items = summary.get("debug_rejections", [])
        if isinstance(debug_items, list) and debug_items:
            print("\n=== DEBUG (NOT SHOWN TO USERS) ===")
            for item in debug_items[:30]:
                if not isinstance(item, dict):
                    continue
                market = f" | {item.get('market')}" if item.get("market") else ""
                selection = f" | {item.get('selection')}" if item.get("selection") else ""
                reasons = ", ".join(item.get("reasons", []))
                print(f"- {item.get('match')} ({item.get('league')}){market}{selection}: {reasons}")
            if len(debug_items) > 30:
                print(f"... {len(debug_items) - 30} more rejection rows omitted from console output.")


def _print_recommendations(
    recommendations: list[Prediction],
    window_hours: int = 24,
    max_window_hours: int = 48,
) -> None:
    if not recommendations:
        print("No upcoming tiered outputs passed the configured filters.")
        return
    primary = [
        item
        for item in recommendations
        if _recommendation_window(item, window_hours=window_hours, max_window_hours=max_window_hours) == "primary"
    ]
    secondary = [
        item
        for item in recommendations
        if _recommendation_window(item, window_hours=window_hours, max_window_hours=max_window_hours) == "secondary"
    ]
    _print_recommendation_window(f"=== NEXT {window_hours}H BETS ===", primary)
    if secondary:
        _print_recommendation_window(f"=== NEXT {window_hours}-{max_window_hours}H BETS ===", secondary)

    visible_items = primary + secondary
    for item in visible_items:
        print("\n--------------------------------------------------")
        print(f"Type: {_display_tier(item.recommendation_type)}")
        print(f"Risk: {item.risk_tier.upper()}")
        print(f"Match: {item.match}")
        print(f"League: {item.league}")
        print(f"Kickoff: {item.kickoff_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"Data Source: {item.data_source}")
        print()
        print(f"Market: {item.market}")
        print(f"Pick: {item.selection}")
        print(f"Odds: {item.odds:.2f}")
        print()
        print(f"Model Probability: {_format_percent(item.model_probability)}")
        print(f"Implied Probability: {_format_percent(item.normalized_implied_probability)}")
        print(f"Edge: {_format_percent(item.edge, signed=True)}")
        print(f"EV: {_format_percent(item.expected_value, signed=True)}")
        print()
        print(f"Confidence: {item.confidence_score:.2f}")
        print(f"Data Quality: {item.data_quality_score:.2f}")
        print(f"Stake: {item.recommended_stake_pct:.1f}% bankroll")
        print()
        print("Reasons:")
        for reason in item.reason_codes[:4]:
            print(f"- {reason}")
        print("--------------------------------------------------")


def _print_recommendation_window(title: str, recommendations: list[Prediction]) -> None:
    print(f"\n{title}")
    main = [item for item in recommendations if item.recommendation_type == "main"]
    watchlist = [item for item in recommendations if item.recommendation_type == "watchlist"]
    _print_risk_grouped_tables(main)
    _print_recommendation_table("=== WATCHLIST (POSITIVE EV, NO BET YET) ===", watchlist)


def _print_risk_grouped_tables(recommendations: list[Prediction]) -> None:
    if not recommendations:
        print("\n=== MAIN BETS ===: none")
        return
    for risk_tier, title in (
        ("low", "=== LOW-RISK BETS ==="),
        ("medium", "=== MEDIUM-RISK BETS ==="),
        ("high", "=== HIGH-RISK BETS ==="),
    ):
        grouped = [item for item in recommendations if item.risk_tier == risk_tier]
        _print_recommendation_table(title, grouped)


def _print_recommendation_table(title: str, recommendations: list[Prediction]) -> None:
    if not recommendations:
        print(f"\n{title}: none")
        return
    print(f"\n{title}")
    print("| Match | Market | Pick | Risk | Odds | Edge | EV | Conf | Stake |")
    print("|------|--------|------|------|------|------|----|------|-------|")
    for item in recommendations:
        print(
            "| "
            f"{item.match} | {item.market} | {item.selection} | {item.risk_tier.upper()} | {item.odds:.2f} | "
            f"{_format_percent(item.edge, signed=True)} | {_format_percent(item.expected_value, signed=True)} | "
            f"{item.confidence_score:.2f} | {item.recommended_stake_pct:.1f}% |"
        )


def _display_tier(recommendation_type: str) -> str:
    if recommendation_type == "main":
        return "Main Bet"
    return "Watchlist"


def _compact_distribution(value: object, max_items: int = 12) -> str:
    if not isinstance(value, dict) or not value:
        return "{}"
    items = sorted(value.items(), key=lambda item: (-item[1], item[0]))[:max_items]
    rendered = ", ".join(f"{key}: {count}" for key, count in items)
    remaining = len(value) - len(items)
    if remaining > 0:
        rendered += f", ... +{remaining} more"
    return rendered


def _format_percent(value: float, signed: bool = False) -> str:
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value * 100:.1f}%"


if __name__ == "__main__":
    main()
