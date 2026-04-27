from __future__ import annotations

from src.config import Settings
from src.main import analyze_command, analyze_matches, fetch_command, is_fake_match_id
from src.nt_api import NorskTippingClient, mocked_odds_response
from src.normalizer import normalize_odds_response


def test_market_only_mode_analyzes_without_external_stats() -> None:
    fetch_command(use_mock=True)
    matches = normalize_odds_response(mocked_odds_response(), data_source="mock")
    recommendations, summary = analyze_matches(
        matches=matches,
        markets=["match_winner", "over_under", "btts"],
        max_results=20,
        country=None,
        league=None,
        sport="football",
        min_ev=0.03,
        settings=Settings(FOOTBALL_STATS_PROVIDER="none", REQUIRE_EXTERNAL_STATS=False),
        production_mode=False,
    )

    assert isinstance(recommendations, list)
    assert summary["matches_analyzed"] > 0
    assert summary["external_stats_coverage"]["market_only_matches"] > 0


def test_require_external_stats_skips_missing_data() -> None:
    matches = normalize_odds_response(mocked_odds_response(), data_source="mock")
    recommendations, summary = analyze_matches(
        matches=matches,
        markets=["match_winner", "over_under", "btts"],
        max_results=20,
        country=None,
        league=None,
        sport="football",
        min_ev=0.03,
        settings=Settings(FOOTBALL_STATS_PROVIDER="none", REQUIRE_EXTERNAL_STATS=True),
        production_mode=False,
    )

    assert recommendations == []
    assert summary["matches_analyzed"] == 0


def test_does_not_default_to_eliteserien_only_and_reports_multiple_leagues() -> None:
    matches = normalize_odds_response(mocked_odds_response(), data_source="mock")
    recommendations, summary = analyze_matches(
        matches=matches,
        markets=["match_winner", "over_under", "btts"],
        max_results=20,
        country=None,
        league=None,
        sport="football",
        min_ev=0.03,
        settings=Settings(FOOTBALL_STATS_PROVIDER="none", REQUIRE_EXTERNAL_STATS=True),
        production_mode=False,
    )

    assert len(summary["leagues_found"]) > 1
    assert summary["external_stats_coverage"]["full_match_coverage_pct"] == 0.0


def test_real_fetch_does_not_silently_use_mock_data() -> None:
    client = NorskTippingClient(Settings(NT_BASE_URL="https://127.0.0.1:9/OddsenGameInfo/v1/api", NT_MAX_RETRIES=1))

    try:
        client.fetch_odds(use_mock=False)
    except RuntimeError as exc:
        assert "Failed GET" in str(exc)
    else:
        raise AssertionError("Expected real fetch to fail without falling back to mock")


def test_excludes_fake_demo_match_ids_in_real_mode() -> None:
    matches = normalize_odds_response(mocked_odds_response(), data_source="real_api")
    recommendations, summary = analyze_matches(
        matches=matches,
        markets=["match_winner", "over_under", "btts"],
        max_results=20,
        country=None,
        league=None,
        sport="football",
        min_ev=0.03,
        settings=Settings(FOOTBALL_STATS_PROVIDER="none", REQUIRE_EXTERNAL_STATS=True),
        production_mode=True,
    )

    assert is_fake_match_id("match-001")
    assert recommendations == []
    assert summary["future_matches"] == 0


def test_handles_empty_real_api_data() -> None:
    recommendations, summary = analyze_matches(
        matches=[],
        markets=["match_winner", "over_under", "btts"],
        max_results=20,
        country=None,
        league=None,
        sport="football",
        min_ev=0.03,
        settings=Settings(FOOTBALL_STATS_PROVIDER="none", REQUIRE_EXTERNAL_STATS=True),
        production_mode=True,
    )

    assert recommendations == []
    assert summary["total_matches_loaded"] == 0
    assert summary["recommendations_generated"] == 0
