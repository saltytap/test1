from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient

from app import create_app
from run_daily import export_latest_picks
from src.data_models import Prediction


def test_run_daily_export_creates_latest_picks_json(tmp_path) -> None:
    output_path = tmp_path / "latest_picks.json"
    now = datetime.now(UTC)

    payload = export_latest_picks(
        [
            _prediction(now + timedelta(hours=3), expected_value=0.042),
            _prediction(now + timedelta(hours=4), expected_value=-0.01),
            _prediction(now - timedelta(minutes=5), expected_value=0.05),
        ],
        output_path=output_path,
        now=now,
    )

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == payload
    assert len(saved["picks"]) == 1
    assert saved["picks"][0] == {
        "match": "Team A vs Team B",
        "bet_type": "Over 2.5",
        "pick": "Over 2.5",
        "odds": 1.95,
        "ev": "+4.2%",
    }


def test_api_picks_returns_file_contents(tmp_path) -> None:
    picks_path = tmp_path / "latest_picks.json"
    picks_path.write_text(
        json.dumps(
            {
                "last_updated": "2026-04-27T08:00:00+00:00",
                "picks": [{"match": "A vs B", "bet_type": "BTTS", "pick": "Yes", "odds": 1.8, "ev": "+2.1%"}],
            }
        ),
        encoding="utf-8",
    )

    client = TestClient(create_app(picks_path=picks_path))
    response = client.get("/api/picks")

    assert response.status_code == 200
    assert response.json()["picks"][0]["match"] == "A vs B"


def test_api_picks_returns_empty_list_when_file_missing(tmp_path) -> None:
    client = TestClient(create_app(picks_path=tmp_path / "missing.json"))
    response = client.get("/api/picks")

    assert response.status_code == 200
    assert response.json() == {"last_updated": None, "picks": []}


def test_api_picks_refreshes_when_file_missing_and_key_exists(tmp_path, monkeypatch) -> None:
    picks_path = tmp_path / "latest_picks.json"

    def fake_run_daily(output_path):
        output_path.write_text(
            json.dumps(
                {
                    "last_updated": "2026-04-27T07:00:00+00:00",
                    "picks": [{"match": "A vs B", "bet_type": "Over 2.5", "pick": "Over 2.5", "odds": 1.9, "ev": "+1.2%"}],
                }
            ),
            encoding="utf-8",
        )
        return {}

    monkeypatch.setenv("FOOTYSTATS_API_KEY", "test-key")
    monkeypatch.setattr("run_daily.run_daily", fake_run_daily)

    client = TestClient(create_app(picks_path=picks_path))
    response = client.get("/api/picks")

    assert response.status_code == 200
    assert response.json()["picks"][0]["pick"] == "Over 2.5"


def _prediction(kickoff: datetime, expected_value: float) -> Prediction:
    return Prediction(
        kickoff_time=kickoff,
        league="Example League",
        country="Example",
        data_source="footystats",
        match_id="real-1",
        market_id="ou25",
        selection_id="over25",
        match="Team A vs Team B",
        market="Over/Under 2.5",
        selection="Over 2.5",
        bet_on="Over 2.5",
        odds=1.95,
        model_probability=0.54,
        raw_implied_probability=0.5128,
        normalized_implied_probability=0.50,
        edge=0.04,
        expected_value=expected_value,
        confidence_score=0.72,
        data_quality_score=0.86,
        recommended_stake_pct=0.5,
        recommended_stake_amount=50.0,
        recommendation_type="main",
        risk_tier="medium",
        reason_codes=["Over 2.5 model 54% vs implied 50%"],
    )
