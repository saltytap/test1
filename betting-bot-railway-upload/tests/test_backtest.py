from __future__ import annotations

import pandas as pd

from src.backtest import load_historical_predictions, run_backtest


def test_backtest_loads_history_and_tracks_required_metrics(tmp_path) -> None:
    path = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {
                "match_id": "1",
                "selection_id": "a",
                "odds": 2.0,
                "closing_odds": 1.9,
                "model_probability": 0.54,
                "actual_result": True,
                "stake_pct": 0.01,
            },
            {
                "match_id": "2",
                "selection_id": "b",
                "odds": 1.8,
                "closing_odds": 1.85,
                "model_probability": 0.58,
                "actual_result": False,
                "stake_pct": 0.01,
            },
        ]
    ).to_csv(path, index=False)

    history = load_historical_predictions(str(path))
    report = run_backtest(history=history, bankroll=1000)

    assert report["bets"] == 2
    assert "roi" in report
    assert "win_rate" in report
    assert "max_drawdown" in report
    assert "average_closing_line_value" in report
    assert report["model_accepted"] is False
