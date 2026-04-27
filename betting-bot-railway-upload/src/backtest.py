from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR, Settings, ensure_data_dirs
from src.data_models import HistoricalPrediction


def load_historical_predictions(path_value: str) -> list[HistoricalPrediction]:
    path = Path(path_value)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    if not path.exists():
        raise RuntimeError(f"Backtest history file not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("predictions", [])
        return [HistoricalPrediction.model_validate(row) for row in rows]

    df = pd.read_csv(path, dtype={"match_id": str, "selection_id": str})
    return [HistoricalPrediction.model_validate(row) for row in df.to_dict(orient="records")]


def run_backtest(
    history: list[HistoricalPrediction] | None = None,
    bankroll: float = 10_000.0,
    history_path: str | None = None,
) -> dict[str, float | bool]:
    if history is None:
        if history_path is None:
            history_path = Settings().backtest_history_file
        history = load_historical_predictions(history_path)

    rows = []
    equity = bankroll
    peak = bankroll
    max_drawdown = 0.0
    for item in history:
        stake = bankroll * item.stake_pct
        profit = stake * (item.odds - 1.0) if item.actual_result else -stake
        equity += profit
        peak = max(peak, equity)
        if peak:
            max_drawdown = max(max_drawdown, (peak - equity) / peak)
        closing_line_value = (1.0 / item.closing_odds) - (1.0 / item.odds)
        rows.append(
            {
                "match_id": item.match_id,
                "selection_id": item.selection_id,
                "stake": stake,
                "profit": profit,
                "win": item.actual_result,
                "closing_line_value": closing_line_value,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Backtest history is empty.")

    total_staked = float(df["stake"].sum())
    total_profit = float(df["profit"].sum())
    roi = total_profit / total_staked if total_staked else 0.0
    win_rate = float(df["win"].mean())
    avg_clv = float(df["closing_line_value"].mean())
    model_accepted = bool(len(df) >= 30 and roi > 0.0 and avg_clv > 0.0 and max_drawdown < 0.25)
    return {
        "bets": float(len(df)),
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi": round(roi, 4),
        "win_rate": round(win_rate, 4),
        "max_drawdown": round(max_drawdown, 4),
        "average_closing_line_value": round(avg_clv, 4),
        "model_accepted": model_accepted,
    }


def save_backtest_report(report: dict[str, float | bool]) -> Path:
    ensure_data_dirs()
    path = PROCESSED_DIR / "backtest_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
