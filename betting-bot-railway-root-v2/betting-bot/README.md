# Norsk Tipping Oddsen Value Bot

Read-only Python analysis bot for configurable Norsk Tipping Oddsen data. It fetches documented public odds data, stores raw responses, normalizes matches and markets, optionally combines them with football stats, and produces explainable value-bet recommendations.

This project does not place bets, automate login, access private endpoints, reverse engineer APIs, or bypass anti-bot systems.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Do not commit `.env` or API keys. Configure `FOOTYSTATS_API_KEY` as an environment variable in production.

## Run

```powershell
python -m src.main fetch
python -m src.main fetch --mock
python -m src.main analyze --markets match_winner over_under btts --max-results 20 --debug
python -m src.main analyze --mock --markets match_winner over_under btts --max-results 20
python -m src.main backtest
python -m src.main export --format csv
```

FootyStats-only analysis:

```powershell
python -m src.main footystats-analyze --window-hours 24 --max-leagues 50 --max-results 30
```

Daily cloud export:

```powershell
python run_daily.py
```

This writes website-safe picks to `data/latest_picks.json`:

```json
{
  "last_updated": "2026-04-27T08:00:00+00:00",
  "picks": [
    {
      "match": "Team A vs Team B",
      "bet_type": "Over 2.5",
      "pick": "Over 2.5",
      "odds": 1.95,
      "ev": "+4.2%"
    }
  ],
  "parlays": [
    {
      "type": "high_probability_fun",
      "leg_count": 2,
      "combined_odds": 2.88,
      "estimated_hit_probability": 0.42,
      "estimated_ev": "+21.0%",
      "risk_note": "Estimated hit rate assumes independent legs; keep stakes small.",
      "legs": []
    }
  ]
}
```

Serve picks for BetTrack:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

Endpoint:

```text
GET /api/picks
```

Production mode never falls back to demo data. By default it uses the documented OddsenGameInfo API:

- Base URL: `https://api.norsk-tipping.no/OddsenGameInfo/v1/api`
- Sport discovery: `/sportTypes`
- Football events: `/events/FBL`
- Event markets: `/markets/{eventId}`

Analysis filters:

```powershell
python -m src.main analyze --sport football --country England --league "Premier League" --min-ev 0.05 --max-results 20
```

## Football Stats

External football stats are optional. By default the bot will still analyze real Norsk Tipping odds without an external stats API key using conservative `market_only` mode. Market-only recommendations have capped confidence and are based on no-vig odds calibration, not invented xG/team form.

To require external stats and skip matches without xG/shots/team data:

```powershell
REQUIRE_EXTERNAL_STATS=true
```

The production stats provider is API-Football/API-Sports. It is read-only and uses team search, recent completed fixtures, fixture statistics, shots, and expected goals when the league exposes xG.

Configure:

```powershell
FOOTBALL_STATS_PROVIDER=api_football
REQUIRE_EXTERNAL_STATS=false
API_FOOTBALL_KEY=your_key
API_FOOTBALL_RECENT_MATCHES=10
```

For the FootyStats production path, configure:

```powershell
FOOTBALL_STATS_PROVIDER=footystats
REQUIRE_EXTERNAL_STATS=true
FOOTYSTATS_API_KEY=your_key
BETTING_BOT_WINDOW_HOURS=24
BETTING_BOT_MAX_LEAGUES=50
BETTING_BOT_MAX_RESULTS=30
ALLOWED_ORIGINS=https://bettrack.org,http://localhost:3000
```

You can also use provider-neutral local JSON stats:

```powershell
FOOTBALL_STATS_PROVIDER=json
EXTERNAL_STATS_FILE=data/external_stats.json
MIN_TEAM_MAPPING_SCORE=0.82
```

Example `data/external_stats.json`:

```json
{
  "teams": [
    {
      "team": "Arsenal",
      "league": "England - Premier League",
      "country": "England",
      "elo": 1720,
      "xg_avg": 2.05,
      "xga_avg": 0.95,
      "goals_for_avg": 2.1,
      "goals_against_avg": 0.9,
      "shots_avg": 14.2,
      "shots_on_target_avg": 5.4,
      "league_goals_avg": 2.8,
      "sample_size": 12
    }
  ]
}
```

If no external stats are available and `REQUIRE_EXTERNAL_STATS=false`, the bot logs market-only coverage and may still return recommendations if strict EV/confidence filters pass. If `REQUIRE_EXTERNAL_STATS=true`, missing stats cause the match to be skipped.

## Model Notes

- Odds are converted to raw implied probability, then normalized per market to remove bookmaker margin.
- Match winner uses an Elo-style strength model blended with Poisson goal outcomes and draw calibration.
- Over/under and BTTS use Poisson goal probabilities from external xG/xGA when available, or conservative market-only baselines otherwise.
- Momentum is capped and only improves projections when supported by xG, opponent-adjusted form, and goal trends.
- Recommendations are filtered for upcoming matches only, positive EV, positive edge, confidence, data quality, and odds range.
- Mock recommendations are labeled with `data_source=mock`; real endpoint data is labeled `data_source=real_api`.
- Production mode rejects demo-like event IDs such as `match-001`, `demo`, `sample`, and `test`.
- Model probabilities are blended with no-vig market probabilities to avoid unrealistic EV spikes.
- EV above `0.08` requires stronger confidence; EV above `0.10` is rejected as an outlier by default.
- Staking uses fractional Kelly, clamped to 0.1% to 1.0% of bankroll.
- Backtests require `data/historical/predictions.csv` or a configured `BACKTEST_HISTORY_FILE`; there is no mocked production backtest.
- Daily exports include optional 2-3 leg fun parlay cards built only from positive-EV main picks with low/medium risk, stronger model probability, confidence, data quality, and distinct matches. Parlay hit rate is estimated from independent-leg probabilities and is not a guarantee.

## Railway Deployment

Deploy from GitHub after confirming `.env` is not tracked. If it was previously tracked, remove it from Git history/index before pushing:

```bash
git rm --cached .env
git commit -m "Remove local env file from repository"
```

Railway API service:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

Railway cron job, daily at 07:00 UTC:

```bash
python run_daily.py
```

Cron schedule:

```text
0 7 * * *
```

Required Railway environment variables:

```text
FOOTYSTATS_API_KEY=<your key>
BETTING_BOT_WINDOW_HOURS=24
BETTING_BOT_MAX_LEAGUES=50
BETTING_BOT_MAX_RESULTS=30
ALLOWED_ORIGINS=https://bettrack.org,http://localhost:3000
```

The API returns an empty safe response if `data/latest_picks.json` has not been created yet:

```json
{
  "last_updated": null,
  "picks": []
}
```

If `FOOTYSTATS_API_KEY` is configured, `GET /api/picks` also attempts one safe refresh when the file is missing. This makes BetTrack integration resilient if the cron job has not written the file yet.

BetTrack can fetch:

```text
https://<your-railway-api-domain>/api/picks
```
