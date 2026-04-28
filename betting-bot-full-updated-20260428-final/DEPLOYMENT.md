# Railway + BetTrack Deployment

This bot is read-only. It does not place bets, log in to betting accounts, or automate account actions.

## 1. Upload To GitHub

1. Extract the clean ZIP locally.
2. Create a new private GitHub repository.
3. Upload/push the extracted files.
4. Confirm these files are not in GitHub:
   - `.env`
   - `data/latest_picks.json`
   - `data/cache/`
   - `__pycache__/`
   - `.pytest_cache/`

If `.env` was ever tracked before, remove it before pushing:

```bash
git rm --cached .env
git commit -m "Remove local env file"
```

## 2. Create Railway API Service

1. Go to Railway.
2. Create a new project from the GitHub repository.
3. Set the API service start command:

```bash
uvicorn app:app --host 0.0.0.0 --port $PORT
```

The included `railway.json` already contains this command.

## 3. Add Railway Variables

Add these variables in Railway:

```text
FOOTYSTATS_API_KEY=<your FootyStats key>
BETTING_BOT_WINDOW_HOURS=24
BETTING_BOT_MAX_LEAGUES=50
BETTING_BOT_MAX_RESULTS=30
ALLOWED_ORIGINS=https://bettrack.org,http://localhost:3000
```

Do not commit real keys to GitHub.

## 4. Create Daily Cron Job

Create a Railway cron job that runs:

```bash
python run_daily.py
```

Schedule it for 07:00 UTC:

```text
0 7 * * *
```

The cron writes:

```text
data/latest_picks.json
```

## 5. Connect BetTrack

BetTrack should fetch:

```text
https://<your-railway-domain>/api/picks
```

Expected response:

```json
{
  "last_updated": "2026-04-27T07:00:00+00:00",
  "picks": [
    {
      "match": "Team A vs Team B",
      "bet_type": "Over 2.5",
      "pick": "Over 2.5",
      "odds": 1.95,
      "ev": "+4.2%"
    }
  ]
}
```

If the daily cron has not run yet, the API safely returns:

```json
{
  "last_updated": null,
  "picks": []
}
```

If `FOOTYSTATS_API_KEY` is set on the API service, `/api/picks` will also try to generate `latest_picks.json` once when the file is missing. This helps when Railway has restarted the container or the cron has not populated the file yet.

## 6. Verify

Run locally:

```bash
pip install -r requirements.txt
pytest
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000/api/picks
```
