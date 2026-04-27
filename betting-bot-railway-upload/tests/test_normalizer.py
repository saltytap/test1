from __future__ import annotations

from src.normalizer import normalize_odds_response


def test_normalizer_rejects_unsupported_market_types() -> None:
    payload = {
        "events": [
            {
                "id": "real-1",
                "sport": "football",
                "league": "England - Premier League",
                "homeTeam": "Arsenal",
                "awayTeam": "Chelsea",
                "kickoffTime": "2099-01-01T15:00:00+00:00",
                "markets": [
                    _market("m1", "Match Winner", ["Arsenal", "Draw", "Chelsea"]),
                    _market("m2", "Totalt antall mål - over/under 2.5", ["Over 2.5", "Under 2.5"]),
                    _market("m3", "Begge lag scorer", ["Ja", "Nei"]),
                    _market("m4", "1. omgang - HUB", ["Arsenal", "Draw", "Chelsea"]),
                    _market("m5", "Antall kort over/under 3.5", ["Over 3.5", "Under 3.5"]),
                    _market("m6", "HUB og antall mål", ["Arsenal og Over 2.5", "Draw og Under 2.5"]),
                    _market("m7", "Totalt antall Arsenal mål over/under 1.5", ["Over 1.5", "Under 1.5"]),
                ],
            }
        ]
    }

    match = normalize_odds_response(payload)[0]
    market_types = [market.market_type for market in match.markets]

    assert market_types.count("match_winner") == 1
    assert market_types.count("over_under") == 1
    assert market_types.count("btts") == 1
    assert market_types.count("unknown") == 4


def _market(market_id: str, name: str, selections: list[str]) -> dict[str, object]:
    return {
        "id": market_id,
        "name": name,
        "selections": [
            {"id": f"{market_id}-{index}", "name": selection, "odds": 2.0}
            for index, selection in enumerate(selections)
        ],
    }
