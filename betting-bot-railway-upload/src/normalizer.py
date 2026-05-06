from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Iterable

from src.data_models import Market, Match, Selection
from src.value_engine import implied_probability, normalize_market_margin


def normalize_odds_response(payload: dict[str, Any], data_source: str = "real_api") -> list[Match]:
    """Normalize documented or mocked odds responses into internal models.

    Norsk Tipping response shapes may vary by endpoint, so this walks the whole
    JSON tree and accepts any event-like object with teams, kickoff time, and markets.
    """

    matches: list[Match] = []
    for event in _iter_event_like_objects(payload):
        normalized = _normalize_event(event, data_source)
        if normalized:
            matches.append(normalized)
    return matches


def _iter_event_like_objects(node: Any) -> Iterable[dict[str, Any]]:
    if isinstance(node, dict):
        if _looks_like_event(node):
            yield node
        for value in node.values():
            yield from _iter_event_like_objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_event_like_objects(item)


def _looks_like_event(node: dict[str, Any]) -> bool:
    return bool(
        (_first_present(node, "markets", "marketList", "betOffers") or _first_present(node, "mainMarket"))
        and _first_present(node, "kickoffTime", "kickoff_time", "startTime", "eventTime", "startsAt")
        and (
            (
                _first_present(node, "homeTeam", "home_team", "homeParticipant")
                and _first_present(node, "awayTeam", "away_team", "awayParticipant")
            )
            or _first_present(node, "participants", "competitors", "teams")
        )
    )


def _normalize_event(event: dict[str, Any], data_source: str) -> Match | None:
    kickoff_raw = _first_present(event, "kickoffTime", "kickoff_time", "startTime", "eventTime", "startsAt")
    if not kickoff_raw:
        return None
    try:
        kickoff_time = _parse_kickoff(kickoff_raw)
    except ValueError:
        return None

    home_team, away_team = _extract_teams(event)
    if not home_team or not away_team:
        return None

    markets = [_normalize_market(item) for item in _extract_markets(event)]
    markets = [market for market in markets if market is not None]
    if not markets:
        return None

    return Match(
        match_id=str(_first_present(event, "id", "eventId", "matchId", "gameId")),
        sport=_normalize_sport(str(_first_present(event, "sport", "sportName", "sportId") or "football")),
        league=_name_or_value(_first_present(event, "league", "leagueName", "competition", "competitionName", "tournamentName", "tournament")) or "unknown",
        country=_name_or_value(_first_present(event, "country", "countryName", "categoryName")),
        home_team=str(home_team),
        away_team=str(away_team),
        kickoff_time=kickoff_time,
        status=str(_first_present(event, "status", "eventStatus", "state") or "upcoming").lower(),
        data_source=data_source,
        markets=markets,
    )


def _normalize_market(market_payload: dict[str, Any]) -> Market | None:
    selections_payload = _first_present(market_payload, "selections", "outcomes", "options")
    if not isinstance(selections_payload, list):
        return None
    selections: list[Selection] = []
    for selection_payload in selections_payload:
        if not isinstance(selection_payload, dict):
            continue
        odds = _first_present(selection_payload, "odds", "decimalOdds", "price", "selectionOdds")
        if odds is None:
            return None
        try:
            decimal_odds = float(odds)
        except (TypeError, ValueError):
            return None
        selections.append(
            Selection(
                selection_id=str(_first_present(selection_payload, "id", "selectionId", "outcomeId")),
                name=str(_first_present(selection_payload, "name", "label", "outcome", "selectionName") or ""),
                decimal_odds=decimal_odds,
                selection_value=_optional_str(_first_present(selection_payload, "selectionValue", "value")),
                raw_implied_probability=implied_probability(decimal_odds),
            )
        )
    if not selections:
        return None
    name = str(_first_present(market_payload, "name", "marketName", "label") or "")
    return normalize_market_margin(
        Market(
            market_id=str(_first_present(market_payload, "id", "marketId", "betOfferId")),
            name=_display_market_name(name, market_payload),
            market_type=_market_type(name, market_payload),
            selections=selections,
        )
    )


def _extract_markets(event: dict[str, Any]) -> list[dict[str, Any]]:
    markets = _first_present(event, "markets", "marketList", "betOffers")
    if isinstance(markets, list):
        return markets
    main_market = _first_present(event, "mainMarket")
    return [main_market] if isinstance(main_market, dict) else []


def _extract_teams(event: dict[str, Any]) -> tuple[str | None, str | None]:
    home = _first_present(event, "homeTeam", "home_team", "homeParticipant")
    away = _first_present(event, "awayTeam", "away_team", "awayParticipant")
    if home and away:
        return str(home), str(away)
    participants = _first_present(event, "participants", "competitors", "teams")
    if isinstance(participants, list) and len(participants) >= 2:
        names = []
        for participant in participants:
            if isinstance(participant, dict):
                names.append(str(_first_present(participant, "name", "teamName", "participantName") or ""))
            else:
                names.append(str(participant))
        return names[0], names[1]
    return None, None


def _market_type(name: str, market_payload: dict[str, Any] | None = None) -> str:
    normalized = name.lower()
    line_type = str((market_payload or {}).get("marketLineType", "")).upper()
    selection_values = {
        str(item.get("selectionValue", "")).upper()
        for item in (market_payload or {}).get("selections", [])
        if isinstance(item, dict)
    }
    if _is_unsupported_market(normalized):
        return "unknown"
    if normalized in {"match winner", "1x2", "full time result", "hub"} or (
        selection_values == {"H", "D", "A"} and normalized in {"", "hub"}
    ):
        return "match_winner"
    if _is_full_match_goal_total(normalized, line_type):
        return "over_under"
    if normalized in {"both teams to score", "btts", "begge lag scorer"}:
        return "btts"
    if "double chance" in normalized:
        return "double_chance"
    if "handicap" in normalized:
        return "handicap"
    return "unknown"


def _is_unsupported_market(normalized: str) -> bool:
    unsupported_fragments = [
        "1. omgang",
        "1st half",
        "first half",
        "halvtid",
        "pause",
        "kort",
        "cards",
        "corner",
        "hjørne",
        "hjørnespark",
        "handikap",
        "handicap",
        "asian",
        "resultat og",
        "hub og",
        "hub &",
        " og over",
        " og under",
        "o/u",
        "team total",
    ]
    return any(fragment in normalized for fragment in unsupported_fragments)


def _is_full_match_goal_total(normalized: str, line_type: str) -> bool:
    if line_type == "TOTAL" and normalized in {"", "total goals", "over/under"}:
        return True
    if normalized.startswith("total goals"):
        return True
    if normalized.startswith("over/under"):
        return True
    return normalized.startswith("totalt antall mål") or normalized.startswith("totalt antall maal")


def _display_market_name(name: str, market_payload: dict[str, Any]) -> str:
    if name.lower() == "hub":
        return "Match Winner"
    return name


def _parse_kickoff(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1000
        return datetime.fromtimestamp(timestamp, UTC)
    text = str(value)
    if text.isdigit():
        timestamp = float(text)
        if timestamp > 10_000_000_000:
            timestamp /= 1000
        return datetime.fromtimestamp(timestamp, UTC)
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _normalize_sport(value: str) -> str:
    normalized = value.lower()
    if normalized in {"fbl", "fotball", "football", "soccer"}:
        return "football"
    return normalized


def _name_or_value(value: Any) -> str | None:
    if isinstance(value, dict):
        name = value.get("name")
        return str(name) if name else None
    return str(value) if value is not None else None


def _first_present(node: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in node and node[key] not in (None, ""):
            return node[key]
    return None


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None
