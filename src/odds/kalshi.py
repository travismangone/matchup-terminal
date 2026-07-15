"""
Kalshi golf outrights — read-only market data (no auth needed for quotes).

Kalshi files each Open market (winner, top-5/10/20, make-cut) as its OWN event
(e.g. KXPGATOUR-THOC26 = winner, KXPGATOP10-THOC26 = top-10) whose markets are one
binary per golfer; the Yes ask is that outcome's probability. Gotchas learned the
hard way:
  * events live under generic series (KXPGATOUR / KXPGATOP10 / ...), NOT
    golf-named ones, so you filter by EVENT ticker, not series;
  * tradeable golfer markets have status "active" (not "open"), while past
    champions who aren't playing show up "finalized" at $1 — filtered out here.

Set the per-market event tickers in config (KALSHI_EVENTS). Missing -> skipped.
"""

from __future__ import annotations

import requests

from . import Quote
from ..match import match_player

BASE = "https://api.elections.kalshi.com/trade-api/v2"


def fetch_quotes(
    roster_index: dict,
    events: dict[str, str],
    limit: int = 500,
) -> list[Quote]:
    """Pull every configured market (win + placements) into Quotes."""
    quotes: list[Quote] = []
    for market, event_ticker in (events or {}).items():
        if event_ticker:
            quotes.extend(_fetch_event(roster_index, market, event_ticker, limit))
    return quotes


def _fetch_event(roster_index: dict, market: str, event_ticker: str,
                 limit: int) -> list[Quote]:
    try:
        r = requests.get(
            f"{BASE}/markets",
            params={"limit": limit, "event_ticker": event_ticker},
            timeout=30,
        )
        r.raise_for_status()
        markets = r.json().get("markets", [])
    except Exception as e:
        print(f"[warn] kalshi {market} ({event_ticker}) fetch failed: {e}")
        return []

    quotes: list[Quote] = []
    for m in markets:
        # Only live golfer markets; skip finalized past-champion placeholders.
        if m.get("status") != "active":
            continue
        raw = m.get("yes_sub_title") or m.get("subtitle") or m.get("title") or ""
        player = match_player(raw, roster_index)
        if not player:
            continue
        prob = _yes_ask_prob(m)
        if prob is None:
            continue
        quotes.append(Quote(
            player=player, market=market,
            source="kalshi", source_kind="prediction_market",
            decimal_odds=1.0 / prob,
        ))
    return quotes


def _yes_ask_prob(market: dict) -> float | None:
    """Yes ask as probability 0-1, handling dollars (new) and cents (legacy)."""
    dollars = market.get("yes_ask_dollars")
    if dollars is not None:
        try:
            p = float(dollars)
            return p if 0.0 < p < 1.0 else None
        except (TypeError, ValueError):
            return None
    cents = market.get("yes_ask")
    if cents and 0 < cents < 100:
        return cents / 100.0
    return None
