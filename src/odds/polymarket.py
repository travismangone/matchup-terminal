"""
Polymarket golf outrights via the Gamma API (no key needed).

Mirrors edge-finder/src/sources/polymarket.py::fetch_futures but emits this
project's Quote and maps each golfer name to the field roster via match.py. The
Open winner lives as an event whose sub-markets are one binary per golfer
("Will <Golfer> win the 2026 Open Championship?"); we take each Yes price as the
win probability and convert to decimal odds.

Docs: https://docs.polymarket.com
"""

from __future__ import annotations

import json
import re

import requests

from . import Quote
from ..match import match_player

BASE = "https://gamma-api.polymarket.com"

# Event titles we treat as "the Open winner" market.
DEFAULT_TITLE_CONTAINS = ["open championship", "british open"]


def fetch_winner_quotes(
    roster_index: dict,
    title_contains: list[str] | None = None,
    tag_slug: str = "golf",
) -> list[Quote]:
    titles = [t.lower() for t in (title_contains or DEFAULT_TITLE_CONTAINS)]
    try:
        r = requests.get(
            f"{BASE}/events",
            params={"limit": 100, "closed": "false", "active": "true",
                    "tag_slug": tag_slug},
            timeout=30,
        )
        r.raise_for_status()
        events = r.json()
    except Exception as e:
        print(f"[warn] polymarket golf fetch failed: {e}")
        return []
    if not isinstance(events, list):
        return []

    quotes: list[Quote] = []
    for ev in events:
        title = (ev.get("title") or "").lower()
        if not any(t in title for t in titles):
            continue
        for m in ev.get("markets", []):
            raw = _golfer_from_question(m.get("question", ""))
            player = match_player(raw, roster_index) if raw else None
            if not player:
                continue
            yes = _yes_price(_as_list(m.get("outcomes")), _as_list(m.get("outcomePrices")))
            if yes is None or not (0.0 < yes < 1.0):
                continue
            quotes.append(Quote(
                player=player, market="win",
                source="polymarket", source_kind="prediction_market",
                decimal_odds=1.0 / yes,
            ))
    return quotes


def _golfer_from_question(question: str) -> str | None:
    """'Will Rory McIlroy win the 2026 Open Championship?' -> 'Rory McIlroy'."""
    if not question:
        return None
    m = re.search(r"will\s+(.+?)\s+win\b", question, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return question.strip()


def _yes_price(outcomes, prices):
    if len(outcomes) != len(prices):
        return None
    for name, px in zip(outcomes, prices):
        if str(name).strip().lower() == "yes":
            try:
                return float(px)
            except (TypeError, ValueError):
                return None
    return None


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []
