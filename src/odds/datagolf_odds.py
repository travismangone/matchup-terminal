"""
DataGolf outright odds — the betting-tools/outrights feed.

Why this over The Odds API for golf:
  * carries BOTH FanDuel AND Pinnacle (The Odds API has no Pinnacle for golf),
    so we can use a real two-book sharp reference;
  * a dozen+ books in one call, for the EV scanner;
  * flat-rate (no per-call credits) and it lists an event's odds 1-2 weeks out,
    so continuous snapshotting captures the TRUE market opener.

Emits one Quote per book per player for the win market, names matched to the
field roster so they line up with the model's projections.
"""

from __future__ import annotations

import os

import requests

from . import Quote
from ..match import match_player
from ..odds_math import american_to_decimal

BASE = "https://feeds.datagolf.com"

# Real sportsbooks in the feed (exclude 'datagolf' model line + id/name fields).
BOOKS = [
    "fanduel", "pinnacle", "draftkings", "betmgm", "caesars", "bet365",
    "betonline", "betway", "bovada", "pointsbet", "skybet", "unibet",
    "williamhill", "betcris",
]


def fetch_winner_quotes(roster_index: dict, tour: str = "pga",
                        market: str = "win") -> list[Quote]:
    key = os.getenv("DATAGOLF_KEY")
    if not key:
        return []
    try:
        r = requests.get(
            f"{BASE}/betting-tools/outrights",
            params={"tour": tour, "market": market, "odds_format": "american",
                    "file_format": "json", "key": key},
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json().get("odds", []) or []
    except Exception as e:
        print(f"[warn] datagolf odds fetch failed: {e}")
        return []

    quotes: list[Quote] = []
    for row in rows:
        player = match_player(row.get("player_name", ""), roster_index)
        if not player:
            continue
        for book in BOOKS:
            am = _american(row.get(book))
            if am is None:
                continue
            quotes.append(Quote(
                player=player, market="win",
                source=book, source_kind="sportsbook",
                decimal_odds=american_to_decimal(am),
            ))
    return quotes


def _american(v) -> int | None:
    """'+700' -> 700, '-200' -> -200; None/'-'/blank -> None."""
    if not v or not isinstance(v, str) or v.strip() in ("", "-"):
        return None
    try:
        return int(v.replace("+", ""))
    except ValueError:
        return None
