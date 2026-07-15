"""
The Odds API — sportsbook outrights for the event's winner market.

Same shape as edge-finder/src/sources/the_odds_api.py: one Quote per book per
player. FanDuel + Pinnacle come back in the same payload as every other book, so
the sharp reference and the soft books are all captured in a single request.

Note: The Odds API golf keys expose the WINNER outright. Top-5/10/20 and
make-cut generally aren't offered here, so those markets are priced from the
simulation only (see compare.py).
"""

from __future__ import annotations

import os

import requests

from . import Quote

BASE = "https://api.the-odds-api.com/v4"


def fetch_winner_outrights(
    sport_key: str,
    api_key: str | None = None,
    regions: str = "us,us2,uk,eu",
) -> tuple[list[Quote], int | None]:
    """Returns (quotes, remaining_credits). Raises if ODDS_API_KEY missing."""
    key = api_key or os.getenv("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY not set")

    r = requests.get(
        f"{BASE}/sports/{sport_key}/odds",
        params={
            "apiKey": key,
            "regions": regions,
            "markets": "outrights",
            "oddsFormat": "decimal",
        },
        timeout=30,
    )
    r.raise_for_status()
    remaining = _int(r.headers.get("x-requests-remaining"))

    quotes: list[Quote] = []
    for event in r.json():
        for book in event.get("bookmakers", []):
            source = book.get("key", "")
            for mkt in book.get("markets", []):
                if mkt.get("key") != "outrights":
                    continue
                for oc in mkt.get("outcomes", []):
                    name = oc.get("name", "").strip()
                    price = oc.get("price")
                    if not name or not price:
                        continue
                    quotes.append(Quote(
                        player=name,
                        market="win",
                        source=source,
                        source_kind="sportsbook",
                        decimal_odds=float(price),
                    ))
    return quotes, remaining


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
