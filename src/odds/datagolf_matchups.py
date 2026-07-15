"""
DataGolf head-to-head (tournament matchup) odds.

betting-tools/matchups?market=tournament_matchups — one entry per booked matchup
(p1 vs p2, better 72-hole finish), with per-book p1/p2 prices. Most are "void"
ties (dead-heat refunds → a clean 2-way market); a few offer the tie separately.
FanDuel + Pinnacle both post matchups, so we can grade against the same sharp
reference as the outrights.

Names are matched to the field roster so they line up with the model's skills.
"""

from __future__ import annotations

import os

import requests

from ..match import match_player
from ..odds_math import american_to_decimal

BASE = "https://feeds.datagolf.com"


def fetch_matchups(roster_index: dict, tour: str = "pga") -> list[dict]:
    key = os.getenv("DATAGOLF_KEY")
    if not key:
        return []
    try:
        r = requests.get(
            f"{BASE}/betting-tools/matchups",
            params={"tour": tour, "market": "tournament_matchups",
                    "odds_format": "american", "file_format": "json", "key": key},
            timeout=30,
        )
        r.raise_for_status()
        match_list = r.json().get("match_list", []) or []
    except Exception as e:
        print(f"[warn] datagolf matchups failed: {e}")
        return []

    out: list[dict] = []
    for m in match_list:
        p1 = match_player(m.get("p1_player_name", ""), roster_index)
        p2 = match_player(m.get("p2_player_name", ""), roster_index)
        if not p1 or not p2:
            continue
        books: dict[str, dict] = {}
        for book, o in (m.get("odds") or {}).items():
            d1, d2 = _dec(o.get("p1")), _dec(o.get("p2"))
            if d1 and d2:
                books[book] = {"p1": d1, "p2": d2}
        # 'datagolf' is their model line, not a bookable price — keep separate.
        dg = books.pop("datagolf", None)
        if books:
            out.append({"p1": p1, "p2": p2, "ties": m.get("ties"),
                        "books": books, "dg": dg})
    return out


def _dec(v) -> float | None:
    if not v or not isinstance(v, str) or v.strip() in ("", "-"):
        return None
    try:
        return american_to_decimal(int(v.replace("+", "")))
    except ValueError:
        return None
