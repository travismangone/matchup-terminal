"""
DataGolf field-updates — upcoming-round tee times + AM/PM wave per player.

Feeds the draw model: for the next round we need who's in the early vs late wave
and when they tee off, so we can pair each player with the wind forecast for the
window they're actually on the course.
"""

from __future__ import annotations

import os

import requests

from ..match import match_player

BASE = "https://feeds.datagolf.com"


def fetch_waves(roster_index: dict, rnd: int, tour: str = "pga") -> dict:
    """
    Returns {"date": 'YYYY-MM-DD' or None,
             "players": {name: {wave: 'early'|'late', teetime: 'HH:MM', hour: float,
                                 start_hole: int}}}
    for the given round. Empty if no tee times are posted yet.
    """
    key = os.getenv("DATAGOLF_KEY")
    if not key:
        return {"date": None, "players": {}}
    try:
        r = requests.get(
            f"{BASE}/field-updates",
            params={"tour": tour, "file_format": "json", "key": key},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[warn] datagolf field-updates failed: {e}")
        return {"date": None, "players": {}}

    date = None
    players: dict[str, dict] = {}
    for x in data.get("field", []) or []:
        name = match_player(x.get("player_name", ""), roster_index)
        if not name:
            continue
        for tt in x.get("teetimes", []) or []:
            if tt.get("round_num") != rnd:
                continue
            raw = tt.get("teetime")            # "2026-07-17 09:58"
            hh = _hour_of(raw)
            if raw and date is None:
                date = raw.split(" ")[0]
            players[name] = {
                "wave": tt.get("wave"),        # 'early' | 'late'
                "teetime": raw.split(" ")[1] if raw and " " in raw else None,
                "hour": hh,
                "start_hole": tt.get("start_hole"),
            }
            break
    return {"date": date, "players": players}


def _hour_of(raw):
    # "2026-07-17 09:58" -> 9.97
    try:
        hm = raw.split(" ")[1]
        h, m = hm.split(":")[:2]
        return int(h) + int(m) / 60.0
    except (AttributeError, IndexError, ValueError):
        return None
