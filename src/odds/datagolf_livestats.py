"""
DataGolf live tournament SG stats — round-level strokes-gained during the event.

Used by the regression model: a round's score can be inflated by a hot putter
(low round-to-round repeatability) or deflated by a cold one, while ball-striking
(OTT + APP) is the sustainable part. Comparing a player's round SG *total* to
their *sustainable* SG tells us who over/under-performed and which way they'll
regress next round.
"""

from __future__ import annotations

import os

import requests

from ..match import match_player

BASE = "https://feeds.datagolf.com"


def fetch_round_stats(roster_index: dict, rnd: int, tour: str = "pga") -> dict:
    """{player: {ott, app, arg, putt, total, thru}} for a given completed round."""
    key = os.getenv("DATAGOLF_KEY")
    if not key:
        return {}
    try:
        r = requests.get(
            f"{BASE}/preds/live-tournament-stats",
            params={"stats": "sg_ott,sg_app,sg_arg,sg_putt,sg_total",
                    "round": str(rnd), "display": "value",
                    "file_format": "json", "key": key},
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json().get("live_stats", []) or []
    except Exception as e:
        print(f"[warn] datagolf live-stats r{rnd} failed: {e}")
        return {}

    out: dict[str, dict] = {}
    for x in rows:
        name = match_player(x.get("player_name", ""), roster_index)
        if not name or x.get("sg_total") is None:
            continue
        out[name] = {
            "ott": _f(x.get("sg_ott")), "app": _f(x.get("sg_app")),
            "arg": _f(x.get("sg_arg")), "putt": _f(x.get("sg_putt")),
            "total": _f(x.get("sg_total")), "thru": x.get("thru"),
        }
    return out


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
