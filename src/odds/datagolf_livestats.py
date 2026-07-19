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


def fetch_blended_stats(roster_index: dict, through_round: int,
                        recency: float = 1.5, tour: str = "pga") -> dict:
    """
    Blend per-round SG components across rounds 1..through_round, weighting recent
    rounds more (weight = recency ** (round-1)). This is the tournament-to-date
    form signal for the regression model — steadier than a single round, but still
    leaning on how a player has trended into the next round. Only rounds a player
    actually has stats for count toward their blend.
    """
    if through_round < 1:
        return {}
    per_round = {r: fetch_round_stats(roster_index, r, tour)
                 for r in range(1, through_round + 1)}
    weights = {r: recency ** (r - 1) for r in per_round}

    out: dict[str, dict] = {}
    names = {n for rd in per_round.values() for n in rd}
    for name in names:
        acc = {"ott": 0.0, "app": 0.0, "arg": 0.0, "putt": 0.0, "total": 0.0}
        wsum = 0.0
        for r, rd in per_round.items():
            s = rd.get(name)
            if not s:
                continue
            w = weights[r]
            for k in acc:
                acc[k] += w * s[k]
            wsum += w
        if wsum <= 0:
            continue
        out[name] = {k: v / wsum for k, v in acc.items()}
        out[name]["rounds"] = sum(1 for rd in per_round.values() if name in rd)
    return out


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
