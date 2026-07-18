"""
DataGolf in-play — live tournament state, for the in-tournament (round-based) DFS
model. /preds/in-play returns each player's per-round scores, current position /
score, holes played, and live finish odds; it updates through every round.

We use it to know who's still in, their current standing, and — by finding the
first round with no scores yet — which round is *next* to project.
"""

from __future__ import annotations

import os

import requests

from ..match import match_player

BASE = "https://feeds.datagolf.com"


# Non-numeric position strings that mean a player is out of the tournament.
_OUT_POS = {"CUT", "WD", "DQ", "DNS", "WAGR"}


def fetch_inplay(roster_index: dict, tour: str = "pga") -> dict:
    """
    Returns {"next_round": int|None, "players": {name: {...}}, "cut": {names}}.
    Player fields: current_pos, current_score, thru, round, r1..r4, make_cut.
    "cut" = players who missed the cut / withdrew / were DQ'd (treat as out).
    """
    key = os.getenv("DATAGOLF_KEY")
    if not key:
        return {"next_round": None, "players": {}, "cut": set()}
    try:
        r = requests.get(
            f"{BASE}/preds/in-play",
            params={"tour": tour, "dead_heat": "no", "odds_format": "american",
                    "file_format": "json", "key": key},
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json().get("data", []) or []
    except Exception as e:
        print(f"[warn] datagolf in-play failed: {e}")
        return {"next_round": None, "players": {}, "cut": set()}

    players: dict[str, dict] = {}
    round_has_score = {1: False, 2: False, 3: False, 4: False}
    for row in rows:
        name = match_player(row.get("player_name", ""), roster_index)
        if not name:
            continue
        rd_scores = {i: row.get(f"R{i}") for i in range(1, 5)}
        for i, v in rd_scores.items():
            if v is not None:
                round_has_score[i] = True
        players[name] = {
            "current_pos": row.get("current_pos"),
            "current_score": row.get("current_score"),
            "thru": row.get("thru"),
            "round": row.get("round"),
            "r1": rd_scores[1], "r2": rd_scores[2],
            "r3": rd_scores[3], "r4": rd_scores[4],
            "make_cut": row.get("make_cut"),
        }

    # Next round = the first round nobody has a score in yet (1..4), else None.
    next_round = next((i for i in range(1, 5) if not round_has_score[i]), None)

    # Who's out by position: DataGolf flips a missed-cut/withdrawn player's
    # current_pos to CUT/WD/DQ once official. (The dashboard layers on further
    # signals — no next-round tee time, 0% make-cut — in _out_of_event.)
    cut = {name for name, st in players.items()
           if str(st.get("current_pos") or "").upper() in _OUT_POS}
    return {"next_round": next_round, "players": players, "cut": cut}
