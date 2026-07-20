"""
DataGolf player decompositions — the skill source.

/preds/player-decompositions returns, for the CURRENT event's course, each
player's baseline skill plus DataGolf's own data-driven course-fit, course-
history, timing and age adjustments, all in strokes-gained vs. an average PGA
field per round — and a per-player round-to-round standard deviation.

Why this replaced our hand-tuned course_fit layer: backtesting the Open showed
the elaborate links re-weighting predicted actual results no better than raw
season SG (spearman +0.603 vs +0.605), and the form nudge was actively negative
(-0.111). DataGolf's own fit adjustments span only ±0.34 strokes (SD 0.13)
against a skill spread of SD 0.91 — course fit is a ~14% effect, not the
headline. So we stand on their baseline and spend our edge on the in-round
regression model and DFS construction, which did validate.
"""

from __future__ import annotations

import os

import requests

from .match import match_player

BASE = "https://feeds.datagolf.com"


def fetch(roster_index: dict, tour: str = "pga") -> dict:
    """
    {player: {final, baseline, sd, fit, history, timing, dist, acc, sample}}
    `final` is the skill to simulate; `sd` is that player's round SD.
    Empty dict if unavailable (caller falls back to the course-fit model).
    """
    key = os.getenv("DATAGOLF_KEY")
    if not key:
        return {}
    try:
        r = requests.get(f"{BASE}/preds/player-decompositions",
                         params={"tour": tour, "file_format": "json", "key": key},
                         timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[warn] datagolf decompositions failed: {e}")
        return {}

    out: dict[str, dict] = {}
    for x in data.get("players", []) or []:
        name = match_player(x.get("player_name", ""), roster_index)
        if not name or x.get("final_pred") is None:
            continue
        out[name] = {
            "final": _f(x.get("final_pred")),
            "baseline": _f(x.get("baseline_pred")),
            "sd": _f(x.get("std_deviation")) or None,
            "fit": _f(x.get("total_fit_adjustment")),
            "history": _f(x.get("total_course_history_adjustment")),
            "timing": _f(x.get("timing_adjustment")),
            "dist": _f(x.get("driving_distance_adjustment")),
            "acc": _f(x.get("driving_accuracy_adjustment")),
            "sample": x.get("sample_size"),
        }
    if out:
        print(f"[skill] DataGolf decompositions: {len(out)} players "
              f"({data.get('event_name')} @ {data.get('course_name')})")
    return out


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0
