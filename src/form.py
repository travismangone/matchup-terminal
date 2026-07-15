"""
Recent form — trailing 16- and 24-round strokes-gained.

Season SG is a stable baseline but slow to react; a player can be red-hot or ice
cold relative to it. This adds a rolling-window SG:Total over each player's last
16 and 24 measured rounds, walking recent events newest-first and accumulating
round-weighted SG (partial-weighting the event that crosses the boundary).

Data: per-event SG via the PGA Tour GraphQL (statDetails, EVENT_ONLY) — each call
returns every player's per-round SG + measured-round count for that event. Same
ShotLink limitation as everything else: only U.S. PGA Tour events are measured,
so events abroad (Scottish Open, etc.) return nothing and are skipped; LIV / DP-
only players have no recent-form signal (left None -> neutral in the model).

course_fit turns the recent-vs-season deviation into a skill nudge, so hot form
lifts a player and cold form dings them. Cached to data/form.json (refresh weekly).
"""

from __future__ import annotations

import json
import os

import requests

from .match import normalize_name, build_index, match_player

GRAPHQL = "https://orchestrator.pgatour.com/graphql"
KEY = "da2-gsrx5bibzbb4njvhl7t37wqyl4"
HDR = {"x-api-key": KEY, "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE = os.path.join(DATA, "form.json")

WINDOWS = (16, 24)
MAX_EVENTS = 18     # how far back to walk (enough to reach 24 rounds for regulars)


def _gq(query: str, variables: dict) -> dict:
    return requests.post(GRAPHQL, headers=HDR,
                         json={"query": query, "variables": variables}, timeout=30).json()


def recent_event_ids(max_events: int = MAX_EVENTS) -> list[str]:
    """Completed events, most recent first, across 2026 then 2025."""
    q = ("query S($t:String!,$y:String){schedule(tourCode:$t,year:$y)"
         "{completed{tournaments{id startDate}}}}")
    evs: list[tuple] = []
    for y in ("2026", "2025"):
        j = _gq(q, {"t": "R", "y": y})
        comp = (j.get("data") or {}).get("schedule", {}).get("completed", []) or []
        for m in comp:
            for t in m.get("tournaments", []):
                evs.append((t.get("startDate") or "0", t["id"]))
        if len(evs) >= max_events + 6:
            break
    evs.sort(key=lambda x: int(x[0]), reverse=True)
    return [i for _, i in evs[:max_events]]


def _event_sg(eid: str) -> dict[str, tuple[float, int]]:
    """{normalized_name: (per-round SG:Total, measured rounds)} for one event."""
    q = ("query D($t:TourCode!,$s:String!,$eq:StatDetailEventQuery){"
         "statDetails(tourCode:$t,statId:$s,eventQuery:$eq){rows{... on StatDetailsPlayer{"
         "playerName stats{statName statValue}}}}}")
    j = _gq(q, {"t": "R", "s": "02675",
                "eq": {"tournamentId": eid, "queryType": "EVENT_ONLY"}})
    rows = ((j.get("data") or {}).get("statDetails") or {}).get("rows", []) or []
    out: dict[str, tuple[float, int]] = {}
    for r in rows:
        st = {s["statName"]: s["statValue"] for s in r.get("stats", [])}
        name = normalize_name(r.get("playerName", ""))
        try:
            avg = float(st.get("Avg"))
            rounds = int(float(st.get("Measured Rounds", st.get("Rounds", 0))))
        except (TypeError, ValueError):
            continue
        if name and rounds > 0:
            out[name] = (avg, rounds)
    return out


def _window_avg(seq: list[tuple[float, int]], window: int) -> float | None:
    """Round-weighted SG over the last `window` rounds; None if too few rounds."""
    used = 0.0
    num = 0.0
    for avg, rounds in seq:               # seq is newest-first
        take = min(rounds, window - used)
        if take <= 0:
            break
        num += avg * take
        used += take
    if used < min(8, window * 0.5):       # need a real sample
        return None
    return round(num / used, 3)


def compute(max_events: int = MAX_EVENTS) -> dict[str, dict]:
    seq: dict[str, list[tuple[float, int]]] = {}     # name -> newest-first events
    kept = 0
    for eid in recent_event_ids(max_events):
        try:
            d = _event_sg(eid)
        except Exception as e:
            print(f"[warn] form event {eid}: {e}")
            continue
        if not d:
            continue                       # foreign / no-ShotLink event
        kept += 1
        for name, ar in d.items():
            seq.setdefault(name, []).append(ar)

    out: dict[str, dict] = {}
    for name, lst in seq.items():
        rec = {f"form_{w}": _window_avg(lst, w) for w in WINDOWS}
        rec["rounds"] = sum(r for _, r in lst)
        out[name] = rec
    print(f"  form: {len(out)} players from {kept} measured events")
    return out


def load(refresh: bool = False) -> dict[str, dict]:
    if not refresh and os.path.exists(CACHE):
        try:
            with open(CACHE) as f:
                return json.load(f)
        except Exception:
            pass
    data = compute()
    os.makedirs(DATA, exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(data, f)
    return data


def apply_to(players, data: dict[str, dict] | None = None) -> int:
    data = data if data is not None else load()
    index = build_index(list(data.keys()))
    matched = 0
    for p in players:
        key = match_player(p.name, index)
        if key:
            rec = data[key]
            p.form_16 = rec.get("form_16")
            p.form_24 = rec.get("form_24")
            if p.form_16 is not None or p.form_24 is not None:
                matched += 1
    return matched
