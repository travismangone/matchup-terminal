"""
Historical backtest — does the SG → simulation pipeline produce *calibrated*
probabilities? (When it says 20%, does it happen ~20% of the time?)

Data, all free from the PGA Tour GraphQL:
  * schedule(tourCode, year)  -> completed tournament IDs for a season
  * leaderboardV3(id)         -> final finish positions for one event
  * statDetails (SG:Total)    -> per-round skill for a season

Method (no lookahead): to score season Y, we feed each event the field's skill
from season Y−1, simulate, and compare predicted win / top-N / make-cut
probabilities to what actually happened. Accumulated across a whole season that's
thousands of player-events — enough for a real reliability curve + Brier score.

Scope / honesty:
  * Runs with NEUTRAL (equal-weight) skill = SG:Total, so it validates the
    ENGINE (SG -> sim -> probabilities), not the Birkdale links weighting — that
    thesis needs links-only events, too sparse to calibrate here.
  * Players without prior-season SG get replacement level (keeps field size
    right). No historical OWGR, so no LIV fill — fine, this tests the core.
  * make_cut is approximate: the sim uses a top-70 cut; real events vary.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from .match import build_index, match_player
from . import simulate

GRAPHQL = "https://orchestrator.pgatour.com/graphql"
KEY = "da2-gsrx5bibzbb4njvhl7t37wqyl4"
HDR = {"x-api-key": KEY, "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}


def _gq(query: str, variables: dict) -> dict:
    r = requests.post(GRAPHQL, headers=HDR, json={"query": query, "variables": variables},
                      timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_schedule(year: int, tour: str = "R") -> list[tuple[str, str]]:
    q = ("query S($t:String!,$y:String){schedule(tourCode:$t,year:$y)"
         "{completed{tournaments{id tournamentName}}}}")
    j = _gq(q, {"t": tour, "y": str(year)})
    months = (j.get("data") or {}).get("schedule", {}).get("completed", []) or []
    return [(t["id"], t["tournamentName"]) for m in months for t in m.get("tournaments", [])]


def fetch_results(tid: str) -> list[tuple[str, int | None]]:
    """[(player_name, finish_position or None)] — None = missed cut / WD / DQ."""
    q = ("query L($id:ID!){leaderboardV3(id:$id){players{... on PlayerRowV3{"
         "player{displayName} scoringData{position}}}}}")
    j = _gq(q, {"id": tid})
    rows = (j.get("data") or {}).get("leaderboardV3", {}).get("players", []) or []
    out = []
    for r in rows:
        name = (r.get("player") or {}).get("displayName")
        pos = _parse_pos((r.get("scoringData") or {}).get("position"))
        if name:
            out.append((name, pos))
    return out


def _parse_pos(p) -> int | None:
    if not p:
        return None
    s = str(p).strip().upper().lstrip("T")
    return int(s) if s.isdigit() else None     # CUT / WD / DQ / MDF -> None


def season_sg_total(year: int, tour: str = "R") -> dict[str, float]:
    """{normalized_name: per-round SG:Total} for a season (statId 02675)."""
    from .match import normalize_name
    q = ("query D($t:TourCode!,$s:String!,$y:Int){statDetails(tourCode:$t,statId:$s,year:$y)"
         "{rows{... on StatDetailsPlayer{playerName stats{statName statValue}}}}}")
    j = _gq(q, {"t": tour, "s": "02675", "y": year})
    rows = (j.get("data") or {}).get("statDetails", {}).get("rows", []) or []
    out = {}
    for r in rows:
        n = normalize_name(r.get("playerName", ""))
        avg = next((s["statValue"] for s in r.get("stats", []) if s["statName"] == "Avg"), None)
        if n and avg is not None:
            try:
                out[n] = float(avg)
            except ValueError:
                pass
    return out


# --------------------------------------------------------------------------
@dataclass
class Backtest:
    target_year: int
    n_events: int
    n_player_events: int
    brier: dict[str, float]
    base_rate: dict[str, float]
    reliability: dict[str, list[tuple[float, float, int]]]  # market -> [(pred, obs, n)]
    coverage: str


MARKETS = ["win", "top_5", "top_10", "top_20", "make_cut"]


def run(target_year: int, max_events: int = 25, tour: str = "R",
        throttle: float = 0.25) -> Backtest:
    skills_prior = season_sg_total(target_year - 1, tour)
    skill_index = build_index(list(skills_prior.keys()))
    replacement = min(skills_prior.values()) if skills_prior else -1.5

    schedule = fetch_schedule(target_year, tour)[:max_events]
    pairs = {m: [] for m in MARKETS}     # market -> [(pred_prob, outcome 0/1)]
    n_events = matched = total = 0

    for tid, name in schedule:
        try:
            results = fetch_results(tid)
        except Exception as e:
            print(f"[warn] results {tid} {name}: {e}")
            continue
        if len(results) < 30:
            continue
        # Build field skills from PRIOR season (point-in-time).
        field_names, skills = [], {}
        for pname, _pos in results:
            key = match_player(pname, skill_index)
            skills[pname] = skills_prior[key] if key else replacement
            field_names.append(pname)
            if key:
                matched += 1
            total += 1
        sim = simulate.simulate(skills)
        probs = {r["name"]: r for r in sim.as_dicts()}

        for pname, pos in results:
            pr = probs.get(pname)
            if not pr:
                continue
            finished = pos is not None
            outcome = {
                "win": 1 if pos == 1 else 0,
                "top_5": 1 if (finished and pos <= 5) else 0,
                "top_10": 1 if (finished and pos <= 10) else 0,
                "top_20": 1 if (finished and pos <= 20) else 0,
                "make_cut": 1 if finished else 0,
            }
            for m in MARKETS:
                pairs[m].append((pr[m], outcome[m]))
        n_events += 1
        if throttle:
            time.sleep(throttle)

    brier, base, rel = {}, {}, {}
    for m in MARKETS:
        ps = pairs[m]
        if not ps:
            continue
        brier[m] = sum((p - o) ** 2 for p, o in ps) / len(ps)
        base[m] = sum(o for _p, o in ps) / len(ps)
        rel[m] = reliability(ps)
    cov = f"{matched}/{total} player-events had prior-season SG ({matched/total*100:.0f}%)" \
        if total else "no data"
    return Backtest(target_year, n_events, len(pairs["win"]), brier, base, rel, cov)


def reliability(pairs: list[tuple[float, float]], n_buckets: int = 10):
    """
    QUANTILE buckets (equal count), sorted by predicted probability ->
    [(mean_pred, observed_freq, count)]. Equal-width deciles are useless for rare
    events (every win prob < 10% lands in one bucket); equal-count groups spread
    the predictions so a real calibration curve emerges.
    """
    s = sorted(pairs, key=lambda x: x[0])
    if not s:
        return []
    n = len(s)
    size = max(1, n // n_buckets)
    out = []
    for i in range(0, n, size):
        chunk = s[i:i + size]
        if len(chunk) < size // 2 and out:      # fold a tiny trailing chunk back
            break
        mp = sum(p for p, _ in chunk) / len(chunk)
        mo = sum(o for _, o in chunk) / len(chunk)
        out.append((mp, mo, len(chunk)))
    return out
