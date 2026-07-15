"""
Free strokes-gained ingestion — no DataGolf subscription required.

Two public sources, no API keys:

  1. PGA Tour stats GraphQL (orchestrator.pgatour.com) — real per-round SG
     components (OTT / APP / ARG / putting) + driving accuracy, ~180 players.
     Blended across the current + prior season for stability and coverage.

  2. OWGR rankings API (apiweb.owgr.com) — points-average for the whole world,
     INCLUDING LIV / DP World players who aren't in PGA Tour stats. We calibrate
     OWGR points -> expected SG on the players we have both for, then use that to
     estimate skill for anyone the PGA feed misses.

Output is a list of datagolf.Player, so course_fit / simulate consume it
unchanged. Players filled from OWGR carry an "owgr-est" flag — their overall
level is estimated but their component split (needed for links weighting) is the
tour-average shape, so treat their course-fit edge as low-confidence.

Limitation vs. DataGolf: no true off-tour SG components and no course-history
signal. This is the honest cost of the free route.
"""

from __future__ import annotations

import math
import os

import requests

from . import datagolf
from .datagolf import Player
from .match import normalize_name, build_index, match_player

# PGA Tour stats GraphQL. The x-api-key is the site's public AppSync key.
PGA_GRAPHQL = "https://orchestrator.pgatour.com/graphql"
PGA_KEY = "da2-gsrx5bibzbb4njvhl7t37wqyl4"
_STAT_QUERY = (
    "query StatDetails($tourCode: TourCode!, $statId: String!, $year: Int) "
    "{ statDetails(tourCode: $tourCode, statId: $statId, year: $year) "
    "{ statTitle rows { ... on StatDetailsPlayer "
    "{ playerName stats { statName statValue } } } } }"
)
STAT_IDS = {
    "sg_ott": "02567",   # SG: Off-the-Tee
    "sg_app": "02568",   # SG: Approach the Green
    "sg_arg": "02569",   # SG: Around-the-Green
    "sg_putt": "02564",  # SG: Putting
    "drv_acc": "102",    # Driving Accuracy Percentage
}

OWGR_URL = "https://apiweb.owgr.com/api/owgr/rankings/getRankings"

# Scoring-profile stats (percentages). Birdie-or-Better % = ceiling/upside;
# Bogey Avoidance = % of holes bogey-or-worse (floor — lower is better).
SCORING_STAT_IDS = {"birdie_pct": "352", "bogey_pct": "02414"}

# Tour-average component shares of SG total — used to split an OWGR-estimated
# total into the four components so links weighting still has something to act on.
COMPONENT_SHARES = {"sg_ott": 0.28, "sg_app": 0.38, "sg_arg": 0.14, "sg_putt": 0.20}


# --------------------------------------------------------------------------
# PGA Tour SG
# --------------------------------------------------------------------------
def _pga_stat(stat_id: str, year: int) -> dict[str, float]:
    """{normalized_name: per-round Avg} for one stat + season."""
    r = requests.post(
        PGA_GRAPHQL,
        headers={"x-api-key": PGA_KEY, "Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0"},
        json={"variables": {"tourCode": "R", "statId": stat_id, "year": year},
              "query": _STAT_QUERY},
        timeout=30,
    )
    r.raise_for_status()
    rows = (r.json().get("data") or {}).get("statDetails", {}).get("rows", []) or []
    out: dict[str, float] = {}
    for row in rows:
        name = normalize_name(row.get("playerName", ""))
        avg = next((s["statValue"] for s in row.get("stats", [])
                    if s["statName"] == "Avg"), None)
        if name and avg is not None:
            out[name] = _num(avg)
    return out


def fetch_pga_sg(years: tuple[int, ...] = (2026, 2025),
                 weights: tuple[float, ...] = (0.6, 0.4)) -> dict[str, dict]:
    """
    {normalized_name: {sg_ott, sg_app, sg_arg, sg_putt, drv_acc, name}} blended
    across seasons. A player present in only one season uses that season alone.
    """
    per_year: dict[int, dict[str, dict[str, float]]] = {}
    for y in years:
        cols = {}
        for field, sid in STAT_IDS.items():
            try:
                cols[field] = _pga_stat(sid, y)
            except Exception as e:
                print(f"[warn] PGA stat {field} {y} failed: {e}")
                cols[field] = {}
        # invert to per-player
        players: dict[str, dict[str, float]] = {}
        names = set().union(*[set(c) for c in cols.values()]) if cols else set()
        for n in names:
            players[n] = {f: cols[f].get(n) for f in STAT_IDS}
        per_year[y] = players

    all_names = set().union(*[set(p) for p in per_year.values()])
    out: dict[str, dict] = {}
    for n in all_names:
        blended: dict[str, float] = {}
        for field in STAT_IDS:
            num = den = 0.0
            for y, w in zip(years, weights):
                v = per_year[y].get(n, {}).get(field)
                if v is not None:
                    num += w * v
                    den += w
            if den > 0:
                blended[field] = num / den
        # Require the four SG components to count a player as PGA-covered.
        if all(k in blended for k in ("sg_ott", "sg_app", "sg_arg", "sg_putt")):
            blended["name"] = n
            out[n] = blended
    return out


def _pga_stat_pct(stat_id: str, year: int) -> dict[str, float]:
    """{normalized_name: primary % value} for a percentage stat (BoB%, bogey)."""
    r = requests.post(
        PGA_GRAPHQL,
        headers={"x-api-key": PGA_KEY, "Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0"},
        json={"variables": {"tourCode": "R", "statId": stat_id, "year": year},
              "query": _STAT_QUERY},
        timeout=30,
    )
    r.raise_for_status()
    rows = (r.json().get("data") or {}).get("statDetails", {}).get("rows", []) or []
    out: dict[str, float] = {}
    for row in rows:
        name = normalize_name(row.get("player_name", "") or row.get("playerName", ""))
        stats = row.get("stats", [])
        val = next((s["statValue"] for s in stats if s["statName"] == "Avg"), None)
        if val is None and stats:
            val = stats[0].get("statValue")     # percentage stats lead with the %
        if name and val is not None:
            out[name] = _num(val)
    return out


def fetch_scoring_stats(years: tuple[int, ...] = (2026, 2025),
                        weights: tuple[float, ...] = (0.6, 0.4)) -> dict[str, dict]:
    """{normalized_name: {birdie_pct, bogey_pct}} blended across seasons."""
    per_stat = {}
    for field_name, sid in SCORING_STAT_IDS.items():
        by_year = {}
        for y in years:
            try:
                by_year[y] = _pga_stat_pct(sid, y)
            except Exception as e:
                print(f"[warn] scoring stat {field_name} {y} failed: {e}")
                by_year[y] = {}
        per_stat[field_name] = by_year

    names = set()
    for field_name, by_year in per_stat.items():
        for y in by_year:
            names |= set(by_year[y])
    out: dict[str, dict] = {}
    for n in names:
        rec = {}
        for field_name, by_year in per_stat.items():
            num = den = 0.0
            for y, w in zip(years, weights):
                v = by_year.get(y, {}).get(n)
                if v is not None:
                    num += w * v
                    den += w
            if den > 0:
                rec[field_name] = round(num / den, 2)
        if rec:
            out[n] = rec
    return out


# --------------------------------------------------------------------------
# OWGR
# --------------------------------------------------------------------------
def fetch_owgr(max_players: int = 1000) -> dict[str, float]:
    """{normalized_name: pointsAverage} across the top world ranking."""
    try:
        r = requests.get(
            OWGR_URL,
            params={"pageSize": max_players, "pageNumber": 1,
                    "sortString": "Rank ASC", "regionId": 0, "countryId": 0,
                    "pageDivider": 1000},
            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json().get("rankingsList", [])
    except Exception as e:
        print(f"[warn] OWGR fetch failed: {e}")
        return {}
    out: dict[str, float] = {}
    for e in rows:
        name = normalize_name(e.get("player", {}).get("fullName", ""))
        pa = e.get("pointsAverage")
        if name and pa and pa > 0:
            out[name] = float(pa)
    return out


def _calibrate_owgr(pga: dict[str, dict], owgr: dict[str, float]) -> tuple[float, float]:
    """
    Fit sg_total ≈ a + b·ln(pointsAverage) on players present in both feeds.
    Returns (a, b). Falls back to a sane slope if the overlap is too small.
    """
    xs, ys = [], []
    for n, comp in pga.items():
        pa = owgr.get(n)
        if pa and pa > 0:
            xs.append(math.log(pa))
            ys.append(comp["sg_ott"] + comp["sg_app"] + comp["sg_arg"] + comp["sg_putt"])
    if len(xs) < 10:
        return (-1.2, 0.9)     # rough prior: ln(pts)=0 -> -1.2 SG, +1 ln -> +0.9
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.9
    a = my - b * mx
    return (a, b)


# --------------------------------------------------------------------------
# Base SG source: DataGolf (paid, covers LIV/global) when a key is present,
# else the free PGA Tour scrape. Both return {norm_name: {sg components + drv_acc}}.
# --------------------------------------------------------------------------
def _base_skill_map() -> tuple[dict[str, dict], str]:
    if os.getenv("DATAGOLF_KEY"):
        try:
            ratings = datagolf.fetch_skill_ratings()      # {name: Player}
            m = {n: {"sg_ott": p.sg_ott, "sg_app": p.sg_app, "sg_arg": p.sg_arg,
                     "sg_putt": p.sg_putt, "drv_acc": p.driving_acc}
                 for n, p in ratings.items()}
            if m:
                print(f"  base SG: DataGolf skill ratings ({len(m)} players, incl. LIV/global)")
                return m, "dg"
        except Exception as e:
            print(f"[warn] DataGolf skill ratings failed ({e}); falling back to free SG")
    return fetch_pga_sg(), "pga"


# --------------------------------------------------------------------------
# Assemble the field
# --------------------------------------------------------------------------
def build_field(field_names: list[str]) -> list[Player]:
    """
    field_names: canonical display names (e.g. from the DK pool / odds board).
    Measured SG from the base source (DataGolf or PGA scrape); OWGR-estimated
    skill for anyone the base source doesn't rate; replacement level otherwise.
    """
    base, base_src = _base_skill_map()
    owgr = fetch_owgr()
    a, b = _calibrate_owgr(base, owgr)

    # Fuzzy indexes (exact-normalized -> last-name+initial) so name variants
    # like "Matthew"/"Matt" or "Christopher"/"Chris" still resolve.
    base_index = build_index(list(base.keys()))
    owgr_index = build_index(list(owgr.keys()))

    # Standardize driving accuracy -> z-score across covered players (handles
    # either scale: PGA percentage or DataGolf rating).
    accs = [c["drv_acc"] for c in base.values() if c.get("drv_acc") is not None]
    acc_mu = sum(accs) / len(accs) if accs else 0.0
    acc_sd = (sum((x - acc_mu) ** 2 for x in accs) / len(accs)) ** 0.5 if accs else 1.0
    acc_sd = acc_sd or 1.0

    # Clamp OWGR estimates to the observed SG range so we don't extrapolate wildly.
    covered_totals = [c["sg_ott"] + c["sg_app"] + c["sg_arg"] + c["sg_putt"]
                      for c in base.values()]
    lo = min(covered_totals) if covered_totals else -2.5
    hi = max(covered_totals) if covered_totals else 3.0
    meas_flag = ["dg"] if base_src == "dg" else []

    # Links aptitude (Open + Scottish Open finish history) — populates each
    # player's links_history so course_fit can reward proven links performers.
    try:
        from . import links
        links_scores = links.load()
    except Exception as e:
        print(f"[warn] links history unavailable: {e}")
        links_scores = {}

    players: list[Player] = []
    n_pga = n_owgr = n_none = 0
    for raw in field_names:
        base_key = match_player(raw, base_index)
        owgr_key = match_player(raw, owgr_index)
        if base_key:
            c = base[base_key]
            acc_z = ((c["drv_acc"] - acc_mu) / acc_sd) if c.get("drv_acc") is not None else 0.0
            players.append(Player(
                name=raw,
                sg_ott=round(c["sg_ott"], 3), sg_app=round(c["sg_app"], 3),
                sg_arg=round(c["sg_arg"], 3), sg_putt=round(c["sg_putt"], 3),
                sg_total=round(c["sg_ott"] + c["sg_app"] + c["sg_arg"] + c["sg_putt"], 3),
                driving_acc=round(acc_z, 2), links_history=0.5, flags=list(meas_flag),
            ))
            n_pga += 1
        elif owgr_key:
            est = max(lo, min(hi, a + b * math.log(owgr[owgr_key])))
            players.append(Player(
                name=raw,
                sg_ott=round(est * COMPONENT_SHARES["sg_ott"], 3),
                sg_app=round(est * COMPONENT_SHARES["sg_app"], 3),
                sg_arg=round(est * COMPONENT_SHARES["sg_arg"], 3),
                sg_putt=round(est * COMPONENT_SHARES["sg_putt"], 3),
                sg_total=round(est, 3),
                driving_acc=0.0, links_history=0.5, flags=["owgr-est"],
            ))
            n_owgr += 1
        else:
            # No signal at all — replacement level, flagged so it's visible.
            players.append(Player(name=raw, sg_total=round(lo, 3),
                                  sg_ott=round(lo * 0.28, 3), sg_app=round(lo * 0.38, 3),
                                  sg_arg=round(lo * 0.14, 3), sg_putt=round(lo * 0.20, 3),
                                  driving_acc=0.0, links_history=0.5, flags=["no-data"]))
            n_none += 1

    # Apply links aptitude to links_history (matched by name).
    if links_scores:
        try:
            from . import links
            n_links = links.apply_to(players, links_scores)
        except Exception as e:
            print(f"[warn] links apply failed: {e}")
            n_links = 0
    else:
        n_links = 0

    # Apply recent form (trailing 16/24-round SG).
    try:
        from . import form
        n_form = form.apply_to(players)
    except Exception as e:
        print(f"[warn] recent form unavailable: {e}")
        n_form = 0

    # Apply scoring profile (Birdie-or-Better % + Bogey Avoidance).
    n_score = 0
    try:
        scoring = fetch_scoring_stats()
        if scoring:
            sc_index = build_index(list(scoring.keys()))
            for p in players:
                k = match_player(p.name, sc_index)
                if k:
                    p.birdie_pct = scoring[k].get("birdie_pct")
                    p.bogey_pct = scoring[k].get("bogey_pct")
                    n_score += 1
    except Exception as e:
        print(f"[warn] scoring stats unavailable: {e}")

    # Standardize bogey avoidance -> z (positive = avoids bogeys) for the links
    # floor bonus. Lower bogey% is better, so we flip the sign.
    bvals = [p.bogey_pct for p in players if p.bogey_pct is not None]
    if len(bvals) >= 5:
        bmu = sum(bvals) / len(bvals)
        bsd = (sum((x - bmu) ** 2 for x in bvals) / len(bvals)) ** 0.5 or 1.0
        for p in players:
            if p.bogey_pct is not None:
                p.bogey_z = round((bmu - p.bogey_pct) / bsd, 3)

    src_label = "DataGolf-SG" if base_src == "dg" else "PGA-SG"
    print(f"  skills: {n_pga} {src_label}, {n_owgr} OWGR-est, {n_none} no-data "
          f"(OWGR fit: sg_total ≈ {a:.2f} + {b:.2f}·ln(pts)) | "
          f"{n_links} links, {n_form} form, {n_score} scoring")
    return players


def _num(v) -> float:
    try:
        return float(str(v).replace("%", "").strip())
    except (TypeError, ValueError):
        return 0.0
