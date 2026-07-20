"""
Dashboard state builder — turns the latest snapshot into one JSON-serializable
dict for the web UI. Reads the store (no API pull, so it's fast and free); the
"Pull live" action in the web app is what spends credits.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from . import store, free_sg, course_fit, simulate, datagolf
from .course_fit import adjusted_skill, skill_breakdown
from .compare import find_plays, sharp_reference, scan_ev, scan_matchups
from .clv import line_movement
from .odds_math import decimal_to_american, prob_to_decimal, expected_value
from config import SHARP_BOOKS, EVENT, SIM

_MATCHUPS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "data", "matchups.json")


def _save_matchups(mus: list) -> None:
    os.makedirs(os.path.dirname(_MATCHUPS_PATH), exist_ok=True)
    with open(_MATCHUPS_PATH, "w") as f:
        json.dump(mus, f)


def _load_matchups() -> list:
    try:
        with open(_MATCHUPS_PATH) as f:
            return json.load(f)
    except Exception:
        return []

MARKETS = ["win", "top_5", "top_10", "top_20", "make_cut"]


def pull_and_snapshot() -> str:
    """Live pull -> store. Sportsbooks (incl. FanDuel + Pinnacle) come from
    DataGolf, which is flat-rate (no per-pull credits) and lists an event's odds
    1-2 weeks out. Prediction markets from Kalshi/Polymarket. Returns run ts."""
    from .odds import datagolf_odds, polymarket, kalshi, datagolf_matchups
    from .match import build_index
    from . import dk
    from config import POLYMARKET_TITLE_CONTAINS, KALSHI_EVENTS

    # Roster = DK pool + DataGolf field, so odds names line up with the model.
    roster = dk.pool_names()
    try:
        roster += datagolf.fetch_field()
    except Exception:
        pass
    idx = build_index(roster)

    sb = datagolf_odds.fetch_winner_quotes(idx)      # FanDuel + Pinnacle + 12 more
    pm = polymarket.fetch_winner_quotes(idx, POLYMARKET_TITLE_CONTAINS)
    kq = kalshi.fetch_quotes(idx, KALSHI_EVENTS)
    # Head-to-head matchups are pairwise -> own file, not the per-player line store.
    try:
        _save_matchups(datagolf_matchups.fetch_matchups(idx))
    except Exception as e:
        print(f"[warn] matchups save failed: {e}")
    return store.snapshot(sb + pm + kq)


def credit_status() -> dict:
    # Odds now come from DataGolf (flat-rate subscription) — no per-pull metering.
    return {"metered": False, "note": "DataGolf odds — flat-rate, no per-pull credits"}


def _skill_source(flags: list[str]) -> str:
    if "wd" in flags:            # missed cut / withdrew / DQ'd -> out, shown as WD
        return "WD"
    if "no-data" in flags:
        return "none"
    if "owgr-est" in flags:
        return "OWGR"
    if "dg" in flags:
        return "DG"
    return "PGA"


_ESTIMATED = {"owgr-est", "no-data"}


def build_state(demo: bool = False) -> dict:
    run = store.closing_run(demo)
    if not run:
        return {"empty": True, "event": EVENT["name"], "course": EVENT["course"]}

    # All-market quotes from the latest run.
    quotes = []
    for m in MARKETS:
        quotes += store.quotes_at(run, m, demo)

    win_q = store.quotes_at(run, "win", demo)
    odds_players = sorted({q.player for q in win_q if q.source in SHARP_BOOKS}) or \
        sorted({q.player for q in win_q})

    # Field = the full DK pool (so every rosterable player has data) + any
    # odds-board players not in the pool, deduped by fuzzy name.
    from .match import match_player
    from . import dk
    dk_sal = {} if demo else dk.load()
    dk_idx = dk.index(dk_sal) if dk_sal else None
    own_map = {} if demo else dk.load_ownership()      # projected GPP ownership
    own_idx = dk.index(own_map) if own_map else None
    if demo:
        players = datagolf.synthetic_field()
    else:
        field = [v["name"] for v in dk_sal.values()]
        for onm in odds_players:
            if not (dk_idx and match_player(onm, dk_idx)):
                field.append(onm)
        players = free_sg.build_field(field or odds_players)

    # Flag anyone out of the tournament (missed cut / WD / DQ). We keep them in the
    # lists — tagged WD with zeroed projections — rather than dropping them, so you
    # can see who's out at a glance. Fetch live state once here (in-play + the next
    # round's tee sheet) and hand both to _live_round to avoid re-fetching.
    from .odds import datagolf_inplay, datagolf_field
    from .match import build_index
    ip = {"next_round": None, "players": {}, "cut": set()}
    wv = {"date": None, "players": {}}
    full_idx = None
    wd: set = set()
    if not demo and players:
        full_idx = build_index([p.name for p in players])
        ip = datagolf_inplay.fetch_inplay(full_idx)
        nr = ip.get("next_round")
        if nr:
            wv = datagolf_field.fetch_waves(full_idx, nr)
        wd = _out_of_event(players, dk_sal, full_idx, ip, wv)
        for p in players:
            if p.name in wd and "wd" not in p.flags:
                p.flags.append("wd")
        if wd:
            print(f"[live] {len(wd)} players out (cut/WD/DQ) -> tagged WD, 0 proj")

    # Join DK salary onto each player.
    sal_by = {}
    for p in players:
        s = dk.lookup(p.name, dk_sal, dk_idx) if dk_idx else None
        sal_by[p.name] = s
    pl_by = {p.name: p for p in players}

    flags_by = {p.name: p.flags for p in players}
    # Only live players drive the sim/optimal/H2H — cut players occupying finishing
    # positions would distort everyone else's win/top-N odds (badly so post-cut).
    live_players = [p for p in players if p.name not in wd]
    skills_map = course_fit.build_skills(live_players)
    sim = simulate.simulate(skills_map)
    rows = sim.as_dicts()
    row_by = {r["name"]: r for r in rows}
    model_probs = {m: {r["name"]: r[m] for r in rows} for m in MARKETS}

    # Strokes-gained inputs (what drives the sim), sorted by course-fit skill.
    def _srow(p):
        bd = skill_breakdown(p)
        return {
            "name": p.name,
            "adjusted": round(bd["adjusted"], 3),   # course-fit output -> the sim
            "sg_total": round(p.sg_total, 2),
            "sg_ott": round(p.sg_ott, 2), "sg_app": round(p.sg_app, 2),
            "sg_arg": round(p.sg_arg, 2), "sg_putt": round(p.sg_putt, 2),
            "driving_acc": round(p.driving_acc, 2),
            "links": round(p.links_history, 2),
            "form_16": p.form_16, "form_24": p.form_24,
            "birdie_pct": p.birdie_pct, "bogey_pct": p.bogey_pct,
            "src": _skill_source(p.flags),
            # Per-signal contributions to adjusted skill (they sum to it).
            "bd": {k: round(v, 3) for k, v in bd.items() if k != "adjusted"},
        }
    skills = sorted((_srow(p) for p in players),
                    key=lambda r: r["adjusted"], reverse=True)

    # Projections (+ salary). Iterate the full field so out players (WD/cut) still
    # show — zeroed, tagged WD, and sinking to the bottom.
    projections = []
    for p in players:
        r = row_by.get(p.name)
        is_wd = p.name in wd or r is None
        s = sal_by.get(p.name)
        projections.append({
            "name": p.name,
            "win": 0 if is_wd else r["win"],
            "top_5": 0 if is_wd else r["top_5"],
            "top_10": 0 if is_wd else r["top_10"],
            "top_20": 0 if is_wd else r["top_20"],
            "make_cut": 0 if is_wd else r["make_cut"],
            "salary": s["salary"] if s else None,
            "src": _skill_source(flags_by.get(p.name, [])),
        })

    # DFS value view — projected DK placement points and value per $1k.
    dfs = []
    for p in players:
        r = row_by.get(p.name)
        is_wd = p.name in wd or r is None
        s = sal_by.get(p.name)
        salary = s["salary"] if s else None
        src = _skill_source(flags_by.get(p.name, []))
        pts = 0.0 if is_wd else r["dk_points"]     # out -> no points left to score
        # DK's own PPG is a garbage placeholder for un-rated club pros / amateurs
        # (e.g. 86 for a mini-tour player) — suppress it for no-data players so it
        # doesn't read as a real projection, matching their blanked BoB%/Bogey%.
        dkppg = None if src in ("none", "WD") else (s["dkppg"] if s else None)
        o = dk.lookup(p.name, own_map, own_idx) if own_idx else None
        dfs.append({
            "name": p.name,
            "salary": salary,
            "dk_points": round(pts, 1),
            "dk_placement": 0.0 if is_wd else round(r["dk_placement"], 1),
            "dk_scoring": 0.0 if is_wd else round(r["dk_scoring"], 1),
            "value": 0.0 if is_wd else (round(pts / (salary / 1000.0), 2) if salary else None),
            "own_large": o["own_large"] if o else None,   # GPP (large-field) ownership %
            "own_small": o["own_small"] if o else None,
            "dkppg": dkppg,
            "birdie_pct": getattr(pl_by.get(p.name), "birdie_pct", None),
            "bogey_pct": getattr(pl_by.get(p.name), "bogey_pct", None),
            "win": 0 if is_wd else r["win"],
            "top_20": 0 if is_wd else r["top_20"],
            "make_cut": 0 if is_wd else r["make_cut"],
            "src": src,
        })
    dfs = [d for d in dfs if d["salary"]]      # DFS view = rosterable players only
    dfs.sort(key=lambda d: (d["value"] or 0), reverse=True)

    # Sim / optimal-lineup exposure: % of sims each player is in the optimal DK
    # lineup, vs. projected ownership -> leverage.
    sim_rows = []
    if not demo:
        salaries = {p.name: sal_by[p.name]["salary"]
                    for p in players if sal_by.get(p.name) and p.name not in wd}
        from . import optimal
        exposure = optimal.compute(skills_map, salaries)
        dfs_by = {d["name"]: d for d in dfs}
        for name, opt in exposure.items():
            d = dfs_by.get(name, {})
            own = d.get("own_large")
            sim_rows.append({
                "name": name, "salary": d.get("salary"),
                "dk_points": d.get("dk_points"), "value": d.get("value"),
                "optimal": round(opt * 100, 1),
                "own_large": own,
                "leverage": round(opt * 100 - own, 1) if own is not None else None,
                "src": d.get("src", "PGA"),
            })
        sim_rows.sort(key=lambda r: r["optimal"], reverse=True)

    # Best plays, grouped by market.
    plays = find_plays(quotes, model_probs, skill_flags=flags_by)
    plays_by_market: dict[str, list] = {m: [] for m in MARKETS}
    for p in plays:
        d = asdict(p)
        d["american"] = decimal_to_american(p.offered_decimal)
        d["estimated"] = bool({"owgr-est", "no-data"} & set(p.flags))
        plays_by_market.setdefault(p.market, []).append(d)

    # EV scanner — every +EV bet vs sharp / model / blend (own tab).
    ev_scan = scan_ev(quotes, model_probs)
    for r in ev_scan:
        r["estimated"] = bool(_ESTIMATED & set(flags_by.get(r["player"], [])))
    ev_scan = ev_scan[:250]

    # Head-to-head matchup edges (P(A beats B) from the skill gap vs the books).
    sigma = SIM["round_sigma"] * SIM["wind_factor"]
    h2h = scan_matchups(_load_matchups(), skills_map, sigma)
    for r in h2h:
        r["estimated"] = bool(_ESTIMATED & (set(flags_by.get(r["player"], [])) |
                                            set(flags_by.get(r["opponent"], []))))
    h2h = h2h[:200]

    return {
        "empty": False,
        "event": EVENT["name"],
        "course": EVENT["course"],
        "mode": "DEMO" if demo else "LIVE",
        "run": run,
        "coverage": _coverage(players),
        "dk_coverage": _dk_coverage(players, sal_by),
        "skills": skills,
        "projections": projections,
        "dfs": dfs,
        "sim": sim_rows,
        "live": _live_round(live_players, skills_map, sal_by, own_map, own_idx, demo, ip, full_idx, wv),
        "plays": plays_by_market,
        "ev_scan": ev_scan,
        "h2h": h2h,
        "board": _board(win_q),
        "movement": _movement(demo),
        "sharp_label": " + ".join(b.replace("_", " ").title() for b in SHARP_BOOKS),
        "credits": credit_status(),
    }


def _out_of_event(players, dk_sal, idx, ip, wv) -> set:
    """Names to drop from the whole model — missed cut / withdrew / DQ'd. Combines,
    most authoritative first:
      1. DK CSV Status column (WD/CUT/DQ) — set the moment DK marks it.
      2. DataGolf in-play position flipped to CUT/WD/DQ.
      3. No next-round tee time, once the tee sheet is actually posted (the
         definitive cut signal — the field-updates sheet only lists who's playing).
      4. Backstop: a parseable 0% make-cut once a weekend round is next.
    Nothing fires until a real signal exists, so pre-cut the full field is kept."""
    from .match import match_player

    out = set(ip.get("cut", set()))                       # (2) position-based

    dk_out = {v["name"] for v in dk_sal.values()          # (1) DK Status
              if v.get("status") in {"WD", "CUT", "DQ", "DNS"}}
    out |= {match_player(n, idx) or n for n in dk_out}

    nr = ip.get("next_round")
    teed = wv.get("players", {}) if wv else {}
    # (3) Only trust "no tee time" once a real weekend sheet is posted (guards the
    # transitional window where the cut is done but pairings aren't out yet).
    if nr and nr >= 3 and len(teed) >= 50:
        out |= {p.name for p in players if p.name not in teed}

    if nr and nr >= 3:                                    # (4) make-cut backstop
        for name, st in ip.get("players", {}).items():
            mc = _as_float(st.get("make_cut"))
            if mc is not None and mc <= 0:
                out.add(name)
    return out


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _live_round(players, skills_map, sal_by, own_map, own_idx, demo, ip=None, idx=None, wv=None) -> dict:
    """In-tournament next-round DFS: current standing + projected next-round DK
    points (single-round scoring) + value/ceiling. Empty until a round is live.
    `ip`/`idx` are the in-play result + name index already fetched by build_state."""
    if demo:
        return {"next_round": None, "rows": []}
    from .odds import datagolf_inplay, datagolf_livestats, datagolf_field
    from .match import build_index
    from . import live, dk, weather

    if idx is None:
        idx = build_index([p.name for p in players])
    if ip is None:
        ip = datagolf_inplay.fetch_inplay(idx)
    nr = ip.get("next_round")
    if not nr or not ip.get("players"):
        return {"next_round": None, "rows": []}

    # Regression: blend this tournament's completed rounds (recency-weighted) into
    # a form signal, split its sustainable part from the fluky part, and combine
    # with season skill -> the form-adjusted expectation that drives the projection.
    recent_sg = datagolf_livestats.fetch_blended_stats(idx, nr - 1) if nr > 1 else {}
    reg = live.regression_scores(skills_map, recent_sg)

    # Draw: pair each player's own next-round tee window with the wind forecast;
    # the calmer window gets a positive SG nudge (captures earliest-tee edge too).
    if wv is None:
        wv = datagolf_field.fetch_waves(idx, nr)
    waves = wv.get("players", {})
    hourly = weather.fetch_hourly(wv["date"]) if wv.get("date") and waves else None
    draw = live.draw_edges(waves, hourly) if hourly else {}
    per_player = draw.get("per_player", {})

    # Fold the draw nudge into the projection mean (on top of the regression expectation).
    proj_skills = {name: reg[name]["expected"] + per_player.get(name, 0.0)
                   for name in skills_map}
    # R4: finishing position is scored, so sim the tournament to the finish from
    # each player's 54-hole score. Earlier rounds are single-round scoring only.
    cur_scores = {}
    if nr >= 4:
        for name in proj_skills:
            st = ip["players"].get(name)
            cs = _parse_score(st.get("current_score")) if st else None
            if cs is not None:
                cur_scores[name] = cs
        proj = live.project_final_round(proj_skills, cur_scores)
    else:
        proj = live.project_next_round(proj_skills)

    # Optimal-lineup exposure for THIS slate. The Optimal tab runs the 4-round
    # sim, which can omit the best single-round plays entirely (it left out the
    # eventual Open winner on R4) — so the showdown gets its own knapsack.
    salaries = {p.name: sal_by[p.name]["salary"] for p in players
                if sal_by.get(p.name) and p.name in proj_skills and "wd" not in p.flags}
    try:
        expo = live.showdown_exposure(proj_skills, salaries, cur_scores or None) if salaries else {}
    except Exception as e:
        print(f"[warn] showdown exposure failed: {e}")
        expo = {}

    rows = []
    for p in players:
        if "wd" in p.flags:             # cut/withdrawn -> no next round to project
            continue
        st = ip["players"].get(p.name)
        if not st:                      # only golfers actually in the field
            continue
        pr = proj.get(p.name)
        rg = reg.get(p.name, {})
        w = waves.get(p.name) or {}
        s = sal_by.get(p.name)
        salary = s["salary"] if s else None
        o = dk.lookup(p.name, own_map, own_idx) if own_idx else None
        rows.append({
            "name": p.name,
            "pos": st["current_pos"], "score": st["current_score"], "thru": st["thru"],
            "salary": salary,
            "proj": pr["proj"] if pr else None,
            "ceiling": pr["ceiling"] if pr else None,
            "value": round(pr["proj"] / (salary / 1000.0), 2) if (pr and salary) else None,
            "r1_sg": rg.get("r1_sg"), "r1_app": rg.get("r1_app"),
            "r1_putt": rg.get("r1_putt"), "regression": rg.get("regression"),
            "wave": w.get("wave"), "teetime": w.get("teetime"),
            "draw": round(per_player.get(p.name, 0.0), 2) if per_player else None,
            "place": pr.get("place_ev") if pr else None,   # R4 finishing-position pts (None pre-R4)
            "hole": pr.get("hole_ev") if pr else None,     # R4 hole-scoring pts
            "optimal": round(expo[p.name] * 100, 1) if p.name in expo else None,
            "leverage": (round(expo[p.name] * 100 - o["own_large"], 1)
                         if (p.name in expo and o and o.get("own_large") is not None) else None),
            "own_large": o["own_large"] if o else None,
            "src": _skill_source(p.flags),
        })
    rows.sort(key=lambda r: (r["value"] or 0), reverse=True)
    return {"next_round": nr, "rows": rows, "draw": draw.get("summary")}


def _parse_score(v):
    """To-par score -> float. Handles 'E', '+2', '-10', ints, and None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().upper()
    if s in ("E", "EVEN", ""):
        return 0.0
    try:
        return float(s.replace("+", ""))
    except ValueError:
        return None


def _coverage(players) -> dict:
    measured = sum(1 for p in players if not (_ESTIMATED & set(p.flags)))
    owgr = sum(1 for p in players if "owgr-est" in p.flags)
    none = sum(1 for p in players if "no-data" in p.flags)
    return {"total": len(players), "pga": measured, "owgr": owgr, "none": none}


def _dk_coverage(players, sal_by) -> dict:
    """Every DK-pool player's data status. `none`/`owgr` lists surface who's
    running on estimated (not measured) skill so nothing is silently missing."""
    dk_players = [p for p in players if sal_by.get(p.name)]
    measured = [p.name for p in dk_players if not (_ESTIMATED & set(p.flags))]
    owgr = [p.name for p in dk_players if "owgr-est" in p.flags]
    none = [p.name for p in dk_players if "no-data" in p.flags]
    return {
        "pool": len(dk_players),
        "pga": len(measured), "owgr": len(owgr), "none": len(none),
        "owgr_players": owgr, "none_players": none,
    }


def _board(win_q) -> list:
    fair = sharp_reference(win_q)
    best = {}
    for x in win_q:
        if x.source in SHARP_BOOKS:
            continue
        if x.player not in best or x.decimal_odds > best[x.player].decimal_odds:
            best[x.player] = x
    out = []
    for p in sorted(fair, key=lambda k: fair[k], reverse=True):
        b = best.get(p)
        row = {
            "name": p,
            "fair": fair[p],
            "fair_american": decimal_to_american(prob_to_decimal(fair[p])),
            "best_book": b.source if b else None,
            "best_american": decimal_to_american(b.decimal_odds) if b else None,
            "edge": expected_value(fair[p], b.decimal_odds) if b else None,
        }
        out.append(row)
    return out


def _movement(demo) -> dict:
    rows = line_movement("win", demo)
    o, c = store.opening_run(demo), store.closing_run(demo)
    return {
        "open_run": o, "close_run": c, "single": (o == c),
        # EVERY golfer currently priced by the sharp book, sorted by biggest mover.
        # No 40-row cap, and no requirement that they had an opening line — a golfer
        # with only a current line shows "—" for open/Δ rather than being dropped.
        # The table is sortable, so you can find any player or rank by steam/drift.
        "rows": [{
            "name": m.player,
            "open_american": m.open_american, "close_american": m.close_american,
            "open_prob": m.open_prob, "close_prob": m.close_prob,
            "delta": m.delta_prob,
        } for m in rows if m.close_prob is not None],
    }
