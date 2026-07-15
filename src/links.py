"""
Links aptitude — a free, event-specific skill signal for The Open.

Why finish-based, not strokes-gained: SG requires ShotLink, which only runs at
U.S. PGA Tour events. The Open (UK) and Scottish Open (Scotland) have NO
strokes-gained data at all — so the best free links signal is historical FINISH
performance at links events, which is exactly what the model was missing (it had
no way to see that Fleetwood / McIlroy / Fitzpatrick travel to links golf).

We score each player's record at The Open + Genesis Scottish Open over recent
years into a links_history value in [0, 1] (0.5 = neutral / no evidence),
recency-weighted and shrunk toward 0.5 for small samples so one hot week doesn't
dominate. That value feeds course_fit's links-history bonus.

Bonus side effect: LIV players (Rahm, DeChambeau, ...) still play The Open, so
they DO get a links signal here even though OWGR underrates them elsewhere.

Cached to data/links_history.json — it's historical, so it rarely changes.
"""

from __future__ import annotations

import json
import os

from . import backtest
from .match import normalize_name, build_index, match_player

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE = os.path.join(DATA, "links_history.json")

# Co-sanctioned links events in the PGA Tour ("R") data, by year:
#   The Open Championship = R{year}100 ,  Genesis Scottish Open = R{year}541
LINKS_EVENTS = {"open": "R{y}100", "scottish": "R{y}541"}
YEARS = [2025, 2024, 2023, 2022, 2021]

DECAY = 0.80        # recency weight per year back
MC_PERF = 0.25      # score for a missed cut (mild penalty, not disqualifying)
PRIOR_K = 1.5       # shrinkage toward 0.5 (in units of appearance-weight)


def _appearance_perf(finish: int | None, field_size: int) -> float:
    """Finish -> performance in [0,1]. 1st ~1.0, mid-pack ~0.5, missed cut = MC_PERF."""
    if finish is None:
        return MC_PERF
    n = max(field_size, 2)
    return max(0.0, 1.0 - (finish - 1) / (n - 1))


def compute(years: list[int] = YEARS) -> dict[str, float]:
    """{normalized_name: links_history in [0,1]} across the links events."""
    # name -> list of (weight, perf)
    recs: dict[str, list[tuple[float, float]]] = {}
    for y in years:
        w = DECAY ** (2026 - y)
        for eid in (v.format(y=y) for v in LINKS_EVENTS.values()):
            try:
                results = backtest.fetch_results(eid)
            except Exception as e:
                print(f"[warn] links {eid}: {e}")
                continue
            if len(results) < 30:
                continue
            fs = len(results)
            for name, pos in results:
                perf = _appearance_perf(pos, fs)
                recs.setdefault(normalize_name(name), []).append((w, perf))

    out: dict[str, float] = {}
    for name, rs in recs.items():
        sw = sum(w for w, _ in rs)
        wavg = sum(w * p for w, p in rs) / sw if sw else 0.5
        # Shrink toward neutral 0.5 by sample weight.
        out[name] = round((sw * wavg + PRIOR_K * 0.5) / (sw + PRIOR_K), 4)
    return out


def load(refresh: bool = False) -> dict[str, float]:
    """Cached links_history map. Computes + caches on first use."""
    if not refresh and os.path.exists(CACHE):
        try:
            with open(CACHE) as f:
                return json.load(f)
        except Exception:
            pass
    scores = compute()
    os.makedirs(DATA, exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(scores, f)
    print(f"  links history: computed {len(scores)} players from The Open + Scottish Open")
    return scores


def apply_to(players, scores: dict[str, float] | None = None) -> int:
    """Set each player's links_history from the score map. Returns # matched."""
    scores = scores if scores is not None else load()
    index = build_index(list(scores.keys()))
    matched = 0
    for p in players:
        key = match_player(p.name, index)
        if key:
            p.links_history = scores[key]
            matched += 1
        # else: leave the neutral 0.5 default
    return matched
