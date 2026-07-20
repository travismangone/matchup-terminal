"""
Round-by-round model validation — did the live model actually predict anything?

Rather than logging predictions live (Render's disk is ephemeral, so a log would
vanish on redeploy), this RECONSTRUCTS what the model would have projected before
each round from the event's own round-by-round SG, then grades it against what
actually happened. One command, any completed event, nothing to persist.

Grades the signals the live model leans on:
  * regression  — the flagship: does over/under-performing your ball-striking
                  predict the next round?
  * skill       — did the season-long links-adjusted rating rank the field?
  * repeatability — how much each SG component actually carries round to round,
                  which is what the ARG/PUTT discount weights encode.

    python -m src.validate_rounds            # current event on DataGolf
"""

from __future__ import annotations

import statistics as st

from . import free_sg, course_fit, live
from .match import build_index, match_player


def _corr(a, b):
    if len(a) < 3:
        return 0.0
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else 0.0


def _spearman(a, b):
    ra = {v: i for i, v in enumerate(sorted(range(len(a)), key=lambda i: a[i]))}
    rb = {v: i for i, v in enumerate(sorted(range(len(b)), key=lambda i: b[i]))}
    return _corr([ra[i] for i in range(len(a))], [rb[i] for i in range(len(b))])


def _blend(rounds: dict, names, through: int, recency: float = 1.5) -> dict:
    """Same recency-weighted blend the live model feeds into regression_scores."""
    w = {r: recency ** (r - 1) for r in range(1, through + 1)}
    out = {}
    for n in names:
        acc = {"ott": 0.0, "app": 0.0, "arg": 0.0, "putt": 0.0, "total": 0.0}
        ws = 0.0
        for r in range(1, through + 1):
            s = rounds.get(r, {}).get(n)
            if not s:
                continue
            for k in acc:
                acc[k] += w[r] * s[k]
            ws += w[r]
        if ws:
            out[n] = {k: v / ws for k, v in acc.items()}
    return out


def grade_event(rounds: dict[int, dict], skills: dict[str, float]) -> dict:
    """
    rounds: {round_no: {player: {ott, app, arg, putt, total}}} actual per-round SG.
    skills: the model's pre-event links-adjusted skill per player.
    Returns a dict of metrics (also pretty-printed by report()).
    """
    played = sorted(r for r in rounds if rounds[r])
    out: dict = {"rounds": played, "regression": [], "repeatability": {}}

    # 1. Skill vs actual per-round SG over the event.
    ev = {n: st.mean([rounds[r][n]["total"] for r in played if n in rounds[r]])
          for n in skills if any(n in rounds[r] for r in played)}
    ns = sorted(ev)
    if ns:
        out["skill"] = {
            "n": len(ns),
            "pearson": round(_corr([skills[n] for n in ns], [ev[n] for n in ns]), 3),
            "spearman": round(_spearman([skills[n] for n in ns], [ev[n] for n in ns]), 3),
        }

    # 2. Regression signal: predict round R from rounds 1..R-1.
    for R in played:
        if R < 2:
            continue
        blend = _blend(rounds, skills.keys(), R - 1)
        reg = live.regression_scores(skills, blend)
        valid = [n for n in reg if reg[n].get("regression") is not None and n in rounds[R]]
        if len(valid) < 5:
            continue
        rv = [reg[n]["regression"] for n in valid]
        av = [rounds[R][n]["total"] for n in valid]
        pos = [rounds[R][n]["total"] for n in valid if reg[n]["regression"] > 0]
        neg = [rounds[R][n]["total"] for n in valid if reg[n]["regression"] < 0]
        out["regression"].append({
            "round": R, "n": len(valid), "corr": round(_corr(rv, av), 3),
            "pos_avg": round(st.mean(pos), 2) if pos else None,
            "neg_avg": round(st.mean(neg), 2) if neg else None,
            "edge": round(st.mean(pos) - st.mean(neg), 2) if pos and neg else None,
        })

    # 3. Component repeatability (pooled consecutive rounds) — the empirical check
    #    on the ARG x0.60 / PUTT x0.35 discounts in live.regression_scores.
    for comp in ("ott", "app", "arg", "putt", "total"):
        xs, ys = [], []
        for R in played:
            if R - 1 not in rounds:
                continue
            for n in rounds[R]:
                if n in rounds[R - 1]:
                    xs.append(rounds[R - 1][n][comp])
                    ys.append(rounds[R][n][comp])
        out["repeatability"][comp] = round(_corr(xs, ys), 3)
    return out


def report(res: dict) -> str:
    L = ["=" * 66, "ROUND-MODEL VALIDATION", "=" * 66]
    s = res.get("skill")
    if s:
        L.append(f"\nSKILL vs actual SG/round: pearson {s['pearson']:+.3f}  "
                 f"spearman {s['spearman']:+.3f}  (n={s['n']})")
    if res["regression"]:
        L.append("\nREGRESSION — predicting each round from the ones before it")
        L.append(f"  {'round':>6} {'n':>5} {'corr':>8} {'pos avg':>9} {'neg avg':>9} {'edge':>7}")
        for r in res["regression"]:
            L.append(f"  {'R'+str(r['round']):>6} {r['n']:>5} {r['corr']:>+8.3f} "
                     f"{r['pos_avg']:>+9.2f} {r['neg_avg']:>+9.2f} {r['edge']:>+7.2f}")
        wins = sum(1 for r in res["regression"] if (r["edge"] or 0) > 0)
        edges = [r["edge"] for r in res["regression"] if r["edge"] is not None]
        L.append(f"  -> positive regression outscored negative in {wins}/{len(res['regression'])}"
                 f" rounds (avg edge {st.mean(edges):+.2f} SG)" if edges else "")
    if res["repeatability"]:
        L.append("\nCOMPONENT REPEATABILITY (round-to-round corr)")
        for k, v in res["repeatability"].items():
            L.append(f"  {k:>5}: {v:+.3f}")
        L.append("  model assumes OTT/APP repeat fully, ARG x0.60, PUTT x0.35 —")
        L.append("  compare these before retuning those weights.")
    return "\n".join(L)


def run(tour: str = "pga") -> dict:
    """Pull the current event's rounds from DataGolf and grade the model on it."""
    from .odds import datagolf_livestats, datagolf_inplay

    ip_names = []
    from .odds.datagolf_inplay import fetch_inplay          # noqa: F401
    import os, requests
    key = os.getenv("DATAGOLF_KEY")
    d = requests.get("https://feeds.datagolf.com/preds/in-play",
                     params={"tour": tour, "dead_heat": "no", "odds_format": "percent",
                             "file_format": "json", "key": key}, timeout=30).json()
    for x in d.get("data", []):
        n = x.get("player_name", "")
        ip_names.append(f"{n.split(', ')[1]} {n.split(', ')[0]}" if ", " in n else n)

    players = free_sg.build_field(ip_names)
    skills = course_fit.build_skills(players)
    idx = build_index([p.name for p in players])

    rounds = {}
    for r in (1, 2, 3, 4):
        raw = datagolf_livestats.fetch_round_stats(idx, r, tour)
        if raw:
            rounds[r] = raw
    return grade_event(rounds, skills)


if __name__ == "__main__":
    print(report(run()))
