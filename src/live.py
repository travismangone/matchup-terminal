"""
In-tournament (round-based / showdown) DFS projections.

For a SINGLE round, DK points are that round's hole-scoring (birdies/bogeys/
streak bonuses) — no placement, since placement is decided at the tournament's
end. So the next-round projection is driven purely by skill + variance: simulate
one round's SG for every player and map it to DK scoring points.

One round is high-variance, so the CEILING (a low round) matters as much as the
mean for GPPs — a birdie-heavy player has a higher single-round ceiling than a
steady par-maker with the same mean. We return mean / ceiling / floor.

(Round 4 is the exception: the final round's finishing position matters, so that
one will layer placement on top — handled at the dashboard level by round.)
"""

from __future__ import annotations

import numpy as np

from config import SIM, DK_SCORING, DRAW, DK_R4_PLACEMENT, DK_R4_SCORING

# Single-round upside cap. The tournament model caps a round's contribution at ~30
# (to tame outliers in a 4-round sum), but a single showdown round genuinely tops
# out higher — a bogey-free 63 with 8 birdies + an eagle is ~40+ DK points — so we
# lift the cap here to keep the GPP ceiling differentiating.
LIVE_ROUND_CAP = 50.0


# Round-to-round SG repeatability. Ball-striking (OTT+APP) is fully "real" round
# to round; around-the-green is noisier; putting barely carries over (the single
# biggest source of a fluky round). These weights turn a round's SG into its
# SUSTAINABLE part.
ARG_REPEAT = 0.60
PUTT_REPEAT = 0.35
# How much the *baseline* season skill vs. the sustainable last-round form drives
# next-round expectation.
BASELINE_WEIGHT = 0.55


def regression_scores(skills: dict[str, float], last_round: dict[str, dict]) -> dict[str, dict]:
    """
    Who over/under-performed last round, and which way they regress next round.

      sustainable = OTT + APP + 0.6·ARG + 0.35·Putt          (discount the fluky bits)
      expected    = 0.55·baseline_skill + 0.45·sustainable   (skill + real form)
      regression  = expected − last_round_total

    regression > 0  -> POSITIVE regression (scored worse than they played; bounce-back)
    regression < 0  -> NEGATIVE regression (scored better than they played; fade)
    """
    out: dict[str, dict] = {}
    for name, skill in skills.items():
        s = last_round.get(name)
        if not s:
            out[name] = {"r1_sg": None, "r1_putt": None,
                         "expected": skill, "regression": None}
            continue
        sustainable = s["ott"] + s["app"] + ARG_REPEAT * s["arg"] + PUTT_REPEAT * s["putt"]
        expected = BASELINE_WEIGHT * skill + (1 - BASELINE_WEIGHT) * sustainable
        out[name] = {
            "r1_sg": round(s["total"], 2),
            "r1_app": round(s["app"], 2),
            "r1_putt": round(s["putt"], 2),
            "expected": expected,
            "regression": round(expected - s["total"], 2),
        }
    return out


def draw_edges(waves: dict, hourly: dict) -> dict:
    """
    Per-player draw SG: average the forecast wind over each golfer's own tee ->
    finish window, then nudge them relative to the field-mean window wind (calmer
    window = positive SG). Captures the within-wave edge (earliest tees play the
    least wind), not just AM vs PM.

    waves:  {name: {wave, hour, teetime, ...}} from datagolf_field.
    hourly: {hours, wind, gust, precip} from weather.fetch_hourly.

    Returns {"per_player": {name: sg}, "summary": {...}} or {} if unusable.
    """
    from . import weather

    if not hourly or not waves:
        return {}

    # Each player's gust-blended wind over their on-course window.
    pw = {}
    for name, v in waves.items():
        hr = v.get("hour")
        if hr is None:
            continue
        w = weather.window_wind(hourly, hr, hr + DRAW["round_hours"])
        if w is not None:
            pw[name] = w
    if not pw:
        return {}

    mean_wind = sum(pw.values()) / len(pw)
    cap = DRAW["cap"]
    per_player = {
        name: round(max(-cap, min(cap, -DRAW["sg_per_mph"] * (w - mean_wind))), 3)
        for name, w in pw.items()
    }

    # Split the field into an early vs late group for the summary banner. On
    # Thu/Fri that's the real AM/PM wave; on the weekend the cut field plays one
    # continuous wave, so the wave labels are missing/uniform -> fall back to a
    # tee-time-median split ("early tees" vs "late tees"), which is what the draw
    # comes down to once wind builds through a single-wave day.
    def _avg(names):
        vals = [pw[n] for n in names if n in pw]
        return round(sum(vals) / len(vals), 1) if vals else None

    early_n = [n for n, v in waves.items() if v.get("wave") == "early"]
    late_n = [n for n, v in waves.items() if v.get("wave") == "late"]
    split = "wave"
    if not early_n or not late_n:                       # weekend / single wave
        split = "teetime"
        withhr = sorted((v["hour"], n) for n, v in waves.items()
                        if v.get("hour") is not None and n in pw)
        half = len(withhr) // 2
        early_n = [n for _, n in withhr[:half]]
        late_n = [n for _, n in withhr[half:]]

    ew, lw = _avg(early_n), _avg(late_n)
    favored = None
    if ew is not None and lw is not None:
        favored = "early" if ew < lw else "late"
    spread = round(max(per_player.values()) - min(per_player.values()), 2) if per_player else 0.0
    return {
        "per_player": per_player,
        "summary": {
            "early_wind": ew, "late_wind": lw, "favored": favored, "split": split,
            "edge_sg": spread,          # best-to-worst draw gap across the field, in SG
            "mean_wind": round(mean_wind, 1),
            "n_early": len(early_n), "n_late": len(late_n),
            "precip": hourly.get("precip", 0.0),
        },
    }


def project_next_round(skills: dict[str, float], n_sims: int = 4000) -> dict[str, dict]:
    """{player: {proj (mean), ceiling (85th pct), floor (15th pct)}} DK scoring
    for one round."""
    names = list(skills.keys())
    mu = np.array([skills[n] for n in names], dtype=np.float32)
    sigma = SIM["round_sigma"] * SIM["wind_factor"]
    rng = np.random.default_rng(SIM["seed"])

    sg = mu[None, :] + rng.normal(0.0, sigma, size=(n_sims, len(names))).astype(np.float32)
    sc = np.clip(DK_SCORING["base"] + DK_SCORING["slope"] * sg,
                 DK_SCORING["floor"], LIVE_ROUND_CAP)

    out: dict[str, dict] = {}
    for i, n in enumerate(names):
        col = sc[:, i]
        out[n] = {
            "proj": round(float(col.mean()), 1),
            "ceiling": round(float(np.percentile(col, 85)), 1),
            "floor": round(float(np.percentile(col, 15)), 1),
        }
    return out


def _r4_place_points(n: int) -> np.ndarray:
    """0-based finishing rank -> DK R4 placement points (0 beyond the table)."""
    pts = np.zeros(n, dtype=np.float32)
    for i in range(min(n, len(DK_R4_PLACEMENT))):
        pts[i] = DK_R4_PLACEMENT[i]
    return pts


def project_final_round(skills: dict[str, float], current_scores: dict[str, float],
                        n_sims: int = 5000) -> dict[str, dict]:
    """
    Round 4 projection = simulated R4 hole scoring + finishing-position points.

    For each sim we draw every player's R4 SG, map it to hole-scoring points on
    the true showdown scale, and add their simulated R4 (to par) to their 54-hole
    score to get a final 72-hole total. Ranking the field gives each player's
    finishing position -> DK R4 placement points (ties take the better position,
    per the rules). proj/ceiling/floor come from the distribution of the sum.

    skills: form + draw adjusted expected R4 SG per player.
    current_scores: each player's to-par score through 54 holes.
    """
    names = [n for n in skills if n in current_scores]
    if not names:
        return {}
    n = len(names)
    mu = np.array([skills[nm] for nm in names], dtype=np.float32)
    cur = np.array([current_scores[nm] for nm in names], dtype=np.float32)
    sigma = SIM["round_sigma"] * SIM["wind_factor"]
    rng = np.random.default_rng(SIM["seed"])
    place_pts = _r4_place_points(n)

    hole_all = np.empty((n_sims, n), dtype=np.float32)
    place_all = np.empty((n_sims, n), dtype=np.float32)
    CH = 1000                                   # chunk sims to bound the n×n rank memory
    for start in range(0, n_sims, CH):
        m = min(CH, n_sims - start)
        sg = mu[None, :] + rng.normal(0.0, sigma, size=(m, n)).astype(np.float32)
        hole = np.clip(DK_R4_SCORING["base"] + DK_R4_SCORING["slope"] * sg,
                       DK_R4_SCORING["floor"], DK_R4_SCORING["cap"])
        # Final 72-hole score to par; the field's R4 scoring average is a constant
        # that cancels in the ranking, so R4 to-par ≈ -SG is all we need.
        final = cur[None, :] - sg
        # 0-based finishing rank with ties = number of players strictly better.
        rank0 = (final[:, None, :] < final[:, :, None]).sum(axis=2)
        np.clip(rank0, 0, n - 1, out=rank0)
        hole_all[start:start + m] = hole
        place_all[start:start + m] = place_pts[rank0]

    total = hole_all + place_all
    out: dict[str, dict] = {}
    for i, nm in enumerate(names):
        col = total[:, i]
        out[nm] = {
            "proj": round(float(col.mean()), 1),
            "ceiling": round(float(np.percentile(col, 85)), 1),
            "floor": round(float(np.percentile(col, 15)), 1),
            "hole_ev": round(float(hole_all[:, i].mean()), 1),
            "place_ev": round(float(place_all[:, i].mean()), 1),
        }
    return out
