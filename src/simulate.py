"""
Monte Carlo tournament simulation.

Each player has an adjusted skill = expected strokes gained per round at this
course (from course_fit). We simulate the full 72-hole tournament many times:

    round_sg[player] = skill[player] + Normal(0, sigma)     # higher = better
    36-hole total    -> apply the cut (top 70 and ties advance)
    72-hole total    -> rank; ties broken by tiny jitter (proxy for playoff)

Counting how often each player lands in each bucket gives calibrated win /
top-5 / top-10 / top-20 / make-cut probabilities — the model's fair prices.

Vectorized with numpy: scores are an (n_sims x n_players) array, so 20k sims
over a 156-man field is a fraction of a second.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import SIM, EVENT, DK_PLACEMENT_POINTS, DK_SCORING


@dataclass
class SimResult:
    names: list[str]
    win: np.ndarray        # P(win)
    top_5: np.ndarray
    top_10: np.ndarray
    top_20: np.ndarray
    make_cut: np.ndarray
    dk_points: np.ndarray        # expected total DK points (placement + scoring)
    dk_placement: np.ndarray     # placement component
    dk_scoring: np.ndarray       # hole-scoring component
    n_sims: int

    def as_dicts(self) -> list[dict]:
        rows = []
        for i, name in enumerate(self.names):
            rows.append({
                "name": name,
                "win": float(self.win[i]),
                "top_5": float(self.top_5[i]),
                "top_10": float(self.top_10[i]),
                "top_20": float(self.top_20[i]),
                "make_cut": float(self.make_cut[i]),
                "dk_points": float(self.dk_points[i]),
                "dk_placement": float(self.dk_placement[i]),
                "dk_scoring": float(self.dk_scoring[i]),
            })
        rows.sort(key=lambda r: r["win"], reverse=True)
        return rows


def simulate(skills: dict[str, float]) -> SimResult:
    names = list(skills.keys())
    mu = np.array([skills[n] for n in names], dtype=float)
    n_players = len(names)

    n_sims = SIM["n_sims"]
    sigma = SIM["round_sigma"] * SIM["wind_factor"]
    rng = np.random.default_rng(SIM["seed"])

    # --- Simulate 4 rounds of strokes-gained (higher = better) -------------
    # shape: (n_sims, n_players, rounds)
    rounds = EVENT["rounds"]
    noise = rng.normal(0.0, sigma, size=(n_sims, n_players, rounds))
    sg = mu[None, :, None] + noise

    first36 = sg[:, :, :2].sum(axis=2)     # (n_sims, n_players)
    total72 = sg.sum(axis=2)               # (n_sims, n_players)

    # --- Cut after R2: top 70 and ties advance (highest SG = best) ---------
    top_n = EVENT["cut_rule"]["top_n"]
    # For each sim, the cut line is the 70th-best 36-hole score. Players at or
    # above the line (ties included) make the cut.
    k = min(top_n, n_players) - 1
    # partition to find the (top_n)-th largest value per sim without full sort.
    cut_line = np.partition(first36, n_players - 1 - k, axis=1)[:, n_players - 1 - k]
    made_cut = first36 >= cut_line[:, None]     # (n_sims, n_players) bool

    # Missed-cut players cannot win or place; sink their 72-hole score.
    total_eff = np.where(made_cut, total72, -np.inf)

    # --- Rank the 72-hole totals (add tiny jitter to break exact ties) -----
    total_eff = total_eff + rng.normal(0, 1e-6, size=total_eff.shape)
    # rank 0 = winner (largest total). argsort descending -> position of each.
    order = np.argsort(-total_eff, axis=1)              # player indices by finish
    finish = np.empty_like(order)
    ar = np.arange(n_players)
    # finish[sim, player_index] = finishing position (0-based)
    np.put_along_axis(finish, order, np.broadcast_to(ar, order.shape), axis=1)

    win = (finish == 0).mean(axis=0)
    top_5 = (finish < 5).mean(axis=0)
    top_10 = (finish < 10).mean(axis=0)
    top_20 = (finish < 20).mean(axis=0)
    make_cut = made_cut.mean(axis=0)

    # --- DK placement points: map each sim's finishing rank -> DK points -----
    rank_pts = np.zeros(n_players)
    tbl = DK_PLACEMENT_POINTS
    fill = min(len(tbl), n_players)
    rank_pts[:fill] = tbl[:fill]
    place_by_sim = rank_pts[finish]                       # (n_sims, n_players)
    dk_placement = place_by_sim.mean(axis=0)

    # --- DK hole-scoring points: per-round SG -> scoring pts, only rounds -----
    # actually played (2 pre-cut always, 2 more if the sim made the cut).
    sc = np.clip(DK_SCORING["base"] + DK_SCORING["slope"] * sg,
                 DK_SCORING["floor"], DK_SCORING["cap"])   # (n_sims, n_players, rounds)
    pre_cut = sc[:, :, :2].sum(axis=2)                     # rounds 1-2
    weekend = sc[:, :, 2:].sum(axis=2) * made_cut          # rounds 3-4 if advanced
    scoring_by_sim = pre_cut + weekend
    dk_scoring = scoring_by_sim.mean(axis=0)

    dk_points = dk_placement + dk_scoring

    return SimResult(
        names=names, win=win, top_5=top_5, top_10=top_10,
        top_20=top_20, make_cut=make_cut, dk_points=dk_points,
        dk_placement=dk_placement, dk_scoring=dk_scoring, n_sims=n_sims,
    )
