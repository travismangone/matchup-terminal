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
    """
    Monte Carlo, run in memory-bounded CHUNKS (float32) so peak RAM stays low
    regardless of n_sims — the full-array version OOM-kills a 512MB instance.
    Each chunk is independent sims; we accumulate per-sim counts/points and
    divide at the end, so the result matches the all-at-once version within
    Monte Carlo noise.
    """
    names = list(skills.keys())
    mu = np.array([skills[n] for n in names], dtype=np.float32)
    n_players = len(names)

    n_sims = SIM["n_sims"]
    sigma = SIM["round_sigma"] * SIM["wind_factor"]
    rounds = EVENT["rounds"]
    top_n = EVENT["cut_rule"]["top_n"]
    rng = np.random.default_rng(SIM["seed"])

    # DK placement points by finishing rank (0-based).
    rank_pts = np.zeros(n_players, dtype=np.float32)
    fill = min(len(DK_PLACEMENT_POINTS), n_players)
    rank_pts[:fill] = DK_PLACEMENT_POINTS[:fill]

    # Accumulators over sims (sums; divided by n_sims at the end).
    c_win = np.zeros(n_players); c_t5 = np.zeros(n_players)
    c_t10 = np.zeros(n_players); c_t20 = np.zeros(n_players)
    c_cut = np.zeros(n_players)
    s_place = np.zeros(n_players); s_score = np.zeros(n_players)

    k = min(top_n, n_players) - 1
    ar = np.arange(n_players)
    chunk = SIM.get("chunk", 4000)      # peak arrays ~ chunk x players x rounds
    done = 0
    while done < n_sims:
        m = min(chunk, n_sims - done)
        # Simulate m tournaments (float32 to halve memory).
        noise = rng.normal(0.0, sigma, size=(m, n_players, rounds)).astype(np.float32)
        sg = mu[None, :, None] + noise
        first36 = sg[:, :, :2].sum(axis=2)
        total72 = sg.sum(axis=2)

        # Cut: top-N-and-ties after R2.
        cut_line = np.partition(first36, n_players - 1 - k, axis=1)[:, n_players - 1 - k]
        made_cut = first36 >= cut_line[:, None]

        # Rank 72-hole totals (missed cut sinks to -inf; jitter breaks ties).
        total_eff = np.where(made_cut, total72, -np.inf)
        total_eff = total_eff + rng.normal(0, 1e-6, size=total_eff.shape)
        order = np.argsort(-total_eff, axis=1)
        finish = np.empty_like(order)
        np.put_along_axis(finish, order, np.broadcast_to(ar, order.shape), axis=1)

        c_win += (finish == 0).sum(axis=0)
        c_t5 += (finish < 5).sum(axis=0)
        c_t10 += (finish < 10).sum(axis=0)
        c_t20 += (finish < 20).sum(axis=0)
        c_cut += made_cut.sum(axis=0)

        # DK points: placement + hole-scoring (only rounds actually played).
        s_place += rank_pts[finish].sum(axis=0)
        sc = np.clip(DK_SCORING["base"] + DK_SCORING["slope"] * sg,
                     DK_SCORING["floor"], DK_SCORING["cap"])
        played = sc[:, :, :2].sum(axis=2) + sc[:, :, 2:].sum(axis=2) * made_cut
        s_score += played.sum(axis=0)
        done += m

    dk_placement = s_place / n_sims
    dk_scoring = s_score / n_sims
    return SimResult(
        names=names,
        win=c_win / n_sims, top_5=c_t5 / n_sims, top_10=c_t10 / n_sims,
        top_20=c_t20 / n_sims, make_cut=c_cut / n_sims,
        dk_points=dk_placement + dk_scoring,
        dk_placement=dk_placement, dk_scoring=dk_scoring, n_sims=n_sims,
    )


def simulate_dk_matrix(skills: dict[str, float], n_sims: int):
    """Like simulate() but RETAINS the per-sim DK point total for every player
    (n_sims x n_players), for the optimal-lineup exposure calc. float32, chunked."""
    names = list(skills.keys())
    mu = np.array([skills[n] for n in names], dtype=np.float32)
    P = len(names)
    rounds = EVENT["rounds"]
    sigma = SIM["round_sigma"] * SIM["wind_factor"]
    top_n = EVENT["cut_rule"]["top_n"]
    rng = np.random.default_rng(SIM["seed"])
    rank_pts = np.zeros(P, dtype=np.float32)
    fill = min(len(DK_PLACEMENT_POINTS), P)
    rank_pts[:fill] = DK_PLACEMENT_POINTS[:fill]

    dk = np.empty((n_sims, P), dtype=np.float32)
    k = min(top_n, P) - 1
    ar = np.arange(P)
    chunk = SIM.get("chunk", 4000)
    done = 0
    while done < n_sims:
        m = min(chunk, n_sims - done)
        sg = mu[None, :, None] + rng.normal(0.0, sigma, size=(m, P, rounds)).astype(np.float32)
        first36 = sg[:, :, :2].sum(axis=2)
        total72 = sg.sum(axis=2)
        cut_line = np.partition(first36, P - 1 - k, axis=1)[:, P - 1 - k]
        made_cut = first36 >= cut_line[:, None]
        eff = np.where(made_cut, total72, -np.inf) + rng.normal(0, 1e-6, size=total72.shape)
        order = np.argsort(-eff, axis=1)
        finish = np.empty_like(order)
        np.put_along_axis(finish, order, np.broadcast_to(ar, order.shape), axis=1)
        place = rank_pts[finish]
        sc = np.clip(DK_SCORING["base"] + DK_SCORING["slope"] * sg,
                     DK_SCORING["floor"], DK_SCORING["cap"])
        score = sc[:, :, :2].sum(axis=2) + sc[:, :, 2:].sum(axis=2) * made_cut
        dk[done:done + m] = place + score
        done += m
    return names, dk
