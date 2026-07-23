"""
Optimal-lineup exposure — the sharp DFS leverage tool.

Simulate the tournament many times; in each sim, solve for the OPTIMAL DraftKings
lineup (6 golfers maximizing total DK points under the $50K cap); count how often
each golfer lands in that optimal lineup. That optimal% vs. the field's projected
ownership% is the real leverage: a high-optimal / low-owned player is one the
model says belongs in winning lineups that the field is under-rostering.

Per-sim solve is an exact DP knapsack (exactly 6 picks, salary-bucketed budget)
with backtracking for membership. Pruned to the top-`pool` players by mean points
(a $5K no-hoper is essentially never optimal), so ~2.5k sims run in a couple secs.
"""

from __future__ import annotations

import numpy as np

DK_ROSTER = 6
DK_CAP = 50_000


def compute(skills: dict[str, float], salaries: dict[str, int],
            n_sims: int = 2500, sigmas: dict[str, float] | None = None) -> dict[str, float]:
    """Full-tournament optimal exposure. {player: fraction of sims in the optimal lineup}."""
    from .simulate import simulate_dk_matrix

    names, dk = simulate_dk_matrix(skills, n_sims, sigmas=sigmas)
    return exposure_from_matrix(names, dk, salaries)


def exposure_from_matrix(names: list[str], dk, salaries: dict[str, int],
                         pool: int = 55, salary_relief: int = 20,
                         bucket: int = 100) -> dict[str, float]:
    """
    Optimal-lineup exposure for ANY per-sim DK points matrix (n_sims x n_players).
    Split out from compute() so the single-round showdown model can reuse the exact
    same knapsack on its own projections instead of the 4-round tournament sim.

    The candidate pool is the top `pool` players by mean points UNION the
    `salary_relief` cheapest players. Pruning by points alone drops the cheap
    punts you need to AFFORD a stud, which made an expensive favorite (Scheffler
    at ~30% of the cap) show 0% optimal even though he's the top play. bucket=100
    is exact for DK golf (salaries are multiples of $100), so no rounding wastes
    cap — the old $500 bucket rounded each salary up and made feasible stud+punt
    lineups infeasible.
    """
    n_sims = dk.shape[0]
    means = dk.mean(axis=0)

    salaried = [i for i, n in enumerate(names) if salaries.get(n)]
    if not salaried:
        return {}
    # Pool = top-by-points  ∪  cheapest-by-salary (salary relief for stud builds).
    by_points = sorted(salaried, key=lambda i: means[i], reverse=True)[:pool]
    by_cheap = sorted(salaried, key=lambda i: salaries[names[i]])[:salary_relief]
    idx = sorted(set(by_points) | set(by_cheap))

    # Salary in buckets, rounded UP so the DP never exceeds the real cap (with
    # bucket=100 there's nothing to round — DK golf salaries are multiples of 100).
    sal_b = np.array([int(np.ceil(salaries[names[i]] / bucket)) for i in idx])
    pts_all = dk[:, idx]
    B = DK_CAP // bucket
    n = len(idx)
    counts = np.zeros(n)
    NEG = -1e9

    for s in range(n_sims):
        pts = pts_all[s]
        # dp stage after each player: dp[j][b] = best points, exactly j picks, budget <= b.
        stages = [np.full((DK_ROSTER + 1, B + 1), NEG, dtype=np.float32)]
        stages[0][0, :] = 0.0
        for kk in range(n):
            prev = stages[-1]
            cur = prev.copy()
            c, p = int(sal_b[kk]), pts[kk]
            for j in range(DK_ROSTER, 0, -1):
                cand = np.full(B + 1, NEG, dtype=np.float32)
                cand[c:] = prev[j - 1, :B + 1 - c] + p
                np.maximum(cur[j], cand, out=cur[j])
            stages.append(cur)
        # Backtrack the optimal 6.
        b = int(np.argmax(stages[-1][DK_ROSTER]))
        j = DK_ROSTER
        for kk in range(n, 0, -1):
            if j == 0:
                break
            if stages[kk][j, b] > stages[kk - 1][j, b] + 1e-6:
                counts[kk - 1] += 1
                j -= 1
                b -= int(sal_b[kk - 1])

    return {names[idx[i]]: counts[i] / n_sims for i in range(n)}
