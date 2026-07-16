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

from config import SIM, DK_SCORING

# Single-round upside cap. The tournament model caps a round's contribution at ~30
# (to tame outliers in a 4-round sum), but a single showdown round genuinely tops
# out higher — a bogey-free 63 with 8 birdies + an eagle is ~40+ DK points — so we
# lift the cap here to keep the GPP ceiling differentiating.
LIVE_ROUND_CAP = 50.0


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
