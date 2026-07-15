"""
Synthetic odds for `--demo`. Builds a believable market off the simulation so
the comparison tool has real structure to chew on with no API keys:

  * A hidden "true" probability that differs slightly from our model.
  * FanDuel + Pinnacle priced tight to truth with a small hold (the sharps).
  * Soft books priced with more hold and a few deliberate mispricings.
  * One prediction market (Polymarket) with its own noise.

The point isn't realism to the cent — it's to exercise devig + edge detection
end-to-end so wiring the real feeds is a drop-in swap.
"""

from __future__ import annotations

import numpy as np

from . import Quote
from ..odds_math import prob_to_decimal


def synth_winner_market(sim_rows: list[dict], seed: int = 3,
                        phase: str = "open") -> list[Quote]:
    """
    phase controls the CLV demo:
      'open'  -> looser, noisier market (Monday openers).
      'close' -> market has sharpened toward the model's truth + tighter holds
                 (Wednesday night). Running both then `clv` shows real movement.
    """
    # Different seed per phase so the two snapshots genuinely differ.
    rng = np.random.default_rng(seed + (100 if phase == "close" else 0))
    rows = [r for r in sim_rows if r["win"] > 0.0004]   # skip deep longshots

    # "True" market prob = model win prob nudged, renormalized. At close the
    # market converges toward the model (less nudge).
    nudge = 0.08 if phase == "close" else 0.15
    p_model = np.array([r["win"] for r in rows])
    p_true = np.clip(p_model * rng.normal(1.0, nudge, size=len(rows)), 1e-4, None)
    p_true /= p_true.sum()

    tighten = 0.5 if phase == "close" else 1.0   # holds shrink into the close
    books = [
        ("fanduel", "sportsbook", 1 + 0.05 * tighten, 0.03),   # sharp
        ("pinnacle", "sportsbook", 1 + 0.03 * tighten, 0.02),  # sharpest
        ("draftkings", "sportsbook", 1 + 0.12 * tighten, 0.08),
        ("betmgm", "sportsbook", 1 + 0.14 * tighten, 0.10),
        ("caesars", "sportsbook", 1 + 0.13 * tighten, 0.09),
        ("polymarket", "prediction_market", 1.02, 0.10),
        ("kalshi", "prediction_market", 1.02, 0.09),
    ]

    quotes: list[Quote] = []
    for source, kind, hold, noise in books:
        for i, r in enumerate(rows):
            # Each book's view = truth * multiplicative noise, then vig added.
            p = p_true[i] * rng.normal(1.0, noise)
            p = float(np.clip(p, 1e-4, 0.95)) * hold      # hold inflates implied prob
            p = min(p, 0.98)
            quotes.append(Quote(
                player=r["name"],
                market="win",
                source=source,
                source_kind=kind,
                decimal_odds=round(prob_to_decimal(p), 2),
            ))
    return quotes
