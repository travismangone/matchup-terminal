"""
Model validation — market-agreement diagnostics.

We can't measure true calibration (did 20% shots win 20%?) until the event is
over, so before then the best check is coherence against the sharp market:

  * Rank agreement   — does the model order players like FanDuel does? (Spearman)
  * Dispersion       — is the model over/under-concentrated on the favorites vs.
                       the market? A sim with too little round-to-round variance
                       crowns favorites too often; too much flattens the field.
  * Disagreements    — the biggest model-likes and model-fades. This is where
                       your edge comes from AND where the model is most likely
                       wrong, so you look at them by hand.
  * Group residual   — mean (model − market) for real-SG players vs. OWGR-est
                       ones. We EXPECT OWGR-est (LIV/Intl) to sit below market,
                       confirming the known OWGR-underrates-LIV bias.

None of this proves the model is right — it proves it's sane and shows where it
diverges from the market on purpose. True calibration is the backtest TODO.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Disagreement:
    player: str
    model: float
    market: float
    diff: float          # model - market (probability points)
    flags: list[str] = field(default_factory=list)


@dataclass
class Validation:
    n: int
    spearman: float
    mean_abs_diff: float
    fav_player: str
    fav_model: float
    fav_market: float
    top5_model: float
    top5_market: float
    likes: list[Disagreement]
    fades: list[Disagreement]
    resid_pga: float | None
    resid_owgr: float | None
    dispersion_note: str


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def ranks(a):
        order = a.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a))
        return r
    rx, ry = ranks(x), ranks(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom else 0.0


def validate(model_probs: dict[str, float],
             market_fair: dict[str, float],
             flags_by_player: dict[str, list[str]] | None = None,
             top: int = 8) -> Validation:
    flags_by_player = flags_by_player or {}
    names = [n for n in model_probs if n in market_fair]
    m = np.array([model_probs[n] for n in names])
    q = np.array([market_fair[n] for n in names])

    spear = _spearman(m, q)
    mad = float(np.abs(m - q).mean())

    # Favorite + top-5 concentration.
    fav_i = int(q.argmax())
    order_q = q.argsort()[::-1]
    top5 = order_q[:5]
    top5_model = float(m[top5].sum())
    top5_market = float(q[top5].sum())

    # Read the favorite ratio and the top-5 ratio together. round_sigma moves
    # BOTH the same direction, so only a same-direction miss is a variance
    # problem. If they diverge (favorite too high but top-5 too thin, or vice
    # versa) the skill SHAPE at the top differs from market and sigma won't fix
    # it — that's a skill-input difference, and often your intended edge.
    fav_ratio = (float(m[fav_i]) / q[fav_i]) if q[fav_i] else 1.0
    t5_ratio = (top5_model / top5_market) if top5_market else 1.0
    hi, lo = 1.10, 0.90
    if fav_ratio > hi and t5_ratio > hi:
        disp = ("Model OVER-concentrated top-to-bottom — too little round variance. "
                "Consider RAISING SIM.round_sigma.")
    elif fav_ratio < lo and t5_ratio < lo:
        disp = ("Model UNDER-concentrated top-to-bottom — too much round variance. "
                "Consider LOWERING SIM.round_sigma.")
    elif abs(fav_ratio - 1) < 0.10 and abs(t5_ratio - 1) < 0.10:
        disp = "Concentration matches market at both the favorite and top-5 level."
    else:
        disp = (f"SHAPE mismatch (favorite {fav_ratio:.2f}x, top-5 {t5_ratio:.2f}x market) "
                f"— the model's skill gap at the very top differs from the market. "
                f"A skill-input difference, not a variance one; sigma won't reconcile it.")

    diffs = [Disagreement(n, model_probs[n], market_fair[n],
                          model_probs[n] - market_fair[n],
                          flags_by_player.get(n, []))
             for n in names]
    likes = sorted(diffs, key=lambda d: d.diff, reverse=True)[:top]
    fades = sorted(diffs, key=lambda d: d.diff)[:top]

    def _resid(pred):
        vals = [d.diff for d in diffs if pred(d)]
        return float(np.mean(vals)) if vals else None
    resid_pga = _resid(lambda d: "owgr-est" not in d.flags and "no-data" not in d.flags)
    resid_owgr = _resid(lambda d: "owgr-est" in d.flags)

    return Validation(
        n=len(names), spearman=spear, mean_abs_diff=mad,
        fav_player=names[fav_i], fav_model=float(m[fav_i]), fav_market=float(q[fav_i]),
        top5_model=top5_model, top5_market=top5_market,
        likes=likes, fades=fades, resid_pga=resid_pga, resid_owgr=resid_owgr,
        dispersion_note=disp,
    )
