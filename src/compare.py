"""
Best-plays engine.

Three probability estimates per player, per market:

  1. model_fair  — from the Monte Carlo simulation.
  2. sharp_fair  — de-vig FanDuel + Pinnacle separately, then average. This is
                   the market's sharpest truth estimate.
  3. blend_fair  — MODEL_BLEND * model_fair + (1-MODEL_BLEND) * sharp_fair,
                   renormalized. The number we actually bet against.

For every price at every OTHER source (soft books + prediction markets) we
compute the edge vs. blend_fair. We also surface edge_vs_sharp and edge_vs_model
separately so you can see WHERE the value comes from:

  * +edge vs sharp  -> a soft book / market is simply lagging the sharps.
  * +edge vs model but not sharp -> your simulation disagrees with the market;
    higher variance, this is your independent read.

Highest conviction = both agree. Everything is ranked by edge and sized with
fractional Kelly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .odds import Quote
from .odds_math import remove_vig, expected_value, kelly_fraction
from config import (
    SHARP_BOOKS, MODEL_BLEND, MIN_EDGE, MIN_FAIR_PROB, MAX_PRICE_DECIMAL,
    KELLY_MULTIPLIER,
)


@dataclass
class Play:
    player: str
    market: str
    source: str               # where you'd place the bet
    source_kind: str
    offered_decimal: float
    model_fair: float
    sharp_fair: float | None
    blend_fair: float
    edge: float               # vs blend_fair, per $1
    edge_vs_sharp: float | None
    edge_vs_model: float
    kelly: float
    flags: list[str] = field(default_factory=list)


def _devig_book(quotes: list[Quote], book: str) -> dict[str, float]:
    """Fair probs for one book across the players it quotes in this market."""
    q = [x for x in quotes if x.source == book]
    if len(q) < 2:
        return {}
    fair = remove_vig([x.decimal_odds for x in q])
    return {x.player: f for x, f in zip(q, fair)}


def sharp_reference(quotes: list[Quote]) -> dict[str, float]:
    """
    Consensus fair prob per player from the sharp books (FanDuel + Pinnacle):
    de-vig each independently, then average across whichever sharps quote them.
    """
    per_book = {b: _devig_book(quotes, b) for b in SHARP_BOOKS}
    players = {p for d in per_book.values() for p in d}
    out: dict[str, float] = {}
    for p in players:
        vals = [d[p] for d in per_book.values() if p in d]
        if vals:
            out[p] = sum(vals) / len(vals)
    return out


def _blend(model_p: float, sharp_p: float | None) -> float:
    if sharp_p is None:
        return model_p
    return MODEL_BLEND * model_p + (1.0 - MODEL_BLEND) * sharp_p


def find_plays(
    quotes: list[Quote],
    model_probs: dict[str, dict[str, float]],
    min_edge: float = MIN_EDGE,
    skill_flags: dict[str, list[str]] | None = None,
) -> list[Play]:
    """
    quotes:       every price we pulled (all books, all markets).
    model_probs:  {market: {player: sim_prob}} from the simulation.
    skill_flags:  {player: [...]} skill-source flags (owgr-est / no-data) so the
                  board shows when an "edge" rests on an estimated, not measured,
                  skill — those are model noise more often than real value.
    """
    skill_flags = skill_flags or {}
    plays: list[Play] = []

    # Group quotes by market so devig is done within a market.
    markets = sorted({q.market for q in quotes})
    for market in markets:
        mq = [q for q in quotes if q.market == market]
        sharp = sharp_reference(mq)
        model_market = model_probs.get(market, {})

        for q in mq:
            # Don't grade a sharp book against itself.
            if q.source in SHARP_BOOKS:
                continue
            # Skip non-actionable lottery prices (deep-longshot placeholders).
            if q.decimal_odds > MAX_PRICE_DECIMAL:
                continue
            model_p = model_market.get(q.player)
            if model_p is None:
                continue
            sharp_p = sharp.get(q.player)
            blend_p = _blend(model_p, sharp_p)

            # Noise floor: tail EV is dominated by devig error, skip it.
            if blend_p < MIN_FAIR_PROB:
                continue

            edge = expected_value(blend_p, q.decimal_odds)
            if edge < min_edge:
                continue

            e_sharp = expected_value(sharp_p, q.decimal_odds) if sharp_p else None
            e_model = expected_value(model_p, q.decimal_odds)

            flags: list[str] = []
            if sharp_p is None:
                flags.append("no-sharp-line")       # model-only read, higher variance
            elif e_sharp is not None and e_sharp > 0 and e_model > 0:
                flags.append("model+sharp agree")   # highest conviction
            if q.source_kind == "prediction_market":
                flags.append("prediction-market")
            # Skill-source flags — an edge on estimated skill is usually noise.
            for sf in skill_flags.get(q.player, []):
                if sf in ("owgr-est", "no-data"):
                    flags.append(sf)

            plays.append(Play(
                player=q.player, market=market,
                source=q.source, source_kind=q.source_kind,
                offered_decimal=q.decimal_odds,
                model_fair=model_p, sharp_fair=sharp_p, blend_fair=blend_p,
                edge=edge, edge_vs_sharp=e_sharp, edge_vs_model=e_model,
                kelly=kelly_fraction(blend_p, q.decimal_odds) * KELLY_MULTIPLIER,
                flags=flags,
            ))

    plays.sort(key=lambda p: p.edge, reverse=True)
    return plays
