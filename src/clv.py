"""
Closing line value.

Two views off the snapshot store:

1. line_movement() — for each golfer, the sharp (FanDuel + Pinnacle) no-vig win
   probability at the OPENING run vs. the CLOSING run, and how it moved. Steamed
   (shortened) vs. drifted (lengthened) tells you where the money went. This is
   your "opening lines to compare Wednesday night" report.

2. grade_bets() — for every bet you logged, CLV vs. the closing no-vig sharp
   line. CLV = closing_fair_prob * your_decimal_odds - 1. Positive means you got
   a better price than the market's closing estimate of the true probability —
   the single best long-run predictor that a bet was +EV.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import store
from .compare import sharp_reference
from .odds_math import prob_to_decimal, decimal_to_american, expected_value


@dataclass
class Movement:
    player: str
    open_prob: float | None
    close_prob: float | None
    open_american: int | None
    close_american: int | None
    delta_prob: float | None       # close - open (positive => steamed/shortened)


def line_movement(market: str = "win", demo: bool = False) -> list[Movement]:
    o_run, c_run = store.opening_run(demo), store.closing_run(demo)
    if not o_run:
        return []
    open_fair = sharp_reference(store.quotes_at(o_run, market, demo))
    close_fair = sharp_reference(store.quotes_at(c_run, market, demo)) if c_run else {}

    players = set(open_fair) | set(close_fair)
    rows: list[Movement] = []
    for p in players:
        op, cp = open_fair.get(p), close_fair.get(p)
        rows.append(Movement(
            player=p,
            open_prob=op, close_prob=cp,
            open_american=decimal_to_american(prob_to_decimal(op)) if op else None,
            close_american=decimal_to_american(prob_to_decimal(cp)) if cp else None,
            delta_prob=(cp - op) if (op is not None and cp is not None) else None,
        ))
    # Biggest movers first (by absolute probability shift).
    rows.sort(key=lambda m: abs(m.delta_prob) if m.delta_prob is not None else -1,
              reverse=True)
    return rows


@dataclass
class BetGrade:
    player: str
    market: str
    source: str
    bet_american: int
    close_american: int | None
    clv: float | None        # EV of the bet priced at the closing fair prob
    beat_close: bool | None


def grade_bets(demo: bool = False) -> list[BetGrade]:
    c_run = store.closing_run(demo)
    close_by_market: dict[str, dict[str, float]] = {}
    bets = store.load_bets()
    if c_run:
        for mkt in {b["market"] for b in bets}:
            close_by_market[mkt] = sharp_reference(store.quotes_at(c_run, mkt, demo))

    grades: list[BetGrade] = []
    for b in bets:
        close_fair = close_by_market.get(b["market"], {}).get(b["player"])
        clv = expected_value(close_fair, b["decimal_odds"]) if close_fair else None
        grades.append(BetGrade(
            player=b["player"], market=b["market"], source=b["source"],
            bet_american=decimal_to_american(b["decimal_odds"]),
            close_american=decimal_to_american(prob_to_decimal(close_fair)) if close_fair else None,
            clv=clv,
            beat_close=(clv > 0) if clv is not None else None,
        ))
    grades.sort(key=lambda g: g.clv if g.clv is not None else -99, reverse=True)
    return grades
