"""
Core odds math. Mirrors edge-finder/src/odds_math.py so the two projects speak
the same language — prices carried internally as DECIMAL odds; de-vig via the
proportional (multiplicative) method.
"""

from __future__ import annotations


# --------------------------------------------------------------------------
# Format conversions
# --------------------------------------------------------------------------
def american_to_decimal(american: float) -> float:
    """+150 -> 2.50, -200 -> 1.50"""
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> int:
    """2.50 -> +150, 1.50 -> -200"""
    if decimal >= 2.0:
        return round((decimal - 1.0) * 100.0)
    return round(-100.0 / (decimal - 1.0))


def decimal_to_implied(decimal: float) -> float:
    """Decimal odds -> implied probability (vig included)."""
    return 1.0 / decimal


def prob_to_decimal(prob: float) -> float:
    """Fair probability -> decimal odds. A 25% shot -> 4.00."""
    if prob <= 0.0:
        raise ValueError("probability must be > 0")
    return 1.0 / prob


# --------------------------------------------------------------------------
# Vig removal
#
# Golf outrights are ~150-way markets. The proportional devig over whatever the
# book actually quotes is robust to a couple of missing longshots — we don't
# require the book to cover every player.
# --------------------------------------------------------------------------
def overround(decimals: list[float]) -> float:
    return sum(decimal_to_implied(d) for d in decimals)


def remove_vig(decimals: list[float]) -> list[float]:
    """Set of mutually-exclusive prices -> fair probs summing to 1.0."""
    implied = [decimal_to_implied(d) for d in decimals]
    total = sum(implied)
    if total <= 0:
        raise ValueError("prices sum to zero")
    return [p / total for p in implied]


# --------------------------------------------------------------------------
# Edge / staking
# --------------------------------------------------------------------------
def expected_value(fair_prob: float, decimal_odds: float) -> float:
    """EV per $1 staked. Positive => +EV. EV = p * odds - 1."""
    return fair_prob * decimal_odds - 1.0


def kelly_fraction(fair_prob: float, decimal_odds: float) -> float:
    """Full-Kelly stake as a fraction of bankroll. 0 for -EV bets."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    p = fair_prob
    q = 1.0 - p
    return max(0.0, (b * p - q) / b)
