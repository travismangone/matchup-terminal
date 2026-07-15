from dataclasses import dataclass


@dataclass
class Quote:
    """One price, on one player, from one source."""
    player: str
    market: str            # 'win', 'top_5', ...
    source: str            # 'fanduel', 'pinnacle', 'draftkings', 'polymarket', ...
    source_kind: str       # 'sportsbook' | 'prediction_market'
    decimal_odds: float
