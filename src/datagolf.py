"""
DataGolf ingestion — strokes-gained skill ratings + the tournament field.

DataGolf is the single best source for SG components (incl. events off the PGA
Tour) plus a pre-tournament model we can benchmark against. Endpoints used:

    /preds/skill-ratings   -> per-player baseline SG: ott/app/arg/putt + driving
    /field-updates         -> who's actually in the field this week
    /preds/pre-tournament  -> DataGolf's own win/top-N probs (sanity check)

Everything degrades gracefully: with no DATAGOLF_KEY we synthesize a plausible
156-player field so the whole pipeline runs end-to-end (`--demo`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import requests

BASE = "https://feeds.datagolf.com"


@dataclass
class Player:
    name: str
    dg_id: int | None = None
    # Baseline strokes-gained per round vs. a scratch/field baseline.
    sg_ott: float = 0.0
    sg_app: float = 0.0
    sg_arg: float = 0.0
    sg_putt: float = 0.0
    sg_total: float = 0.0
    driving_acc: float = 0.0   # standardized (z) vs. field, +ve = straighter
    # Prior links/Open form in [0, 1]; 1 = strong links pedigree. From history
    # feeds when available, else a neutral 0.5.
    links_history: float = 0.5
    # Trailing per-round SG:Total over the last 16 / 24 measured rounds (recent
    # form). None when unavailable (no U.S. ShotLink rounds — LIV / DP-only).
    form_16: float | None = None
    form_24: float | None = None
    # Scoring profile (season %). birdie_pct = Birdie-or-Better % (ceiling/GPP);
    # bogey_pct = Bogey Avoidance, % of holes bogey-or-worse (floor — LOWER better).
    birdie_pct: float | None = None
    bogey_pct: float | None = None
    # Standardized bogey avoidance (z-score, positive = avoids bogeys). Feeds a
    # links-specific "floor" bonus in course_fit. None when no bogey data.
    bogey_z: float | None = None
    flags: list[str] = field(default_factory=list)


def _key(explicit: str | None) -> str | None:
    return explicit or os.getenv("DATAGOLF_KEY")


def fetch_skill_ratings(api_key: str | None = None) -> dict[str, Player]:
    """Player -> baseline SG components from /preds/skill-ratings."""
    key = _key(api_key)
    if not key:
        raise RuntimeError("DATAGOLF_KEY not set")
    r = requests.get(
        f"{BASE}/preds/skill-ratings",
        params={"display": "value", "file_format": "json", "key": key},
        timeout=30,
    )
    r.raise_for_status()
    rows = r.json().get("players", r.json() if isinstance(r.json(), list) else [])
    out: dict[str, Player] = {}
    for row in rows:
        name = _clean_name(row.get("player_name", ""))
        if not name:
            continue
        out[name] = Player(
            name=name,
            dg_id=row.get("dg_id"),
            sg_ott=_f(row.get("sg_ott")),
            sg_app=_f(row.get("sg_app")),
            sg_arg=_f(row.get("sg_arg")),
            sg_putt=_f(row.get("sg_putt")),
            sg_total=_f(row.get("sg_total")),
            driving_acc=_f(row.get("driving_acc")),
        )
    return out


def fetch_field(api_key: str | None = None, tour: str = "pga") -> list[str]:
    """Names in this week's field via /field-updates."""
    key = _key(api_key)
    if not key:
        raise RuntimeError("DATAGOLF_KEY not set")
    r = requests.get(
        f"{BASE}/field-updates",
        params={"tour": tour, "file_format": "json", "key": key},
        timeout=30,
    )
    r.raise_for_status()
    return [_clean_name(p.get("player_name", "")) for p in r.json().get("field", [])]


def load_field(api_key: str | None = None, demo: bool = False) -> list[Player]:
    """
    Assemble the modeling field: intersect skill ratings with the entry list.
    Falls back to a synthetic field when demo=True or no key is present.
    """
    if demo or not _key(api_key):
        return synthetic_field()

    ratings = fetch_skill_ratings(api_key)
    try:
        names = fetch_field(api_key)
    except Exception as e:  # field feed can lag; fall back to all rated players
        print(f"[warn] field-updates failed ({e}); using full skill-ratings list")
        names = list(ratings.keys())

    players = [ratings[n] for n in names if n in ratings]
    if not players:
        print("[warn] no field/rating overlap; using synthetic field")
        return synthetic_field()
    return players


# --------------------------------------------------------------------------
# Synthetic field — lets the pipeline run with zero API access.
# Skill spread mirrors a real major: a handful of ~+2.0 SG stars down to
# a long tail of near-zero journeymen.
# --------------------------------------------------------------------------
def synthetic_field(n: int = 156) -> list[Player]:
    import numpy as np

    rng = np.random.default_rng(11)
    names = _DEMO_NAMES + [f"Player {i:03d}" for i in range(len(_DEMO_NAMES), n)]
    # Total SG skill: exponential-ish top end, most of the field clustered low.
    totals = np.sort(rng.gamma(shape=2.0, scale=0.55, size=n))[::-1] - 0.4
    players: list[Player] = []
    for i in range(n):
        t = float(totals[i])
        # Split total into components with realistic proportions + noise.
        ott = t * 0.28 + rng.normal(0, 0.15)
        app = t * 0.38 + rng.normal(0, 0.18)   # approach is the biggest slice
        arg = t * 0.14 + rng.normal(0, 0.10)
        putt = t - (ott + app + arg)
        players.append(Player(
            name=names[i],
            dg_id=i,
            sg_ott=round(ott, 3), sg_app=round(app, 3),
            sg_arg=round(arg, 3), sg_putt=round(putt, 3),
            sg_total=round(t, 3),
            driving_acc=round(float(rng.normal(0, 1)), 2),
            links_history=round(float(rng.uniform(0.3, 0.9)), 2),
        ))
    return players


_DEMO_NAMES = [
    "Scottie Scheffler", "Rory McIlroy", "Jon Rahm", "Xander Schauffele",
    "Ludvig Aberg", "Collin Morikawa", "Viktor Hovland", "Tommy Fleetwood",
    "Bryson DeChambeau", "Justin Thomas", "Shane Lowry", "Tyrrell Hatton",
    "Robert MacIntyre", "Matt Fitzpatrick", "Hideki Matsuyama", "Patrick Cantlay",
    "Brooks Koepka", "Jordan Spieth", "Tommy Detry", "Sepp Straka",
]


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _clean_name(name: str) -> str:
    """DataGolf returns 'Last, First' — normalize to 'First Last'."""
    name = (name or "").strip()
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
        return f"{first} {last}"
    return name
