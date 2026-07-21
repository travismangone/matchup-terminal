"""
Line snapshot store — append-only log of every price we've seen, so we can
measure how the market moved from open to close.

Each `snapshot()` call stamps the whole batch of quotes with one timestamp (a
"run") and appends them to data/lines.jsonl. The FIRST run that quotes a golfer
is that golfer's opening line; the LAST run before the event is the closing line.
Run it Monday to capture openers, again Wednesday night to capture closers, then
`clv` compares.

Also stores your logged bets (data/bets.jsonl) for closing-line-value grading.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .odds import Quote

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
LINES = os.path.join(DATA_DIR, "lines.jsonl")
LINES_DEMO = os.path.join(DATA_DIR, "lines_demo.jsonl")
BETS = os.path.join(DATA_DIR, "bets.jsonl")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _lines_path(demo: bool) -> str:
    """Demo snapshots go to a separate file so they never pollute real CLV data."""
    return LINES_DEMO if demo else LINES


# --------------------------------------------------------------------------
# Line snapshots
# --------------------------------------------------------------------------
def snapshot(quotes: list[Quote], ts: str | None = None, demo: bool = False) -> str:
    """Append all quotes as one timestamped run. Returns the run timestamp."""
    os.makedirs(DATA_DIR, exist_ok=True)
    ts = ts or _now_iso()
    with open(_lines_path(demo), "a") as f:
        for q in quotes:
            f.write(json.dumps({
                "ts": ts,
                "player": q.player,
                "market": q.market,
                "source": q.source,
                "source_kind": q.source_kind,
                "decimal_odds": round(q.decimal_odds, 4),
            }) + "\n")
    return ts


def load_lines(demo: bool = False) -> list[dict]:
    path = _lines_path(demo)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def runs(demo: bool = False) -> list[str]:
    """Distinct snapshot timestamps, oldest first."""
    return sorted({r["ts"] for r in load_lines(demo)})


def quotes_at(ts: str, market: str, demo: bool = False) -> list[Quote]:
    """Reconstruct Quotes from one run + market."""
    out = []
    for r in load_lines(demo):
        if r["ts"] == ts and r["market"] == market:
            out.append(Quote(
                player=r["player"], market=r["market"], source=r["source"],
                source_kind=r["source_kind"], decimal_odds=r["decimal_odds"],
            ))
    return out


def opening_run(demo: bool = False) -> str | None:
    r = runs(demo)
    return r[0] if r else None


def closing_run(demo: bool = False) -> str | None:
    r = runs(demo)
    return r[-1] if r else None


def _run_players(ts: str, demo: bool) -> set:
    return {r["player"] for r in load_lines(demo)
            if r["ts"] == ts and r["market"] == "win"}


def event_opening_run(demo: bool = False, min_overlap: float = 0.6) -> str | None:
    """
    Opening run FOR THE CURRENT EVENT. The store is append-only across events, so
    the literal first run is often last week's board — comparing this week's
    closer to it produces garbage deltas (a player's Open price vs. his 3M price).

    We treat the latest run's field as "this event", then walk runs oldest-first
    and return the first whose field overlaps it by >= min_overlap. That anchors
    the opener to when THIS event's market first posted, not the prior event's.
    """
    r = runs(demo)
    if not r:
        return None
    ref = _run_players(r[-1], demo)
    if not ref:
        return r[0]
    for ts in r:
        players = _run_players(ts, demo)
        if players and len(players & ref) / min(len(players), len(ref)) >= min_overlap:
            return ts
    return r[-1]


# --------------------------------------------------------------------------
# Bet log
# --------------------------------------------------------------------------
def log_bet(player: str, market: str, source: str, decimal_odds: float,
            stake: float = 1.0, ts: str | None = None) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(BETS, "a") as f:
        f.write(json.dumps({
            "ts": ts or _now_iso(),
            "player": player, "market": market, "source": source,
            "decimal_odds": round(decimal_odds, 4), "stake": stake,
        }) + "\n")


def load_bets() -> list[dict]:
    if not os.path.exists(BETS):
        return []
    with open(BETS) as f:
        return [json.loads(line) for line in f if line.strip()]
