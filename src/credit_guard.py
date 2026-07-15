"""
Credit guard — stop The Odds API spend from running away.

The Odds API bills CREDITS (markets × regions per /odds call). Only the sportsbook
winner pull (incl. FanDuel) costs credits — every other feed (PGA SG, OWGR, Kalshi,
Polymarket) is free. A dashboard on the matchup terminal, with viewers clicking
"Pull live odds", could drain a monthly quota fast. This guard gates every real
pull on two limits:

- daily_limit   : max credits to spend per calendar day (local tally, persisted)
- min_remaining : hard floor on the API's own remaining counter; near empty we stop

State lives in data/credit_usage.json so the budget survives restarts. Set in .env:
    ODDS_DAILY_CREDIT_LIMIT=50
    ODDS_MIN_REMAINING=20

Ported from edge-finder/src/credit_guard.py.
"""

from __future__ import annotations

import json
import os
from datetime import date

STATE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "data", "credit_usage.json")


class CreditGuard:
    def __init__(self, daily_limit: int | None = None, min_remaining: int | None = None):
        self.daily_limit = daily_limit if daily_limit is not None else \
            int(os.getenv("ODDS_DAILY_CREDIT_LIMIT", "50"))
        self.min_remaining = min_remaining if min_remaining is not None else \
            int(os.getenv("ODDS_MIN_REMAINING", "20"))
        self._state = self._load()

    def _load(self) -> dict:
        try:
            with open(STATE_PATH) as f:
                s = json.load(f)
        except Exception:
            s = {}
        today = date.today().isoformat()
        if s.get("date") != today:          # new day -> reset the daily tally
            s = {"date": today, "used_today": 0, "last_remaining": s.get("last_remaining")}
        return s

    def _save(self) -> None:
        os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
        with open(STATE_PATH, "w") as f:
            json.dump(self._state, f)

    def blocked_reason(self) -> str | None:
        """Why a pull would be blocked right now, or None if it's allowed."""
        if self._state["used_today"] >= self.daily_limit:
            return (f"daily limit reached ({self._state['used_today']}/"
                    f"{self.daily_limit} credits today)")
        rem = self._state.get("last_remaining")
        if rem is not None and rem <= self.min_remaining:
            return f"near monthly cap ({rem} left, floor {self.min_remaining})"
        return None

    def allowed(self) -> bool:
        return self.blocked_reason() is None

    def record(self, remaining: int | None) -> None:
        """Call after each pull with the API's x-requests-remaining value.
        Infers spend from the drop in remaining and updates the daily tally."""
        if remaining is None:
            return
        prev = self._state.get("last_remaining")
        if prev is not None and remaining < prev:
            self._state["used_today"] += (prev - remaining)
        self._state["last_remaining"] = remaining
        self._save()

    @property
    def used_today(self) -> int:
        return self._state["used_today"]

    @property
    def last_remaining(self) -> int | None:
        return self._state.get("last_remaining")
