"""
Credit-safe scheduled snapshot for the golf tab — run by a Render Cron Job
(e.g. hourly on tournament days).

Pulls live odds + rebuilds the golf state ONCE, gated by the CreditGuard (which
respects ODDS_DAILY_CREDIT_LIMIT / ODDS_MIN_REMAINING). If the guard says stop,
this exits quietly without spending. Viewers never trigger a pull; only this
does, so your Odds API spend is fixed and predictable.

    python cron_snapshot.py
"""

from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass

from src import dashboard
from src.credit_guard import CreditGuard


def main() -> int:
    guard = CreditGuard()
    blocked = guard.blocked_reason()
    if blocked:
        print(f"[cron] skip — credit guard: {blocked}")
        return 0
    try:
        ts = dashboard.pull_and_snapshot()
        print(f"[cron] snapshot ok @ {ts}")
        return 0
    except Exception as e:
        print(f"[cron] snapshot failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
