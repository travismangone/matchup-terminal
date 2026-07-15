"""
DraftKings salaries — the DFS player pool for the slate.

Loads DKSalaries CSV (copied to data/dk_salaries.csv) into a per-player map keyed
by normalized name, and exposes the pool as the modeling field so EVERY rosterable
player gets projections. Salary + DK id + DK's own AvgPointsPerGame ride along.
"""

from __future__ import annotations

import csv
import os

from .match import normalize_name, build_index, match_player

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_CSV = os.path.join(DATA, "dk_salaries.csv")


def load(path: str | None = None) -> dict[str, dict]:
    """{normalized_name: {name, salary, dk_id, dkppg}} for the slate."""
    path = path or DEFAULT_CSV
    out: dict[str, dict] = {}
    if not os.path.exists(path):
        print(f"[warn] DK salaries not found at {path}")
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("Name") or "").strip()
            if not name:
                continue
            out[normalize_name(name)] = {
                "name": name,
                "salary": _int(row.get("Salary")),
                "dk_id": (row.get("ID") or "").strip(),
                "dkppg": _float(row.get("AvgPointsPerGame")),
            }
    return out


def pool_names(path: str | None = None) -> list[str]:
    """Canonical DK display names — the DFS field."""
    return [v["name"] for v in load(path).values()]


def index(salaries: dict[str, dict] | None = None) -> dict:
    """Fuzzy name index over the DK pool for joining salaries to model players."""
    salaries = salaries if salaries is not None else load()
    return build_index(list(salaries.keys()))


def lookup(name: str, salaries: dict[str, dict], idx: dict) -> dict | None:
    key = match_player(name, idx)
    return salaries.get(key) if key else None


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
