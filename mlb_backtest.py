"""
MLB Matchup Backtest
--------------------
Tests the accuracy of the matchup engine's projections against actual results.

For each past date:
  1. Runs analyze_slate(date) with that date as the data cutoff — no look-ahead.
  2. Fetches actual batting stats from completed game boxscores via the MLB Stats API.
  3. Computes actual DK score and wOBA per batter from those stats.
  4. Caches per-date results to data/backtest/<date>.json so re-runs are instant.

Key metrics:
  - Pearson correlation: projected DK vs actual DK
  - Spearman rank correlation: how well the model ranks batters by actual output
  - MAE / RMSE of DK projection
  - Pearson/Spearman for projected wOBA vs actual single-game wOBA
  - Calibration bins: projected DK quintiles vs mean actual DK (is Q5 really better?)
  - wOBA edge rank correlation: does a high woba_edge predict better actual wOBA?
"""

from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional

import statsapi

import matchup_tool as _mt
from matchup_tool import DK_SCORING, WOBA_W

_DATA_DIR = Path(__file__).parent / "data" / "backtest"


def _ensure_dir():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Actual result fetching from MLB Stats API boxscores
# ---------------------------------------------------------------------------

def fetch_actual_batting_lines(game_pk: int) -> Dict[int, Dict]:
    """Return {player_id: batting_stats} for all batters in a completed game."""
    try:
        box = statsapi.boxscore_data(int(game_pk))
    except Exception:
        return {}

    result: Dict[int, Dict] = {}
    for side in ("away", "home"):
        team = box.get(side) or {}
        players = team.get("players") or {}
        for _key, info in players.items():
            batting = ((info.get("stats") or {}).get("batting") or {})
            if not batting:
                continue
            person = info.get("person") or {}
            try:
                pid = int(person.get("id") or 0)
            except (TypeError, ValueError):
                continue
            if not pid:
                continue

            ab   = int(batting.get("atBats") or 0)
            hits = int(batting.get("hits") or 0)
            d2   = int(batting.get("doubles") or 0)
            d3   = int(batting.get("triples") or 0)
            hr   = int(batting.get("homeRuns") or 0)
            rbi  = int(batting.get("rbi") or 0)
            runs = int(batting.get("runs") or 0)
            bb   = int(batting.get("baseOnBalls") or 0)
            ibb  = int(batting.get("intentionalWalks") or 0)
            hbp  = int(batting.get("hitByPitch") or 0)
            sb   = int(batting.get("stolenBases") or 0)
            h1   = max(0, hits - d2 - d3 - hr)
            pa   = ab + bb + ibb + hbp

            if pa == 0:
                continue

            result[pid] = {
                "ab": ab, "pa": pa, "hits": hits,
                "singles": h1, "doubles": d2, "triples": d3, "hr": hr,
                "rbi": rbi, "runs": runs,
                "bb": bb + ibb,
                "hbp": hbp, "sb": sb,
            }
    return result


def _actual_dk(stats: Dict) -> float:
    """Compute DraftKings fantasy score from actual batting stats."""
    return (
        DK_SCORING["single"] * stats.get("singles", 0) +
        DK_SCORING["double"] * stats.get("doubles", 0) +
        DK_SCORING["triple"] * stats.get("triples", 0) +
        DK_SCORING["hr"]     * stats.get("hr", 0) +
        DK_SCORING["rbi"]    * stats.get("rbi", 0) +
        DK_SCORING["run"]    * stats.get("runs", 0) +
        DK_SCORING["bb"]     * stats.get("bb", 0) +
        DK_SCORING["hbp"]    * stats.get("hbp", 0) +
        DK_SCORING["sb"]     * stats.get("sb", 0)
    )


def _actual_woba(stats: Dict) -> Optional[float]:
    """Compute single-game wOBA from actual stats. Returns None if no PA."""
    pa = stats.get("pa", 0)
    if pa == 0:
        return None
    num = (
        WOBA_W["bb"]  * stats.get("bb", 0) +
        WOBA_W["hbp"] * stats.get("hbp", 0) +
        WOBA_W["h1"]  * stats.get("singles", 0) +
        WOBA_W["h2"]  * stats.get("doubles", 0) +
        WOBA_W["h3"]  * stats.get("triples", 0) +
        WOBA_W["hr"]  * stats.get("hr", 0)
    )
    return num / pa


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def _pearson(xs: list, ys: list) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx  = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy  = math.sqrt(sum((y - my) ** 2 for y in ys))
    return (num / (dx * dy)) if dx and dy else None


def _spearman(xs: list, ys: list) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    def _ranks(v):
        si = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        for rank, i in enumerate(si):
            r[i] = rank + 1.0
        return r
    return _pearson(_ranks(xs), _ranks(ys))


def _rnd(v, digits: int) -> Optional[float]:
    return round(v, digits) if v is not None else None


def compute_metrics(records: List[Dict]) -> Dict:
    """Aggregate accuracy metrics from a list of matched (proj, actual) records."""
    if not records:
        return {"n_records": 0}

    dk_pairs   = [(r["proj_dk"],  r["actual_dk"])   for r in records
                  if r.get("proj_dk") is not None and r.get("actual_dk") is not None]
    woba_pairs = [(r["proj_woba"], r["actual_woba"]) for r in records
                  if r.get("proj_woba") is not None and r.get("actual_woba") is not None]
    edge_pairs = [(r["woba_edge"], r["actual_woba"]) for r in records
                  if r.get("woba_edge") is not None and r.get("actual_woba") is not None]

    m: Dict = {"n_records": len(records), "n_dk": len(dk_pairs), "n_woba": len(woba_pairs)}

    if dk_pairs:
        pd_ = [p for p, _ in dk_pairs]
        ad_ = [a for _, a in dk_pairs]
        m["dk_corr"]        = _rnd(_pearson(pd_, ad_), 4)
        m["dk_spearman"]    = _rnd(_spearman(pd_, ad_), 4)
        m["dk_mae"]         = _rnd(sum(abs(p - a) for p, a in dk_pairs) / len(dk_pairs), 2)
        m["dk_rmse"]        = _rnd(math.sqrt(sum((p - a) ** 2 for p, a in dk_pairs) / len(dk_pairs)), 2)
        m["dk_proj_mean"]   = _rnd(sum(pd_) / len(pd_), 2)
        m["dk_actual_mean"] = _rnd(sum(ad_) / len(ad_), 2)
        m["dk_bias"]        = _rnd(sum(pd_) / len(pd_) - sum(ad_) / len(ad_), 2)

        if len(dk_pairs) >= 10:
            srt  = sorted(dk_pairs, key=lambda x: x[0])
            n    = len(srt)
            sz   = n // 5
            bins = []
            for qi in range(5):
                chunk = srt[qi * sz: ((qi + 1) * sz if qi < 4 else n)]
                bins.append({
                    "label":       f"Q{qi + 1}",
                    "proj_mean":   _rnd(sum(p for p, _ in chunk) / len(chunk), 2),
                    "actual_mean": _rnd(sum(a for _, a in chunk) / len(chunk), 2),
                    "n":           len(chunk),
                })
            m["dk_bins"] = bins

    if woba_pairs:
        pw = [p for p, _ in woba_pairs]
        aw = [a for _, a in woba_pairs]
        m["woba_corr"]     = _rnd(_pearson(pw, aw), 4)
        m["woba_spearman"] = _rnd(_spearman(pw, aw), 4)
        m["woba_mae"]      = _rnd(sum(abs(p - a) for p, a in woba_pairs) / len(woba_pairs), 4)

    if edge_pairs:
        m["edge_spearman"] = _rnd(_spearman(
            [e for e, _ in edge_pairs], [w for _, w in edge_pairs]), 4)

    return m


# ---------------------------------------------------------------------------
# Per-date run
# ---------------------------------------------------------------------------

def run_date(date_str: str, log_fn: Callable[[str], None] = print) -> Dict:
    """Run projections + fetch actuals for one past date. Cached on disk."""
    _ensure_dir()
    cache = _DATA_DIR / f"{date_str}.json"
    if cache.exists():
        log_fn(f"[backtest] {date_str}: loading cached results")
        with open(cache) as f:
            return json.load(f)

    log_fn(f"[backtest] {date_str}: running analyze_slate ...")
    analysis = _mt.analyze_slate(date_str, log_fn=log_fn)

    log_fn(f"[backtest] {date_str}: fetching actual boxscores ...")
    try:
        games = statsapi.schedule(date=date_str) or []
    except Exception as e:
        log_fn(f"[backtest] {date_str}: schedule fetch failed: {e}")
        games = []

    actual_lines: Dict[int, Dict] = {}
    for g in games:
        status = (g.get("status") or "").lower()
        if "final" not in status and "completed" not in status:
            continue
        gpk = g.get("game_id")
        if not gpk:
            continue
        actual_lines.update(fetch_actual_batting_lines(gpk))

    log_fn(f"[backtest] {date_str}: actual lines for {len(actual_lines)} players")

    records = []
    for b in analysis.get("batters", []):
        pid    = b.get("batter_id")
        actual = actual_lines.get(pid) if pid else None
        adk    = round(_actual_dk(actual), 2)   if actual else None
        awoba  = _rnd(_actual_woba(actual), 4)  if actual else None

        records.append({
            "date":         date_str,
            "name":         b.get("name"),
            "batter_id":    pid,
            "team":         b.get("team"),
            "order":        b.get("order"),
            "vs_pitcher":   b.get("vs_pitcher"),
            "proj_dk":      b.get("dk_points"),
            "actual_dk":    adk,
            "proj_woba":    _rnd(b.get("proj_woba"), 4),
            "actual_woba":  awoba,
            "woba_edge":    _rnd(b.get("woba_edge"), 4),
            "hh_heat":      _rnd(b.get("hh_heat"), 4),
            "matched":      actual is not None,
        })

    matched = [r for r in records if r["matched"]]
    metrics = compute_metrics(matched)

    result = {
        "date":        date_str,
        "n_projected": len(records),
        "n_matched":   len(matched),
        "records":     records,
        "metrics":     metrics,
    }
    with open(cache, "w") as f:
        json.dump(result, f)
    log_fn(f"[backtest] {date_str}: done. {len(matched)}/{len(records)} matched")
    return result


# ---------------------------------------------------------------------------
# Multi-date run
# ---------------------------------------------------------------------------

def run_range(start_date: str, end_date: str,
              log_fn: Callable[[str], None] = print) -> Dict:
    """Run backtest over a date range. Automatically skips today/future."""
    today = _dt.date.today()
    start = _dt.date.fromisoformat(start_date)
    end   = min(_dt.date.fromisoformat(end_date), today - _dt.timedelta(days=1))

    by_date: List[Dict] = []
    all_records: List[Dict] = []
    d = start
    while d <= end:
        ds = d.isoformat()
        try:
            r = run_date(ds, log_fn=log_fn)
            by_date.append({
                "date":        ds,
                "n_projected": r["n_projected"],
                "n_matched":   r["n_matched"],
                "metrics":     r.get("metrics", {}),
            })
            all_records.extend([x for x in r["records"] if x["matched"]])
        except Exception as e:
            log_fn(f"[backtest] {ds}: ERROR {e}")
            by_date.append({"date": ds, "error": str(e)})
        d += _dt.timedelta(days=1)

    return {
        "start":     start_date,
        "end":       end.isoformat(),
        "n_dates":   len(by_date),
        "n_records": len(all_records),
        "metrics":   compute_metrics(all_records),
        "by_date":   by_date,
    }


# ---------------------------------------------------------------------------
# Load all saved results from disk
# ---------------------------------------------------------------------------

def load_summary() -> Dict:
    """Aggregate all cached per-date backtest results from disk."""
    if not _DATA_DIR.exists():
        return {"n_records": 0, "n_dates": 0, "by_date": [], "metrics": {}}

    all_records: List[Dict] = []
    by_date: List[Dict] = []
    for path in sorted(_DATA_DIR.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            matched = [r for r in data.get("records", []) if r.get("matched")]
            all_records.extend(matched)
            m = compute_metrics(matched) if matched else {}
            by_date.append({
                "date":      data.get("date", path.stem),
                "n":         len(matched),
                "dk_corr":   m.get("dk_corr"),
                "dk_mae":    m.get("dk_mae"),
                "woba_corr": m.get("woba_corr"),
            })
        except Exception:
            pass

    return {
        "n_records": len(all_records),
        "n_dates":   len(by_date),
        "metrics":   compute_metrics(all_records),
        "by_date":   by_date,
    }
