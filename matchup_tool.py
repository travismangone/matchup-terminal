"""
matchup_tool.py
---------------
Daily MLB pitch-type matchup engine.

Pulls:
  - Probable starting pitchers and lineups for a given date (MLB Stats API)
  - Each starter's last 30 days of Statcast pitches, split by batter handedness
  - Each batter's last two seasons of Statcast results vs RHP / LHP, grouped by pitch type

Computes:
  - Pitcher arsenal usage by batter side (e.g. FF 42%, SL 28% vs RHB)
  - Pitcher wOBA / ISO allowed vs LHB and vs RHB
  - Batter wOBA / ISO by pitch type (Bayesian-shrunk toward league average)
  - Projected wOBA / ISO for today's matchup as the usage-weighted average of
    the batter's pitch-type rates against today's pitcher's arsenal-by-side

Public API:
  analyze_slate(date_str, log_fn=print) -> dict
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
import os
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import statsapi
from pybaseball import statcast_pitcher, statcast_batter, cache as pyb_cache

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Pitch types we care about. Others are bucketed into "OTHER".
PITCH_TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "KC", "CH", "FS", "SV"]

# League-average wOBA per pitch type (rough 2024 figures). Used as the prior
# for Bayesian shrinkage on small samples.
LEAGUE_WOBA_BY_PITCH = {
    "FF": 0.345, "SI": 0.340, "FC": 0.320, "SL": 0.290, "ST": 0.295,
    "CU": 0.290, "KC": 0.285, "CH": 0.305, "FS": 0.295, "SV": 0.295,
    "OTHER": 0.315,
}
LEAGUE_ISO_BY_PITCH = {
    "FF": 0.175, "SI": 0.150, "FC": 0.150, "SL": 0.140, "ST": 0.140,
    "CU": 0.135, "KC": 0.130, "CH": 0.155, "FS": 0.140, "SV": 0.140,
    "OTHER": 0.150,
}

# Bayesian shrinkage: pretend every batter starts with this many "league average"
# plate appearances against each pitch type. Higher => more conservative.
SHRINKAGE_PA = 75

# How many days of recent Statcast to use for the pitcher's current arsenal.
PITCHER_ARSENAL_LOOKBACK_DAYS = 30

# How many seasons of batter history to pull (current + previous).
BATTER_LOOKBACK_SEASONS = 2

logger = logging.getLogger(__name__)

# Enable pybaseball's on-disk cache so repeated runs are fast.
try:
    pyb_cache.enable()
except Exception:  # pragma: no cover
    pass


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _today_iso() -> str:
    return _dt.date.today().isoformat()


def _bucket_pitch(pt: Optional[str]) -> str:
    if pt is None or (isinstance(pt, float) and math.isnan(pt)):
        return "OTHER"
    pt = str(pt).upper()
    return pt if pt in PITCH_TYPES else "OTHER"


def _safe_log(log_fn: Callable[[str], None], msg: str) -> None:
    try:
        log_fn(msg)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Schedule & lineups (MLB Stats API)
# -----------------------------------------------------------------------------

def get_slate(date_str: str, log_fn: Callable[[str], None] = print) -> List[Dict]:
    """Return a list of games for the date, each with probable pitchers and lineups."""
    _safe_log(log_fn, f"Fetching schedule for {date_str} ...")
    games = statsapi.schedule(date=date_str)
    _safe_log(log_fn, f"  Found {len(games)} game(s).")

    slate: List[Dict] = []
    for g in games:
        game_pk = g.get("game_id")
        away_team = g.get("away_name")
        home_team = g.get("home_name")
        away_pitcher_id = g.get("away_probable_pitcher_id") or g.get("away_pitcher_id")
        home_pitcher_id = g.get("home_probable_pitcher_id") or g.get("home_pitcher_id")
        away_pitcher_name = g.get("away_probable_pitcher") or g.get("away_pitcher_name")
        home_pitcher_name = g.get("home_probable_pitcher") or g.get("home_pitcher_name")

        away_lineup = _get_lineup(game_pk, side="away", log_fn=log_fn)
        home_lineup = _get_lineup(game_pk, side="home", log_fn=log_fn)

        slate.append({
            "game_pk": game_pk,
            "away_team": away_team,
            "home_team": home_team,
            "away_pitcher_id": away_pitcher_id,
            "away_pitcher_name": away_pitcher_name,
            "home_pitcher_id": home_pitcher_id,
            "home_pitcher_name": home_pitcher_name,
            "away_lineup": away_lineup,
            "home_lineup": home_lineup,
        })
    return slate


def _get_lineup(game_pk: int, side: str, log_fn: Callable[[str], None]) -> List[Dict]:
    """Pull the posted lineup for one side of a game. Returns [] if not yet posted.

    statsapi.boxscore_data() returns the batting order in different shapes
    depending on game state:
      - Live / completed: team_block["battingOrder"] is a list of player IDs.
      - Pre-game / some responses: each player record in team_block["players"]
        has a "battingOrder" string like "100", "200", ..., "900" (the first
        digit is the spot; trailing digits are sub-positions for replacements).
    We try the explicit list first, then fall back to the per-player field.
    """
    try:
        box = statsapi.boxscore_data(game_pk)
    except Exception as e:
        _safe_log(log_fn, f"  boxscore fetch failed for game {game_pk}: {e}")
        return []

    team_block = box.get(side, {}) or {}
    players_block = team_block.get("players", {}) or {}
    batting_order = team_block.get("battingOrder", []) or []

    def _player_entry(idx: int, pid_or_key) -> Optional[Dict]:
        pid_str = str(pid_or_key)
        key = pid_str if pid_str.startswith("ID") else f"ID{pid_str}"
        info = players_block.get(key, {}) or {}
        person = info.get("person", {}) or {}
        bat_side = (info.get("batSide", {}) or {}).get("code") or "R"
        try:
            player_id = int(person.get("id")) if person.get("id") else int(pid_str.replace("ID", ""))
        except (TypeError, ValueError):
            return None
        return {
            "order": idx,
            "player_id": player_id,
            "name": person.get("fullName", info.get("name", "Unknown")),
            "bat_side": bat_side,  # "L", "R", or "S"
        }

    # Path A: explicit battingOrder list.
    lineup: List[Dict] = []
    if batting_order:
        for idx, pid in enumerate(batting_order, start=1):
            entry = _player_entry(idx, pid)
            if entry:
                lineup.append(entry)
        if lineup:
            return lineup

    # Path B: per-player battingOrder field. Starters are "100","200",..."900";
    # replacements have "101","201",etc. Keep the lowest sub-position per spot.
    by_spot: Dict[int, Dict] = {}
    for key, info in players_block.items():
        bo = info.get("battingOrder")
        if not bo:
            continue
        try:
            bo_int = int(bo)
        except (TypeError, ValueError):
            continue
        spot = bo_int // 100
        sub = bo_int % 100
        if spot < 1 or spot > 9:
            continue
        existing = by_spot.get(spot)
        if existing is None or sub < existing["_sub"]:
            entry = _player_entry(spot, key)
            if entry:
                entry["_sub"] = sub
                by_spot[spot] = entry

    for spot in sorted(by_spot.keys()):
        e = by_spot[spot]
        e.pop("_sub", None)
        lineup.append(e)

    if not lineup:
        _safe_log(log_fn, f"  lineup empty for game {game_pk} side {side}: batting_order_len={len(batting_order)} players_block_size={len(players_block)}")
    return lineup


# -----------------------------------------------------------------------------
# Pitcher arsenal (Statcast)
# -----------------------------------------------------------------------------

def get_pitcher_profile(
    pitcher_id: int,
    end_date: str,
    log_fn: Callable[[str], None] = print,
) -> Dict:
    """Last 30 days of Statcast for this pitcher, summarized.

    Returns:
      {
        "arsenal": {"R": {"FF": 0.42, ...}, "L": {...}},
        "splits": {"R": {"woba": 0.330, "iso": 0.160}, "L": {...}},
        "n_pitches": int,
      }
    """
    end = _dt.date.fromisoformat(end_date)
    start = end - _dt.timedelta(days=PITCHER_ARSENAL_LOOKBACK_DAYS)
    _safe_log(log_fn, f"  Pulling pitcher {pitcher_id} Statcast {start} -> {end} ...")

    try:
        df = statcast_pitcher(start.isoformat(), end.isoformat(), pitcher_id)
    except Exception as e:
        _safe_log(log_fn, f"    pitcher fetch failed: {e}")
        return _empty_pitcher_profile()

    if df is None or df.empty:
        return _empty_pitcher_profile()

    df = df.copy()
    df["pt"] = df["pitch_type"].map(_bucket_pitch)
    df["stand"] = df["stand"].fillna("R").str.upper().str[0]
    df.loc[~df["stand"].isin(["L", "R"]), "stand"] = "R"

    arsenal: Dict[str, Dict[str, float]] = {"R": {}, "L": {}}
    for side in ("R", "L"):
        sub = df[df["stand"] == side]
        if sub.empty:
            continue
        counts = sub["pt"].value_counts(normalize=True)
        arsenal[side] = {pt: float(counts.get(pt, 0.0)) for pt in PITCH_TYPES + ["OTHER"]}

    splits: Dict[str, Dict[str, float]] = {"R": {}, "L": {}}
    # Use rows where an at-bat resolved (events present) for wOBA/ISO.
    if "events" in df.columns and "woba_value" in df.columns:
        results = df[df["events"].notna() & (df["events"] != "")]
        for side in ("R", "L"):
            sub = results[results["stand"] == side]
            if sub.empty:
                continue
            woba = float(pd.to_numeric(sub["woba_value"], errors="coerce").mean())
            iso = _iso_from_events(sub["events"])
            splits[side] = {"woba": woba, "iso": iso}

    return {
        "arsenal": arsenal,
        "splits": splits,
        "n_pitches": int(len(df)),
    }


def _empty_pitcher_profile() -> Dict:
    return {"arsenal": {"R": {}, "L": {}}, "splits": {"R": {}, "L": {}}, "n_pitches": 0}


def _iso_from_events(events: pd.Series) -> float:
    """ISO = (2B + 2*3B + 3*HR) / AB. AB excludes walks/HBP/SH/SF/etc."""
    ab_events = {
        "single", "double", "triple", "home_run", "strikeout", "field_out",
        "force_out", "grounded_into_double_play", "double_play", "field_error",
        "fielders_choice", "fielders_choice_out", "triple_play",
        "strikeout_double_play",
    }
    ab = events[events.isin(ab_events)]
    if ab.empty:
        return 0.0
    doubles = (ab == "double").sum()
    triples = (ab == "triple").sum()
    hrs = (ab == "home_run").sum()
    return float((doubles + 2 * triples + 3 * hrs) / len(ab))


# -----------------------------------------------------------------------------
# Batter profile (Statcast)
# -----------------------------------------------------------------------------

def get_batter_profile(
    batter_id: int,
    end_date: str,
    pitcher_throws: str,
    log_fn: Callable[[str], None] = print,
) -> Dict:
    """Two-season Statcast for this batter vs the relevant pitcher hand,
    grouped by pitch type. Returns Bayesian-shrunk wOBA / ISO per pitch type.
    """
    end = _dt.date.fromisoformat(end_date)
    start = _dt.date(end.year - (BATTER_LOOKBACK_SEASONS - 1), 1, 1)
    _safe_log(log_fn, f"    Pulling batter {batter_id} Statcast {start} -> {end} ...")

    try:
        df = statcast_batter(start.isoformat(), end.isoformat(), batter_id)
    except Exception as e:
        _safe_log(log_fn, f"      batter fetch failed: {e}")
        return _empty_batter_profile()

    if df is None or df.empty:
        return _empty_batter_profile()

    df = df.copy()
    df["pt"] = df["pitch_type"].map(_bucket_pitch)
    df["p_throws"] = df["p_throws"].fillna("R").str.upper().str[0]
    df.loc[~df["p_throws"].isin(["L", "R"]), "p_throws"] = "R"

    hand = pitcher_throws.upper()[0] if pitcher_throws else "R"
    df = df[df["p_throws"] == hand]
    if df.empty:
        return _empty_batter_profile()

    # Restrict to at-bat-resolving events for wOBA / ISO.
    if "events" in df.columns:
        results = df[df["events"].notna() & (df["events"] != "")]
    else:
        results = df.iloc[0:0]

    by_pt: Dict[str, Dict[str, float]] = {}
    overall_pa = int(len(results))
    for pt in PITCH_TYPES + ["OTHER"]:
        sub = results[results["pt"] == pt]
        n = int(len(sub))
        if n == 0:
            woba_raw = LEAGUE_WOBA_BY_PITCH[pt]
            iso_raw = LEAGUE_ISO_BY_PITCH[pt]
        else:
            woba_raw = float(pd.to_numeric(sub["woba_value"], errors="coerce").mean())
            iso_raw = _iso_from_events(sub["events"])
            if math.isnan(woba_raw):
                woba_raw = LEAGUE_WOBA_BY_PITCH[pt]

        # Bayesian shrinkage toward league average.
        w = n / (n + SHRINKAGE_PA)
        woba = w * woba_raw + (1 - w) * LEAGUE_WOBA_BY_PITCH[pt]
        iso = w * iso_raw + (1 - w) * LEAGUE_ISO_BY_PITCH[pt]
        by_pt[pt] = {"woba": woba, "iso": iso, "n": n}

    return {"by_pitch": by_pt, "pa": overall_pa, "vs_hand": hand}


def _empty_batter_profile() -> Dict:
    by_pt = {pt: {"woba": LEAGUE_WOBA_BY_PITCH[pt], "iso": LEAGUE_ISO_BY_PITCH[pt], "n": 0}
             for pt in PITCH_TYPES + ["OTHER"]}
    return {"by_pitch": by_pt, "pa": 0, "vs_hand": "R"}


# -----------------------------------------------------------------------------
# Matchup scoring
# -----------------------------------------------------------------------------

def _effective_bat_side(bat_side: str, pitcher_throws: str) -> str:
    """Switch hitters take the opposite side from the pitcher."""
    bat_side = (bat_side or "R").upper()[0]
    if bat_side == "S":
        return "L" if pitcher_throws.upper().startswith("R") else "R"
    return bat_side if bat_side in ("L", "R") else "R"


def score_matchup(
    pitcher_profile: Dict,
    pitcher_throws: str,
    batter_profile: Dict,
    bat_side: str,
) -> Dict:
    """Weighted-average the batter's pitch-type wOBA/ISO by the pitcher's
    arsenal usage to that batter's side."""
    eff_side = _effective_bat_side(bat_side, pitcher_throws)
    arsenal = (pitcher_profile.get("arsenal", {}) or {}).get(eff_side, {}) or {}

    # If we have no per-side data, fall back to combined usage from both sides.
    if not arsenal:
        combined: Dict[str, float] = defaultdict(float)
        total = 0.0
        for side_arsenal in pitcher_profile.get("arsenal", {}).values():
            for pt, frac in side_arsenal.items():
                combined[pt] += frac
                total += frac
        if total > 0:
            arsenal = {pt: v / total for pt, v in combined.items()}

    by_pt = batter_profile.get("by_pitch", {}) or {}

    proj_woba = 0.0
    proj_iso = 0.0
    lg_woba = 0.0
    lg_iso = 0.0
    weight_sum = 0.0

    for pt, frac in arsenal.items():
        if frac <= 0:
            continue
        b = by_pt.get(pt, {"woba": LEAGUE_WOBA_BY_PITCH.get(pt, 0.315),
                            "iso": LEAGUE_ISO_BY_PITCH.get(pt, 0.150)})
        proj_woba += frac * b["woba"]
        proj_iso += frac * b["iso"]
        lg_woba += frac * LEAGUE_WOBA_BY_PITCH.get(pt, 0.315)
        lg_iso += frac * LEAGUE_ISO_BY_PITCH.get(pt, 0.150)
        weight_sum += frac

    if weight_sum > 0:
        proj_woba /= weight_sum
        proj_iso /= weight_sum
        lg_woba /= weight_sum
        lg_iso /= weight_sum
    else:
        proj_woba = 0.315
        proj_iso = 0.150
        lg_woba = 0.315
        lg_iso = 0.150

    return {
        "proj_woba": proj_woba,
        "proj_iso": proj_iso,
        "woba_edge": proj_woba - lg_woba,
        "iso_edge": proj_iso - lg_iso,
        "effective_side": eff_side,
    }


# -----------------------------------------------------------------------------
# Top-level entry point
# -----------------------------------------------------------------------------

def analyze_slate(date_str: str, log_fn: Callable[[str], None] = print) -> Dict:
    """Run the full pipeline for a single date.

    Returns:
      {
        "date": "2025-05-11",
        "pitchers": [ { ... per starter ... } ],
        "batters":  [ { ... per batter ... } ],
        "skipped":  [ { game, reason } ],
      }
    """
    t0 = time.time()
    _safe_log(log_fn, f"=== analyze_slate({date_str}) ===")

    slate = get_slate(date_str, log_fn=log_fn)

    pitchers_out: List[Dict] = []
    batters_out: List[Dict] = []
    skipped: List[Dict] = []

    # Cache pitcher profiles so each starter is fetched at most once.
    pitcher_cache: Dict[int, Dict] = {}
    pitcher_meta: Dict[int, Dict] = {}

    for g in slate:
        for side, opp_side in (("away", "home"), ("home", "away")):
            pid = g.get(f"{side}_pitcher_id")
            pname = g.get(f"{side}_pitcher_name")
            lineup = g.get(f"{opp_side}_lineup", [])
            opp_team = g.get(f"{opp_side}_team")
            own_team = g.get(f"{side}_team")

            if not pid:
                skipped.append({"game": f"{g['away_team']} @ {g['home_team']}",
                                  "reason": f"no probable pitcher for {own_team}"})
                continue
            if not lineup:
                _safe_log(log_fn, f"  SKIP {opp_team}: no posted lineup (len={len(lineup)})")
                skipped.append({"game": f"{g['away_team']} @ {g['home_team']}",
                                  "reason": f"no posted lineup for {opp_team}"})
                continue

            if pid not in pitcher_cache:
                prof = get_pitcher_profile(pid, date_str, log_fn=log_fn)
                # Infer throwing hand from the most recent pitch row, fallback R.
                throws = _infer_pitcher_throws(pid, date_str)
                pitcher_cache[pid] = prof
                pitcher_meta[pid] = {"name": pname, "team": own_team, "throws": throws,
                                       "opponent": opp_team}
                pitchers_out.append({
                    "pitcher_id": pid,
                    "name": pname,
                    "team": own_team,
                    "opponent": opp_team,
                    "throws": throws,
                    "arsenal": prof["arsenal"],
                    "splits": prof["splits"],
                    "n_pitches": prof["n_pitches"],
                })

            prof = pitcher_cache[pid]
            throws = pitcher_meta[pid]["throws"]

            for b in lineup:
                eff_side = _effective_bat_side(b["bat_side"], throws)
                bprof = get_batter_profile(b["player_id"], date_str, throws, log_fn=log_fn)
                score = score_matchup(prof, throws, bprof, b["bat_side"])
                batters_out.append({
                    "batter_id": b["player_id"],
                    "name": b["name"],
                    "team": opp_team,
                    "order": b["order"],
                    "bat_side": b["bat_side"],
                    "effective_side": eff_side,
                    "vs_pitcher": pname,
                    "pitcher_throws": throws,
                    "proj_woba": score["proj_woba"],
                    "proj_iso": score["proj_iso"],
                    "woba_edge": score["woba_edge"],
                    "iso_edge": score["iso_edge"],
                    "pa_sample": bprof["pa"],
                })

    batters_out.sort(key=lambda r: r["woba_edge"], reverse=True)

    elapsed = time.time() - t0
    _safe_log(log_fn, f"=== done in {elapsed:.1f}s. "
                       f"{len(pitchers_out)} pitchers, {len(batters_out)} batters, "
                       f"{len(skipped)} skipped ===")

    return {
        "date": date_str,
        "pitchers": pitchers_out,
        "batters": batters_out,
        "skipped": skipped,
        "elapsed_sec": elapsed,
    }


def _infer_pitcher_throws(pitcher_id: int, end_date: str) -> str:
    """Best-effort lookup of a pitcher's throwing hand. Falls back to 'R'."""
    try:
        info = statsapi.lookup_player(pitcher_id)
        if info and isinstance(info, list):
            code = (info[0].get("pitchHand", {}) or {}).get("code")
            if code in ("L", "R"):
                return code
    except Exception:
        pass
    return "R"
