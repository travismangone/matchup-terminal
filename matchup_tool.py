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

# FIP / xFIP constants. FIP_CONSTANT roughly scales FIP to league ERA;
# LEAGUE_HR_PER_FB is the league home-run-per-fly-ball rate used by xFIP to
# replace a pitcher's actual HR with an expected HR total (regression signal).
FIP_CONSTANT = 3.10
LEAGUE_HR_PER_FB = 0.12

logger = logging.getLogger(__name__)

# Enable pybaseball's on-disk cache so repeated runs are fast.
try:
    pyb_cache.enable()
except Exception:  # pragma: no cover
    pass

# In-process memo caches (cleared each cold start / process). These avoid
# redundant network round-trips WITHIN a single slate run and across runs in
# the same process, on top of pybaseball's on-disk cache.
_BATTER_DF_CACHE: Dict[tuple, object] = {}
_PROJECT_LINEUP_CACHE: Dict[tuple, List[Dict]] = {}
_THROWS_CACHE: Dict[int, str] = {}
_BAT_SIDE_CACHE: Dict[int, str] = {}


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

# Opener overrides: some teams open with a one-inning pitcher, then bring in
# a bulk/long pitcher. MLB lists the opener as the probable starter, so map
# team name -> {listed opener: pitcher we actually want to analyze}.
OPENER_OVERRIDES: Dict[str, Dict[str, str]] = {
    "Washington Nationals": {"PJ Poulin": "Miles Mikolas"},
}

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
        away_pitcher_name = g.get("away_probable_pitcher") or g.get("away_pitcher_name") or ""
        home_pitcher_name = g.get("home_probable_pitcher") or g.get("home_pitcher_name") or ""
        # Some teams use an "opener" who only faces a batter or two before the
        # bulk pitcher takes over. MLB lists the opener as the probable
        # starter, but we'd rather analyze the bulk pitcher. Swap by team name.
        _away_sub = OPENER_OVERRIDES.get(away_team, {})
        if away_pitcher_name in _away_sub:
            away_pitcher_name = _away_sub[away_pitcher_name]
        _home_sub = OPENER_OVERRIDES.get(home_team, {})
        if home_pitcher_name in _home_sub:
            home_pitcher_name = _home_sub[home_pitcher_name]

        # statsapi.schedule() does not include probable pitcher IDs. Resolve
        # them from the boxscore for completed/in-progress games, and from a
        # name lookup for future-dated games where no boxscore exists yet.
        away_pitcher_id, home_pitcher_id = _resolve_pitcher_ids(
            game_pk, away_pitcher_name, home_pitcher_name, log_fn
        )

        # If we swapped in an opener override, the boxscore-based IDs above
        # still point at the opener (boxscore pitchers[0]). Re-resolve the ID
        # by name so the profile, arsenal, and handedness follow that pitcher.
        if away_pitcher_name in _away_sub.values():
            _ov_id = _lookup_player_id(away_pitcher_name, log_fn)
            if _ov_id:
                away_pitcher_id = _ov_id
        if home_pitcher_name in _home_sub.values():
            _ov_id = _lookup_player_id(home_pitcher_name, log_fn)
            if _ov_id:
                home_pitcher_id = _ov_id

        away_team_id = g.get("away_id")
        home_team_id = g.get("home_id")

        away_lineup = _get_lineup(game_pk, side="away", log_fn=log_fn)
        home_lineup = _get_lineup(game_pk, side="home", log_fn=log_fn)

        # Official lineups are usually not posted until a few hours before
        # first pitch. When a side has no posted lineup yet, fall back to that
        # team's most recent actual lineup so the slate is usable earlier. Each
        # lineup is tagged so the UI can show "official" vs "projected".
        away_status = "official" if away_lineup else "projected"
        if not away_lineup:
            away_lineup = _project_lineup(away_team_id, date_str, log_fn)
        home_status = "official" if home_lineup else "projected"
        if not home_lineup:
            home_lineup = _project_lineup(home_team_id, date_str, log_fn)

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
            "away_lineup_status": away_status,
            "home_lineup_status": home_status,
        })
    return slate


def _project_lineup(team_id, date_str, log_fn):
    """Best-effort projected lineup: the team's most recent actual batting
    order from a completed game before date_str. Used when the official lineup
    for the target date has not been posted yet. Returns [] if none found."""
    if not team_id:
        return []
    _ck = (team_id, date_str)
    if _ck in _PROJECT_LINEUP_CACHE:
        return _PROJECT_LINEUP_CACHE[_ck]
    try:
        target = _dt.date.fromisoformat(date_str)
    except (TypeError, ValueError):
        return []
    start = (target - _dt.timedelta(days=10)).isoformat()
    end = (target - _dt.timedelta(days=1)).isoformat()
    try:
        games = statsapi.schedule(start_date=start, end_date=end, team=team_id)
    except Exception as e:
        _safe_log(log_fn, f"  projected-lineup schedule fetch failed for team {team_id}: {e}")
        return []
    games = sorted(games, key=lambda x: x.get("game_date", ""), reverse=True)
    for gm in games:
        status = (gm.get("status") or "").lower()
        if "final" not in status and "completed" not in status:
            continue
        gpk = gm.get("game_id")
        if not gpk:
            continue
        side = "home" if gm.get("home_id") == team_id else "away"
        lineup = _get_lineup(gpk, side=side, log_fn=log_fn)
        if lineup:
            _safe_log(log_fn, f"  projected lineup for team {team_id} from game {gpk} ({gm.get('game_date')})")
            _PROJECT_LINEUP_CACHE[_ck] = lineup
            return lineup
    _safe_log(log_fn, f"  no projected lineup found for team {team_id}")
    _PROJECT_LINEUP_CACHE[_ck] = []
    return []


def _resolve_pitcher_ids(game_pk: Optional[int], away_name: str, home_name: str,
                         log_fn: Callable[[str], None]) -> tuple:
    """Return (away_pitcher_id, home_pitcher_id) using boxscore first, then name lookup."""
    away_id: Optional[int] = None
    home_id: Optional[int] = None
    if game_pk:
        try:
            box = statsapi.boxscore_data(game_pk)
            away_pitchers = (box.get("away", {}) or {}).get("pitchers", []) or []
            home_pitchers = (box.get("home", {}) or {}).get("pitchers", []) or []
            if away_pitchers:
                away_id = int(away_pitchers[0])
            if home_pitchers:
                home_id = int(home_pitchers[0])
        except Exception as e:
            _safe_log(log_fn, f"  boxscore pitcher-id fetch failed for game {game_pk}: {e}")

    if away_id is None and away_name:
        away_id = _lookup_player_id(away_name, log_fn)
    if home_id is None and home_name:
        home_id = _lookup_player_id(home_name, log_fn)
    return away_id, home_id


def _lookup_player_id(name: str, log_fn: Callable[[str], None]) -> Optional[int]:
    """Best-effort name -> player_id via statsapi.lookup_player()."""
    if not name:
        return None
    try:
        hits = statsapi.lookup_player(name) or []
    except Exception as e:
        _safe_log(log_fn, f"  lookup_player({name!r}) failed: {e}")
        return None
    if not hits:
        return None
    for h in hits:
        if (h.get("fullName") or "").lower() == name.lower():
            try:
                return int(h["id"])
            except (KeyError, TypeError, ValueError):
                continue
    try:
        return int(hits[0]["id"])
    except (KeyError, TypeError, ValueError):
        return None


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
        try:
            _pid_for_side = int(person.get("id")) if person.get("id") else int(pid_str.replace("ID", ""))
        except (TypeError, ValueError):
            _pid_for_side = None
        bat_side = _get_bat_side(_pid_for_side) if _pid_for_side else "R"
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

    def _pull(s, e):
        try:
            return statcast_pitcher(s.isoformat(), e.isoformat(), pitcher_id)
        except Exception as ex:
            _safe_log(log_fn, f"    pitcher fetch failed: {ex}")
            return None

    df = _pull(start, end)

    # Fallback windows when the 30-day pull is empty (e.g., pitcher hasn't
    # appeared in the last month). Try season-to-date, then prior season.
    if df is None or df.empty:
        season_start = _dt.date(end.year, 1, 1)
        if season_start < start:
            _safe_log(log_fn, f"    no 30d data for {pitcher_id}; trying season {season_start} -> {end}")
            df = _pull(season_start, end)

    if df is None or df.empty:
        prev_start = _dt.date(end.year - 1, 1, 1)
        prev_end = _dt.date(end.year - 1, 12, 31)
        _safe_log(log_fn, f"    no current-season data for {pitcher_id}; trying {prev_start} -> {prev_end}")
        df = _pull(prev_start, prev_end)

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

    # ---- FIP / xFIP (regression signal) ----
    fip = None
    xfip = None
    babip = None
    lob_pct = None
    if "events" in df.columns:
        ev = df[df["events"].notna() & (df["events"] != "")]["events"]
        hr = int((ev == "home_run").sum())
        bb = int(ev.isin(["walk"]).sum())
        ibb = int((ev == "intent_walk").sum())
        hbp = int((ev == "hit_by_pitch").sum())
        k = int(ev.isin(["strikeout", "strikeout_double_play"]).sum())
        out_events = {
            "strikeout", "strikeout_double_play", "field_out", "force_out",
            "grounded_into_double_play", "double_play", "fielders_choice_out",
            "sac_fly", "sac_bunt", "other_out", "triple_play",
        }
        outs = int(ev.isin(out_events).sum())
        outs += int((ev == "double_play").sum()) + 2 * int((ev == "triple_play").sum())
        outs += int((ev == "grounded_into_double_play").sum()) + int((ev == "strikeout_double_play").sum())
        ip = outs / 3.0
        fb = 0
        if "bb_type" in df.columns:
            fb = int(df["bb_type"].isin(["fly_ball"]).sum())
        if ip > 0:
            fip = float((13 * hr + 3 * (bb + hbp) - 2 * k) / ip + FIP_CONSTANT)
            exp_hr = fb * LEAGUE_HR_PER_FB
            xfip = float((13 * exp_hr + 3 * (bb + hbp) - 2 * k) / ip + FIP_CONSTANT)
            fip = round(fip, 2)
            xfip = round(xfip, 2)

        # ---- BABIP and LOB% (luck / regression signals) ----
        singles = int((ev == "single").sum())
        doubles = int((ev == "double").sum())
        triples = int((ev == "triple").sum())
        sf = int((ev == "sac_fly").sum())
        hits = singles + doubles + triples + hr
        ab_events_b = {
            "single", "double", "triple", "home_run", "strikeout", 
            "strikeout_double_play", "field_out", "force_out", 
            "grounded_into_double_play", "double_play", "field_error", 
            "fielders_choice", "fielders_choice_out", "triple_play", 
            "other_out", 
        }
        ab = int(ev.isin(ab_events_b).sum())
        bip_denom = ab - k - hr + sf
        if bip_denom > 0:
            babip = round(float((hits - hr) / bip_denom), 3)
        # LOB%: share of baserunners stranded. Runs allowed come straight from
        # the Statcast score columns: on each plate-appearance-ending row,
        # post_bat_score - bat_score is the runs that scored on that play
        # (the batting team is the opponent, so this is runs the pitcher gave up).
        runs_allowed = None
        if {"bat_score", "post_bat_score"}.issubset(df.columns):
            pa_rows = df[df["events"].notna() & (df["events"] != "")]
            bs = pd.to_numeric(pa_rows["bat_score"], errors="coerce")
            ps = pd.to_numeric(pa_rows["post_bat_score"], errors="coerce")
            delta = (ps - bs).clip(lower=0)
            runs_allowed = int(delta.fillna(0).sum())
        reached = hits + bb + ibb + hbp
        lob_denom = reached - 1.4 * hr
        if runs_allowed is not None and lob_denom > 0:
            lob_pct = round(float(100.0 * (reached - runs_allowed) / lob_denom), 1)
            lob_pct = max(0.0, min(100.0, lob_pct))

    # ---- Regression signal (luck-based, NOT a skill grade) ----
    # Compares BABIP vs a league-neutral .300 and LOB% vs ~72%. A pitcher
    # running a LOW BABIP and/or HIGH LOB% has been lucky and is a NEGATIVE
    # regression risk (results likely to worsen); a HIGH BABIP and/or LOW
    # LOB% has been unlucky and is a POSITIVE regression candidate (results
    # likely to improve). reg_score > 0 => expect improvement.
    reg_score = None
    reg_label = None
    if babip is not None or lob_pct is not None:
        _score = 0.0
        if babip is not None:
            _score += (babip - 0.300) / 0.030
        if lob_pct is not None:
            _score += (72.0 - lob_pct) / 8.0
        reg_score = round(float(_score), 2)
        if reg_score >= 0.6:
            reg_label = "Positive"
        elif reg_score <= -0.6:
            reg_label = "Negative"
        else:
            reg_label = "Neutral"

    # ---- Strikeout % and Walk % (30-day window uses the df above) ----
    rates_30d = _k_bb_rates(df)
    # Season-to-date window: dedicated pull so the season column is accurate
    # regardless of any fallback applied to the 30-day df.
    season_df = _pull(_dt.date(end.year, 1, 1), end)
    rates_season = _k_bb_rates(season_df)

    return {
        "arsenal": arsenal,
        "splits": splits,
        "n_pitches": int(len(df)),
        "fip": fip,
        "xfip": xfip,
        "k_pct_30d": rates_30d["k_pct"],
        "bb_pct_30d": rates_30d["bb_pct"],
        "k_pct_season": rates_season["k_pct"],
        "bb_pct_season": rates_season["bb_pct"],
        "babip": babip,
        "lob_pct": lob_pct,
        "reg_score": reg_score,
        "reg_label": reg_label,
    }


def opponent_offense_30d(
    lineup: List[Dict],
    end_date: str,
    log_fn: Callable[[str], None] = print,
) -> Dict[str, Optional[float]]:
    """Aggregate the opposing lineup's last-30-day offense.

    Pulls each hitter's Statcast for the 30 calendar days ending on
    `end_date` (vs all pitchers) and pools every plate appearance together,
    then returns lineup-level rates:

      {"opp_k_pct_30d": <pct>, "opp_iso_30d": <iso>, "opp_woba_30d": <woba>}

    These describe how the offense the pitcher is facing today has actually
    been hitting recently, so the grading engine can reward a pitcher drawing
    a cold/weak lineup and penalize one drawing a hot/strong lineup.
    Returns Nones when no data is available.
    """
    empty = {"opp_k_pct_30d": None, "opp_iso_30d": None, "opp_woba_30d": None}
    if not lineup:
        return empty

    end = _dt.date.fromisoformat(end_date)
    start = end - _dt.timedelta(days=30)
    frames = []
    for b in lineup:
        bid = b.get("player_id")
        if not bid:
            continue
        _bk = (bid, start.isoformat(), end.isoformat())
        df = _BATTER_DF_CACHE.get(_bk)
        if df is None:
            try:
                df = statcast_batter(start.isoformat(), end.isoformat(), bid)
            except Exception as e:
                _safe_log(log_fn, f"      opp batter {bid} fetch failed: {e}")
                df = None
            _BATTER_DF_CACHE[_bk] = df
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return empty

    pooled = pd.concat(frames, ignore_index=True)
    if "events" not in pooled.columns:
        return empty
    results = pooled[pooled["events"].notna()]
    if results.empty:
        return empty

    rates = _k_bb_rates(results)
    out = dict(empty)
    out["opp_k_pct_30d"] = rates.get("k_pct")
    out["opp_iso_30d"] = round(_iso_from_events(results["events"]), 3)
    if "woba_value" in results.columns:
        woba = float(pd.to_numeric(results["woba_value"], errors="coerce").mean())
        out["opp_woba_30d"] = None if math.isnan(woba) else round(woba, 3)
    return out


def _k_bb_rates(df) -> Dict[str, Optional[float]]:
    """Return {"k_pct", "bb_pct"} as fractions of plate appearances.
    A plate appearance is any row with a resolved (non-null) events value."""
    if df is None or df.empty or "events" not in df.columns:
        return {"k_pct": None, "bb_pct": None}
    ev = df[df["events"].notna() & (df["events"] != "")]["events"]
    pa = int(len(ev))
    if pa <= 0:
        return {"k_pct": None, "bb_pct": None}
    k = int(ev.isin(["strikeout", "strikeout_double_play"]).sum())
    bb = int(ev.isin(["walk", "intent_walk"]).sum())
    return {"k_pct": round(100.0 * k / pa, 1), "bb_pct": round(100.0 * bb / pa, 1)}


def _empty_pitcher_profile() -> Dict:
    return {"arsenal": {"R": {}, "L": {}}, "splits": {"R": {}, "L": {}}, "n_pitches": 0, "fip": None, "xfip": None, "k_pct_30d": None, "bb_pct_30d": None, "k_pct_season": None, "bb_pct_season": None, "babip": None, "lob_pct": None}


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
    window: str = "season",
) -> Dict:
    """Statcast for this batter vs the relevant pitcher hand, grouped by pitch type.
    Returns Bayesian-shrunk wOBA / ISO per pitch type.

    `window` controls the lookback:
      - "season": current + previous season (default, BATTER_LOOKBACK_SEASONS).
      - "last30": only the last 30 calendar days ending on end_date.
    """
    end = _dt.date.fromisoformat(end_date)
    if window == "last30":
        start = end - _dt.timedelta(days=30)
    else:
        start = _dt.date(end.year - (BATTER_LOOKBACK_SEASONS - 1), 1, 1)
    _bk = (batter_id, start.isoformat(), end.isoformat())
    df = _BATTER_DF_CACHE.get(_bk)
    if df is None:
        _safe_log(log_fn, f"    Pulling batter {batter_id} Statcast {start} -> {end} ...")
        try:
            df = statcast_batter(start.isoformat(), end.isoformat(), batter_id)
        except Exception as e:
            _safe_log(log_fn, f"      batter fetch failed: {e}")
            return _empty_batter_profile()
        _BATTER_DF_CACHE[_bk] = df
    else:
        _safe_log(log_fn, f"    Batter {batter_id} Statcast (cached {start} -> {end})")

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

def analyze_slate(date_str: str, log_fn: Callable[[str], None] = print, batter_window: str = "season") -> Dict:
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
    _safe_log(log_fn, f"=== analyze_slate({date_str}) batter_window={batter_window} ===")

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
            lineup_status = g.get(f"{opp_side}_lineup_status", "official")
            opp_team = g.get(f"{opp_side}_team")
            own_team = g.get(f"{side}_team")

            if not pid:
                skipped.append({"game": f"{g['away_team']} @ {g['home_team']}",
                                  "reason": f"no probable pitcher for {own_team}"})
                continue
            # Note: a missing lineup no longer skips the pitcher. The starter is
            # always processed below; the batter loop is guarded so we simply
            # show no hitters when neither an official nor a projected lineup exists.
            if not lineup:
                _safe_log(log_fn, f"  {opp_team}: no official or projected lineup; showing pitcher only")

            if pid not in pitcher_cache:
                prof = get_pitcher_profile(pid, date_str, log_fn=log_fn)
                # Infer throwing hand from the most recent pitch row, fallback R.
                throws = _infer_pitcher_throws(pid, date_str)
                pitcher_cache[pid] = prof
                pitcher_meta[pid] = {"name": pname, "team": own_team, "throws": throws,
                                       "opponent": opp_team}
                opp_off = opponent_offense_30d(lineup, date_str, log_fn=log_fn)
                pitchers_out.append({
                    "pitcher_id": pid,
                    "name": pname,
                    "team": own_team,
                    "opponent": opp_team,
                    "throws": throws,
                    "arsenal": prof["arsenal"],
                    "splits": prof["splits"],
                    "n_pitches": prof["n_pitches"],
                    "fip": prof.get("fip"),
                    "xfip": prof.get("xfip"),
                    "k_pct_30d": prof.get("k_pct_30d"),
                    "bb_pct_30d": prof.get("bb_pct_30d"),
                    "k_pct_season": prof.get("k_pct_season"),
                    "bb_pct_season": prof.get("bb_pct_season"),
                    "babip": prof.get("babip"),
                    "lob_pct": prof.get("lob_pct"),
                    "reg_score": prof.get("reg_score"),
                    "reg_label": prof.get("reg_label"),
                    "opp_k_pct_30d": opp_off.get("opp_k_pct_30d"),
                    "opp_iso_30d": opp_off.get("opp_iso_30d"),
                    "opp_woba_30d": opp_off.get("opp_woba_30d"),
                })

            prof = pitcher_cache[pid]
            throws = pitcher_meta[pid]["throws"]

            for b in lineup:
                eff_side = _effective_bat_side(b["bat_side"], throws)
                bprof = get_batter_profile(b["player_id"], date_str, throws, log_fn=log_fn, window=batter_window)
                score = score_matchup(prof, throws, bprof, b["bat_side"])
                batters_out.append({
                    "batter_id": b["player_id"],
                    "name": b["name"],
                    "team": opp_team,
                    "order": b["order"],
                    "lineup_status": lineup_status,
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


def _get_bat_side(player_id: int) -> str:
    """Look up a batter's handedness ("L", "R", or "S") from the MLB Stats
    API people endpoint, cached per player.

    The boxscore player record does NOT carry a batSide field, so the old
    code (info.get("batSide")) always fell through to "R" and every hitter
    rendered as right-handed -- which silently selected the wrong platoon
    split for all lefties and switch hitters. The canonical record lives at
    /api/v1/people/{id} as "batSide.code".
    """
    if not player_id:
        return "R"
    if player_id in _BAT_SIDE_CACHE:
        return _BAT_SIDE_CACHE[player_id]
    try:
        resp = statsapi.get("person", {"personId": int(player_id)})
        people = (resp or {}).get("people") or []
        if people:
            code = (people[0].get("batSide") or {}).get("code")
            if code in ("L", "R", "S"):
                _BAT_SIDE_CACHE[player_id] = code
                return code
    except Exception:
        pass
    return "R"


def _infer_pitcher_throws(pitcher_id: int, end_date: str) -> str:
    """Best-effort lookup of a pitcher's throwing hand. Falls back to 'R'.

    The MLB Stats API `people` endpoint returns `pitchHand.code` ("L" or
    "R") for each player. Earlier versions of this code called
    `statsapi.lookup_player(pitcher_id)`, but `lookup_player` searches by
    name string, not numeric id, so passing an int silently returned no
    matches and we always fell back to "R" (every pitcher rendered as RHP).
    """
    if pitcher_id in _THROWS_CACHE:
        return _THROWS_CACHE[pitcher_id]
    # Primary: hit /api/v1/people/{id} for the canonical record.
    try:
        resp = statsapi.get("person", {"personId": int(pitcher_id)})
        people = (resp or {}).get("people") or []
        if people:
            code = (people[0].get("pitchHand") or {}).get("code")
            if code in ("L", "R"):
                _THROWS_CACHE[pitcher_id] = code
                return code
    except Exception:
        pass
    # Secondary: try the recent Statcast pull's p_throws column as a fallback.
    try:
        end = _dt.date.fromisoformat(end_date)
        start = end - _dt.timedelta(days=PITCHER_ARSENAL_LOOKBACK_DAYS)
        df = statcast_pitcher(start.isoformat(), end.isoformat(), int(pitcher_id))
        if df is not None and not df.empty and "p_throws" in df.columns:
            vals = df["p_throws"].dropna().astype(str).str.upper().str[0]
            vals = vals[vals.isin(["L", "R"])]
            if not vals.empty:
                _r = vals.iloc[-1]
                _THROWS_CACHE[pitcher_id] = _r
                return _r
    except Exception:
        pass
    return "R"
