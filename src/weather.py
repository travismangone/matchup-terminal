"""
Wind forecast for the draw (AM/PM wave) model — Open-Meteo, free, no API key.

At a links major the wind an individual golfer plays in depends on exactly when
they tee off, not just AM vs PM: within the early wave, a 6:30 tee can play dead
calm while an 11:40 tee (still "early") gets slammed by the building afternoon
breeze. We pull the hourly wind for the round's date once; the draw model then
averages it over each player's own tee -> finish window.
"""

from __future__ import annotations

import requests

from config import EVENT_LOCATION, DRAW

OPEN_METEO = "https://api.open-meteo.com/v1/forecast"


def fetch_hourly(date_str: str) -> dict | None:
    """
    date_str: 'YYYY-MM-DD' (round's local date). Returns
    {'hours': [float], 'wind': [mph], 'gust': [mph], 'precip': mm_total} or None.
    """
    try:
        r = requests.get(
            OPEN_METEO,
            params={
                "latitude": EVENT_LOCATION["lat"], "longitude": EVENT_LOCATION["lon"],
                "hourly": "wind_speed_10m,wind_gusts_10m,precipitation",
                "start_date": date_str, "end_date": date_str,
                "wind_speed_unit": "mph", "timezone": "auto",
            },
            timeout=20,
        )
        r.raise_for_status()
        h = r.json().get("hourly", {})
    except Exception as e:
        print(f"[warn] weather fetch failed: {e}")
        return None

    times = h.get("time", []) or []
    spd = h.get("wind_speed_10m", []) or []
    if not times or not spd:
        return None
    return {
        "hours": [_hour_of(t) for t in times],
        "wind": spd,
        "gust": h.get("wind_gusts_10m", []) or [],
        "precip": round(sum(h.get("precipitation", []) or []), 2),
    }


def window_wind(hourly: dict, start_hour: float, end_hour: float) -> float | None:
    """Gust-blended mean effective wind (mph) over [start_hour, end_hour]."""
    if not hourly:
        return None
    hours, spd, gst = hourly["hours"], hourly["wind"], hourly["gust"]
    gw = DRAW["gust_weight"]
    s_vals, g_vals = [], []
    for i, hr in enumerate(hours):
        if hr is None or not (start_hour <= hr <= end_hour):
            continue
        if i < len(spd) and spd[i] is not None:
            s_vals.append(spd[i])
        if i < len(gst) and gst[i] is not None:
            g_vals.append(gst[i])
    if not s_vals:
        return None
    sustained = sum(s_vals) / len(s_vals)
    gust = (sum(g_vals) / len(g_vals)) if g_vals else sustained
    return (1 - gw) * sustained + gw * gust


def _hour_of(t: str):
    # "2026-07-17T14:00" -> 14.0
    try:
        hm = t.split("T")[1]
        h, m = hm.split(":")[:2]
        return int(h) + int(m) / 60.0
    except (IndexError, ValueError):
        return None
