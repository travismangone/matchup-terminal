"""
Player-name matching.

Sportsbooks, Kalshi, and Polymarket all spell golfers slightly differently
("Rory McIlroy", "McIlroy, Rory", "Rory McILROY", accented names). Everything
that emits a Quote routes the raw name through here so prices for the same
golfer group together and line up with the DataGolf field roster.

Matching is deliberately conservative: exact normalized match first, then
last-name + first-initial. Anything ambiguous returns None (unmatched) rather
than guessing — a wrong match corrupts the devig.
"""

from __future__ import annotations

import unicodedata


def normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation, handle 'Last, First'."""
    name = (name or "").strip()
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
        name = f"{first} {last}"
    # Strip accents.
    name = "".join(
        c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)
    )
    # Fold ligatures / non-decomposing letters that NFKD leaves intact — Nordic
    # names (Højgaard, Olesen, Åberg) are common in the field. Without this, ø
    # survives accent-stripping and the player never matches his DK salary.
    name = (name.replace("ø", "o").replace("Ø", "O")
                .replace("æ", "ae").replace("Æ", "AE")
                .replace("œ", "oe").replace("Œ", "OE")
                .replace("ð", "d").replace("Ð", "D")
                .replace("þ", "th").replace("Þ", "TH")
                .replace("ł", "l").replace("Ł", "L")
                .replace("ß", "ss"))
    # Drop suffixes / punctuation.
    name = name.lower().replace(".", "").replace("-", " ").replace("'", "")
    for suffix in (" jr", " sr", " iii", " ii", " iv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return " ".join(name.split())


def _last_first_initial(norm: str) -> str | None:
    parts = norm.split()
    if len(parts) < 2:
        return None
    return f"{parts[-1]}|{parts[0][0]}"


def build_index(roster: list[str]) -> dict:
    """Precompute lookup structures for a roster of canonical player names."""
    exact: dict[str, str] = {}
    lfi: dict[str, list[str]] = {}
    for name in roster:
        n = normalize_name(name)
        exact[n] = name
        key = _last_first_initial(n)
        if key:
            lfi.setdefault(key, []).append(name)
    return {"exact": exact, "lfi": lfi}


def match_player(raw: str, index: dict) -> str | None:
    """Map a raw market name to a canonical roster name, or None if unsure."""
    n = normalize_name(raw)
    if n in index["exact"]:
        return index["exact"][n]
    key = _last_first_initial(n)
    if key and key in index["lfi"]:
        candidates = index["lfi"][key]
        if len(candidates) == 1:      # only accept if unambiguous
            return candidates[0]
    return None
