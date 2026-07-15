"""
Links course-fit weighting.

Turns each player's baseline SG components into a single ADJUSTED SKILL number —
expected strokes gained per round *at Royal Birkdale specifically* — by
re-weighting the components toward the links skill profile and layering on
driving-accuracy and Open-history bonuses.

This adjusted skill is the mean that feeds the Monte Carlo simulation.
"""

from __future__ import annotations

from .datagolf import Player
from config import LINKS_WEIGHTS, FIT_FACTORS, FORM_FACTORS


def skill_breakdown(p: Player) -> dict:
    """
    The additive pieces of a player's expected SG/round at this course. They sum
    to `adjusted` — so you can see exactly what's moving each guy:

        season   = blend·season_total + (1-blend)·links-reweighted components
        driving  = accuracy bonus (straight off the tee gains at links)
        links    = Open/Scottish finish-history bonus (centered on neutral 0.5)
        form     = recent-vs-season deviation nudge (16/24-round SG)
    """
    reweighted = (
        p.sg_ott * LINKS_WEIGHTS["sg_ott"]
        + p.sg_app * LINKS_WEIGHTS["sg_app"]
        + p.sg_arg * LINKS_WEIGHTS["sg_arg"]
        + p.sg_putt * LINKS_WEIGHTS["sg_putt"]
    )
    blend = FIT_FACTORS["datagolf_skill_blend"]
    base = p.sg_total if p.sg_total else (p.sg_ott + p.sg_app + p.sg_arg + p.sg_putt)
    season = blend * base + (1.0 - blend) * reweighted
    driving = p.driving_acc * FIT_FACTORS["driving_accuracy_weight"]
    links = (p.links_history - 0.5) * FIT_FACTORS["links_history_bonus_max"]
    form = _form_adj(p, base)
    bogey = _bogey_adj(p)
    return {
        "season": season, "driving": driving, "links": links, "form": form,
        "bogey": bogey,
        "adjusted": season + driving + links + form + bogey,
    }


def _bogey_adj(p: Player) -> float:
    """Links floor bonus from bogey avoidance (z-scored), capped. 0 when no data."""
    bz = getattr(p, "bogey_z", None)
    if bz is None:
        return 0.0
    cap = FIT_FACTORS.get("bogey_avoid_cap", 0.20)
    return max(-cap, min(cap, FIT_FACTORS.get("bogey_avoid_weight", 0.0) * bz))


def adjusted_skill(p: Player) -> float:
    """Expected SG/round at this course (sum of skill_breakdown)."""
    return skill_breakdown(p)["adjusted"]


def _form_adj(p: Player, season_sg: float) -> float:
    """
    Nudge by recent-vs-season deviation. Blend the 16/24-round windows (whichever
    exist), take (form − season), scale, and cap. Zero when no form data.
    """
    ws, vs = [], []
    if p.form_16 is not None:
        ws.append(FORM_FACTORS["w16"]); vs.append(p.form_16)
    if p.form_24 is not None:
        ws.append(FORM_FACTORS["w24"]); vs.append(p.form_24)
    if not ws:
        return 0.0
    form_blend = sum(w * v for w, v in zip(ws, vs)) / sum(ws)
    adj = FORM_FACTORS["weight"] * (form_blend - season_sg)
    cap = FORM_FACTORS["cap"]
    return max(-cap, min(cap, adj))


def build_skills(players: list[Player]) -> dict[str, float]:
    """Player name -> adjusted SG/round, ready for the simulator."""
    return {p.name: adjusted_skill(p) for p in players}
