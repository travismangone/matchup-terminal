"""
Event + model configuration for the 2026 Open Championship.

Everything course- and event-specific lives here so the engine code stays
generic. Swap this block to point the same pipeline at another major.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# The event
# --------------------------------------------------------------------------
EVENT = {
    "name": "The Open Championship 2026",
    "course": "Royal Birkdale",
    "location": "Southport, England",
    "dates": "2026-07-16..2026-07-19",
    "field_size": 156,
    "rounds": 4,
    "cut_rule": {"top_n": 70, "ties": True},   # Open cut: top 70 and ties after R2
    # The Odds API sport key for this event's outright winner market.
    "odds_sport_key": "golf_the_open_championship_winner",
}

# --------------------------------------------------------------------------
# Course-fit / links weighting  —  ROYAL BIRKDALE thesis
#
# Default strokes-gained total weights every component equally (ott+app+arg+putt).
# We re-weight toward the skills that travel to *this* course. Weights are
# RELATIVE emphasis vs. the equal-weight baseline of 1.0 (>1 matters more here).
#
# Birkdale specifically: unlike the quirkiest links (Carnoustie, St Andrews),
# Birkdale has flatter lies, fairer bounces, and fairways framed by dunes that
# funnel play — so it rewards pure ball-striking more and rewards luck/scramble
# chaos less than the average Open venue. Its champions are elite iron players
# (Spieth '17, Harrington '08, O'Meara '98, Watson '83). Coastal wind + penal pot
# bunkers still make approach control and driving position the separators.
# --------------------------------------------------------------------------
LINKS_WEIGHTS = {
    "sg_app": 1.45,    # approach: THE separator in wind — iron control wins Opens
    "sg_ott": 1.15,    # off-the-tee: position avoids pot bunkers; distance still carries them
    "sg_arg": 1.18,    # around-the-green: links scrambling matters, but SG:ARG is a
                       # noisy low-sample stat — kept above neutral, not amplified

    "sg_putt": 0.60,   # putting: bumpy, wind-affected links greens = noise, downweight hard
}

# Extra multiplicative/additive fit factors (all in strokes-per-round space).
FIT_FACTORS = {
    # Small accuracy nudge on top of SG:OTT (which already rewards driving), so
    # kept low to avoid double-counting — just enough to reward links positioning.
    "driving_accuracy_weight": 0.08,
    # Bonus for players with strong prior links / Open form (see course_fit).
    # Now backed by real data (src/links.py: Open + Scottish Open finish history),
    # not a neutral placeholder. Range is ±(this/2) SG/round at the extremes;
    # 0.40 -> ±0.20, in line with how much course fit actually moves the needle
    # at a major. Raise to lean harder on links pedigree.
    "links_history_bonus_max": 0.35,
    # Links-specific FLOOR bonus: links punishes big numbers (pot bunkers, fescue,
    # wind) more than a normal course, so bogey avoidance is worth extra here on
    # top of what raw SG implies. Bonus = weight · z(bogey avoidance), capped.
    # Deliberate lean most models don't make. per SD of avoidance:
    "bogey_avoid_weight": 0.10,
    "bogey_avoid_cap": 0.20,
    # Weight on the equal-weight SG total vs. the links-reweighted components.
    # LOWER = more links lean. 0.18 -> 82% weight on the links re-weight. This is
    # deliberately MORE aggressive than industry course-fit (which is light-touch)
    # — the model's thesis is that Birkdale rewards the links skill profile harder
    # than the market/consensus prices it.
    "datagolf_skill_blend": 0.18,
}

# --------------------------------------------------------------------------
# Recent form (trailing 16 / 24-round SG). We nudge skill by how much a player's
# recent form DEVIATES from their season baseline — hot form lifts, cold dings.
# (Deviation, not level, so it doesn't double-count the season SG already in.)
# --------------------------------------------------------------------------
FORM_FACTORS = {
    "weight": 0.30,    # fraction of the (form − season) deviation applied to skill.
                       # Trimmed from 0.40: DataGolf skill ratings already weight
                       # recency, so this avoids double-counting and lets the links
                       # signal carry more of the adjustment at this event.
    "cap": 0.50,       # max ± SG/round the form nudge can move a player
    "w16": 0.60,       # blend weight on the 16-round window (sharper, noisier)
    "w24": 0.40,       # blend weight on the 24-round window (steadier)
}

# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
SIM = {
    "n_sims": 20000,
    # Per-round scoring SD in strokes (single-round SG dispersion). Links + wind
    # widen this vs. a calm parkland week. Effective sigma = round_sigma ×
    # wind_factor. Calibration (`main.py validate`) showed effective ~3.3 keeps
    # the favorite's win% near the market without collapsing top-5 dispersion;
    # 3.0 × 1.10 = 3.30, at the realistic high end of single-round SG SD.
    "round_sigma": 3.0,
    # Wind multiplier on round_sigma — links wind genuinely widens round-to-round
    # scoring variance, which compresses the favorites toward the field. 1.20 ->
    # effective sigma 3.60 for Birkdale in wind. Bump higher for a nastier forecast.
    "wind_factor": 1.20,
    "seed": 7,
}

# Placement markets to price off the simulation.
PLACEMENT_MARKETS = ["win", "top_5", "top_10", "top_20", "make_cut"]

# --------------------------------------------------------------------------
# DraftKings (DFS)
# --------------------------------------------------------------------------
# Slate salaries CSV (copied from the DK export). Repoint per slate.
DK_SALARIES_PATH = "data/dk_salaries.csv"

# DFS projections/ownership export (has Large/Small Field ownership per golfer).
# Repoint per slate; drop the new CSV in and the ownership column updates.
DK_PROJECTIONS_PATH = "data/dk_projections.csv"

# DK golf finish-placement points by finishing position (1-indexed). This is the
# placement component of DK scoring — the part the sim can derive. Hole-by-hole
# birdie/streak/bonus points aren't modeled (no hole-level sim), so projected
# points are placement-only and best used for RELATIVE value ranking.
DK_PLACEMENT_POINTS = (
    [30, 20, 18, 16, 14, 12, 10, 9, 8, 7]   # 1st..10th
    + [6] * 5                                # T11-15
    + [5] * 5                                # T16-20
    + [4] * 5                                # T21-25
    + [3] * 5                                # T26-30
    + [2] * 10                               # T31-40
    + [1] * 10                               # T41-50
)

# Hole-scoring component (birdies/pars/bogeys + streak/bogey-free bonuses). We
# can't sim hole-by-hole, so we map each simulated round's SG:Total to expected
# DK scoring points: pts/round = base + slope·(round SG), clamped. Only rounds
# actually played count (2 if the sim misses the cut, 4 if it advances).
# Anchored to two points — elite +2.7-SG ≈ 21/rd, a replacement −2.8-SG ≈ 2.5/rd
# — so bad players don't get propped up (the old base=14/floor=2 handed a club
# pro ~18 pts for shooting +5). A low floor lets ugly rounds score near zero.
# Preserves the ~0.77 correlation with DK's own AvgPointsPerGame.
DK_SCORING = {"base": 11.5, "slope": 3.6, "floor": -3.0, "cap": 30.0}

# --------------------------------------------------------------------------
# Round 4 showdown — finishing position is scored on top of the round's hole
# scoring, so an R4 projection must sim the tournament to the finish (each
# player's 54-hole score + a simulated R4) and award DK's R4 placement points.
# --------------------------------------------------------------------------
# DK R4 showdown finishing-position table (ties get the better position's points,
# not averaged — handled in the sim). 51st+ = 0.
DK_R4_PLACEMENT = (
    [13, 10, 9, 8.5, 8, 7.5, 7, 6.5, 6, 5.5]   # 1st..10th
    + [5] * 5                                   # 11-15
    + [4] * 5                                   # 16-20
    + [3] * 5                                   # 21-25
    + [2] * 5                                   # 26-30
    + [1] * 10                                  # 31-40
    + [0.5] * 10                                # 41-50
)
# Hole scoring on the TRUE showdown per-hole scale (birdie +5.75, par +1.5,
# bogey -1.8, eagle +11, double -3.9, + bogey-free / 3-birdie-streak bonuses),
# so it lands on the same DK-point scale as placement and the two add correctly.
# A field-average links round ≈ +30; slope lifted to ~4.5/SG to absorb the
# bonus convexity that favors low rounds. (The compressed DK_SCORING above is a
# relative-ranking model for R1-R3 single-round slates and is left untouched.)
DK_R4_SCORING = {"base": 30.0, "slope": 4.5, "floor": 12.0, "cap": 58.0}

# --------------------------------------------------------------------------
# Draw (AM/PM wave) modeling — the biggest live edge at a links major.
# At The Open the wind often differs sharply between the early and late waves;
# whichever wave plays the calmer window has a real scoring advantage the skill
# model can't see. We pull the upcoming round's tee-time waves (DataGolf
# field-updates) + an hourly wind forecast (Open-Meteo, no key) and convert the
# wind gap between waves into an SG nudge on the next-round projection.
EVENT_LOCATION = {"lat": 53.6236, "lon": -3.0327, "name": "Royal Birkdale"}
DRAW = {
    "sg_per_mph": 0.055,   # relative SG penalty per mph of wind above the field-mean wave
    "cap": 0.65,           # max |draw adjustment| in SG (keeps a wild forecast from dominating)
    "round_hours": 4.6,    # tee -> finish, used to bound each wave's on-course wind window
    "gust_weight": 0.35,   # blend sustained wind with this much gust (gusts punish links scoring)
}

# --------------------------------------------------------------------------
# Odds comparison
# --------------------------------------------------------------------------
# Sharp reference: de-vig each independently, then average -> the market's best
# truth estimate. Now that odds come from DataGolf (which carries Pinnacle for
# golf, unlike The Odds API), we use the classic two-book sharp: FanDuel + Pinnacle.
SHARP_BOOKS = ["fanduel", "pinnacle"]

# Prediction-market feeds (compared against the sharp reference).
# Polymarket: event titles that mean "the Open winner".
POLYMARKET_TITLE_CONTAINS = ["open championship", "british open"]
# Kalshi EVENT tickers per market. The Open winner + all placement markets exist
# on Kalshi (one active market per golfer), each under its own event ticker. These
# are the ONLY live source for top-5/10/20/make-cut prices — The Odds API offers
# golf WINNER only, so there are no FanDuel/sportsbook placement lines to compare.
# Find a tournament's tickers from its kalshi.com URLs. Drop a market -> skipped.
KALSHI_EVENTS = {
    "win": "KXPGATOUR-THOC26",
    "top_5": "KXPGATOP5-THOC26",
    "top_10": "KXPGATOP10-THOC26",
    "top_20": "KXPGATOP20-THOC26",
    "make_cut": "KXPGAMAKECUT-THOC26",
}

# How much to trust your simulation vs. the sharp market when blending a final
# "fair" probability. 0.0 = pure market, 1.0 = pure model.
MODEL_BLEND = 0.35

# Only surface plays at or above this EV edge (per $1 staked).
MIN_EDGE = 0.02

# Noise floor: ignore plays below this fair probability. Longshot EV is
# unreliable (devig error + favorite-longshot bias are worst in the tail), and a
# noisy model handing a 40/1 shot 1.7% vs. the market's 0.1% is phantom value,
# not a bet. 1% ≈ +100/1 — only players with a real chance make the board.
MIN_FAIR_PROB = 0.01

# Ignore offered prices longer than this (decimal). Books post +100000/+500000
# "lottery" placeholders on deep longshots that you can't meaningfully bet and
# that generate absurd phantom edges. 200 ≈ +19900 — the ceiling of a real
# outright bet. Raise if you genuinely bet 200/1+ tickets.
MAX_PRICE_DECIMAL = 200.0

# Fractional Kelly multiplier applied to recommended stakes.
KELLY_MULTIPLIER = 0.25
