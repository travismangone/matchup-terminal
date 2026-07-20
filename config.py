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
    "name": "3M Open 2026",
    "course": "TPC Twin Cities",
    "location": "Blaine, MN",
    "dates": "2026-07-23..2026-07-26",
    "field_size": 140,
    "rounds": 4,
    "cut_rule": {"top_n": 65, "ties": True},   # standard PGA Tour cut: top 65 and ties
    # The Odds API sport key for this event's outright winner market.
    "odds_sport_key": "golf_pga_championship_winner",
}

# --------------------------------------------------------------------------
# Skill source — where the sim's per-player expected SG/round comes from.
#
#   "datagolf_decomp": DataGolf /preds/player-decompositions final_pred, which
#                      already folds in THEIR data-driven course fit, course
#                      history, timing and age, plus a per-player round SD.
#   "course_fit":      our hand-tuned LINKS_WEIGHTS / FIT_FACTORS model below.
#
# Backtesting the 2026 Open decided this. The hand-tuned links model predicted
# actual results no better than raw season SG (spearman +0.603 vs +0.605), the
# form nudge was actively NEGATIVE (-0.111 vs actual), and DataGolf's own fit
# adjustments span only +/-0.34 strokes (SD 0.13) against a skill spread of
# SD 0.91 — course fit is a ~14% effect, not a thesis worth hand-building.
# course_fit remains the fallback for anyone missing from the decomposition.
SKILL_SOURCE = "datagolf_decomp"

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
    # TPC Twin Cities is parkland target golf, not links — no component re-weight.
    # These stay at neutral 1.0 because the Open backtest showed aggressive
    # re-weighting bought nothing over equal-weight SG. Only move one off 1.0 if
    # there's evidence, not a narrative.
    "sg_app": 1.00,
    "sg_ott": 1.00,
    "sg_arg": 1.00,
    "sg_putt": 1.00,
}

# Extra multiplicative/additive fit factors (all in strokes-per-round space).
FIT_FACTORS = {
    # Fallback-path fit factors, all dialed to ~off. DataGolf's decomposition
    # supplies course fit/history now (SKILL_SOURCE), and the Open backtest
    # showed our hand-built versions of these added nothing.
    "driving_accuracy_weight": 0.0,
    "links_history_bonus_max": 0.0,   # no links history at a parkland course
    "bogey_avoid_weight": 0.0,
    "bogey_avoid_cap": 0.0,
    # 1.0 = trust the equal-weight season SG total outright.
    "datagolf_skill_blend": 1.0,
}

# --------------------------------------------------------------------------
# Recent form (trailing 16 / 24-round SG). We nudge skill by how much a player's
# recent form DEVIATES from their season baseline — hot form lifts, cold dings.
# (Deviation, not level, so it doesn't double-count the season SG already in.)
# --------------------------------------------------------------------------
FORM_FACTORS = {
    # OFF. At the 2026 Open the form nudge correlated -0.111 with actual SG —
    # it actively hurt. DataGolf's baseline already weights recency. Re-enable
    # only if validate_rounds shows form earning its keep across events.
    "weight": 0.0,
    "cap": 0.0,
    "w16": 0.60,
    "w24": 0.40,
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
    "wind_factor": 1.00,
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
EVENT_LOCATION = {"lat": 45.1608, "lon": -93.2347, "name": "TPC Twin Cities"}
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
