"""Pretty terminal output for projections and best plays."""

from __future__ import annotations

from tabulate import tabulate

from .compare import Play
from .clv import Movement, BetGrade
from .odds_math import decimal_to_american
from config import SHARP_BOOKS

_SHARP_LABEL = " + ".join(b.replace("_", " ").title() for b in SHARP_BOOKS)


def print_projections(sim_rows: list[dict], top: int = 20) -> None:
    rows = []
    for r in sim_rows[:top]:
        rows.append([
            r["name"],
            f"{r['win']*100:5.1f}%",
            f"{r['top_5']*100:5.1f}%",
            f"{r['top_10']*100:5.1f}%",
            f"{r['top_20']*100:5.1f}%",
            f"{r['make_cut']*100:5.1f}%",
        ])
    print("\n=== MODEL PROJECTIONS (top {}) ===".format(top))
    print(tabulate(rows, headers=["Player", "Win", "T5", "T10", "T20", "Cut"],
                   tablefmt="simple"))


def print_plays(plays: list[Play], top: int = 25) -> None:
    if not plays:
        print("\n=== BEST PLAYS ===\n(no +EV plays above the edge threshold)")
        return
    rows = []
    for p in plays[:top]:
        am = decimal_to_american(p.offered_decimal)
        rows.append([
            p.player,
            p.market,
            p.source,
            f"{am:+d}",
            f"{p.model_fair*100:4.1f}%",
            f"{p.sharp_fair*100:4.1f}%" if p.sharp_fair is not None else "  —",
            f"{p.blend_fair*100:4.1f}%",
            f"{p.edge*100:+5.1f}%",
            f"{p.kelly*100:4.2f}%",
            ", ".join(p.flags),
        ])
    print(f"\n=== BEST PLAYS (sharp ref: {_SHARP_LABEL}) ===")
    print(tabulate(
        rows,
        headers=["Player", "Mkt", "Book", "Price", "Model", "Sharp",
                 "Blend", "Edge", "Stake", "Notes"],
        tablefmt="simple",
    ))
    print("\nStake = fractional-Kelly % of bankroll. "
          "Edge is vs. the blended fair prob.")


def _am(x) -> str:
    return f"{x:+d}" if x is not None else "  —"


def print_board(fair: dict, best: dict, run: str, top: int = 25) -> None:
    """Sharp no-vig board + best price elsewhere. Pure market view, no model."""
    from .odds_math import prob_to_decimal, expected_value
    print(f"\n=== {_SHARP_LABEL} NO-VIG OPENING BOARD ===")
    print(f"run: {run} | {len(fair)} golfers priced")
    rows = []
    for p in sorted(fair, key=lambda k: fair[k], reverse=True)[:top]:
        fd_am = decimal_to_american(prob_to_decimal(fair[p]))
        b = best.get(p)
        if b:
            edge = expected_value(fair[p], b.decimal_odds)
            rows.append([p, f"{fair[p]*100:4.1f}%", f"{fd_am:+d}",
                         f"{decimal_to_american(b.decimal_odds):+d}", b.source,
                         f"{edge*100:+4.1f}%"])
        else:
            rows.append([p, f"{fair[p]*100:4.1f}%", f"{fd_am:+d}", "—", "—", "—"])
    print(tabulate(rows, headers=["Golfer", "Fair%", "Fair", "Best elsewhere",
                                  "Book", "Edge"], tablefmt="simple"))
    print("\nEdge = best available price vs. the sharp no-vig line. "
          "Prediction-market edges are liquidity-capped — check depth before betting.")


def print_movement(rows: list[Movement], opening: str | None,
                   closing: str | None, top: int = 30) -> None:
    print(f"\n=== LINE MOVEMENT (sharp: {_SHARP_LABEL}, no-vig) ===")
    print(f"open run:  {opening or '—'}")
    print(f"close run: {closing or '—'}")
    if closing == opening:
        print("(only one snapshot so far — run again Wed night to see movement)")
    out = []
    for m in rows[:top]:
        arrow = ""
        if m.delta_prob is not None:
            arrow = "steam ↑" if m.delta_prob > 0 else "drift ↓"
        out.append([
            m.player,
            _am(m.open_american),
            _am(m.close_american),
            f"{m.open_prob*100:4.1f}%" if m.open_prob else "  —",
            f"{m.close_prob*100:4.1f}%" if m.close_prob else "  —",
            f"{m.delta_prob*100:+4.1f}%" if m.delta_prob is not None else "  —",
            arrow,
        ])
    print(tabulate(out, headers=["Player", "Open", "Close", "Open%", "Close%",
                                 "Δ", ""], tablefmt="simple"))


def print_bet_grades(grades: list[BetGrade]) -> None:
    if not grades:
        print("\n=== CLV — no bets logged ===")
        print("Log one with:  python main.py bet --player \"Rory McIlroy\" "
              "--book fanduel --odds +650")
        return
    out = []
    beat = 0
    for g in grades:
        if g.beat_close:
            beat += 1
        out.append([
            g.player, g.market, g.source,
            _am(g.bet_american), _am(g.close_american),
            f"{g.clv*100:+5.1f}%" if g.clv is not None else "  —",
            "beat" if g.beat_close else ("missed" if g.beat_close is not None else "no close"),
        ])
    print("\n=== CLOSING LINE VALUE ===")
    print(tabulate(out, headers=["Player", "Mkt", "Book", "Bet", "Close(fair)",
                                 "CLV", ""], tablefmt="simple"))
    graded = [g for g in grades if g.beat_close is not None]
    if graded:
        print(f"\nBeat the close on {beat}/{len(graded)} graded bets "
              f"({beat/len(graded)*100:.0f}%). Positive CLV is the goal.")


def print_validation(v) -> None:
    print(f"\n=== MODEL vs MARKET ({_SHARP_LABEL} no-vig, {v.n} players) ===")
    print(f"Rank agreement (Spearman):  {v.spearman:+.3f}   "
          f"(1.0 = identical ordering)")
    print(f"Mean abs. prob difference:  {v.mean_abs_diff*100:.2f} pts")
    print(f"Favorite {v.fav_player}: model {v.fav_model*100:.1f}% vs "
          f"market {v.fav_market*100:.1f}%")
    print(f"Dispersion: {v.dispersion_note}")
    if v.resid_pga is not None or v.resid_owgr is not None:
        rp = f"{v.resid_pga*100:+.2f}" if v.resid_pga is not None else "  —"
        ro = f"{v.resid_owgr*100:+.2f}" if v.resid_owgr is not None else "  —"
        print(f"Mean (model−market): real-SG {rp} pts | OWGR-est {ro} pts "
              f"(remaining OWGR-est are obscure qualifiers, still estimated -> sit below market)")

    def _tbl(rows):
        out = []
        for d in rows:
            out.append([d.player, f"{d.model*100:4.1f}%", f"{d.market*100:4.1f}%",
                        f"{d.diff*100:+4.1f}", ", ".join(d.flags)])
        return tabulate(out, headers=["Player", "Model", "Market", "Δpts", "Notes"],
                        tablefmt="simple")

    print("\n-- Model LIKES (model > market) — your edges, verify by hand --")
    print(_tbl(v.likes))
    print("\n-- Model FADES (market > model) — where you're lower than the market --")
    print(_tbl(v.fades))
    print("\nNote: this is market-agreement, not outcome calibration. "
          "True calibration needs a backtest on prior links events.")


def print_backtest(bt) -> None:
    print(f"\n=== BACKTEST {bt.target_year} (skills from {bt.target_year-1}, no lookahead) ===")
    print(f"{bt.n_events} events | {bt.n_player_events} player-events | {bt.coverage}")
    print("\nBrier score (lower = better; vs. always-predict-base-rate):")
    rows = []
    for m, b in bt.brier.items():
        base = bt.base_rate.get(m, 0.0)
        base_brier = base * (1 - base)          # Brier of the base-rate baseline
        skill = 1 - b / base_brier if base_brier else 0.0
        rows.append([m, f"{b:.4f}", f"{base_brier:.4f}", f"{skill*100:+.1f}%"])
    print(tabulate(rows, headers=["Market", "Brier", "Baseline", "Skill vs base"],
                   tablefmt="simple"))
    print("(Skill vs base > 0 = the model beats blindly predicting the base rate.)")

    for m in ("win", "top_10", "make_cut"):
        rel = bt.reliability.get(m)
        if not rel:
            continue
        print(f"\nReliability — {m} (quantile buckets):")
        out = [[f"{mp*100:5.1f}%", f"{mo*100:5.1f}%", n,
                "high" if mp > mo else "low"] for mp, mo, n in rel]
        print(tabulate(out, headers=["Pred", "Observed", "n", "model"],
                       tablefmt="simple"))
    print("\nWell-calibrated = Pred ≈ Observed down each column. Consistently "
          "Pred > Observed = overconfident; Pred < Observed = underconfident.")
