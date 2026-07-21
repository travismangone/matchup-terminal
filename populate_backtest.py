"""
populate_backtest.py
--------------------
Run locally to pre-populate data/backtest/<date>.json for a date range.
These files are committed to git so Render serves historical backtest results
without ever re-pulling Statcast data.

Usage:
  python3 populate_backtest.py                     # last 14 days
  python3 populate_backtest.py 2026-06-01          # from date to yesterday
  python3 populate_backtest.py 2026-06-01 2026-07-01  # specific range

Each new date takes ~5-10 min (Statcast pull). Already-cached dates are instant.
Results go to data/backtest/<date>.json.

After running, commit with:
  git add data/backtest/ && git commit -m "Add historical backtest cache" && git push
"""

import datetime
import sys
import time

import mlb_backtest


def main():
    today = datetime.date.today()

    if len(sys.argv) == 3:
        start = datetime.date.fromisoformat(sys.argv[1])
        end   = datetime.date.fromisoformat(sys.argv[2])
    elif len(sys.argv) == 2:
        start = datetime.date.fromisoformat(sys.argv[1])
        end   = today - datetime.timedelta(days=1)
    else:
        end   = today - datetime.timedelta(days=1)
        start = end - datetime.timedelta(days=13)   # 14 days default

    print(f"Backtest range: {start} → {end} ({(end - start).days + 1} days)")
    print("Cached dates load instantly. New dates take ~5-10 min each.\n")

    t0        = time.time()
    total     = (end - start).days + 1
    done      = 0
    skipped   = 0
    errors    = []

    d = start
    while d <= end:
        ds = d.isoformat()
        cache = mlb_backtest._DATA_DIR / f"{ds}.json"
        if cache.exists():
            print(f"  [{done+1}/{total}] {ds} — cached (skipping)")
            skipped += 1
            done    += 1
            d += datetime.timedelta(days=1)
            continue

        print(f"  [{done+1}/{total}] {ds} — pulling Statcast ...", flush=True)
        t1 = time.time()
        try:
            r = mlb_backtest.run_date(ds, log_fn=lambda msg: print(f"    {msg}", flush=True))
            elapsed = time.time() - t1
            matched  = r.get("n_matched", 0)
            m        = r.get("metrics", {})
            dk_corr  = m.get("dk_corr")
            print(f"    done in {elapsed:.0f}s — {matched} matched  dk_corr={dk_corr}")
        except Exception as e:
            print(f"    ERROR: {e}")
            errors.append((ds, str(e)))

        done += 1
        d += datetime.timedelta(days=1)

    elapsed_total = time.time() - t0
    print(f"\nFinished {done} dates in {elapsed_total/60:.1f} min  ({skipped} cached, {len(errors)} errors)")
    if errors:
        print("Errors:")
        for ds, msg in errors:
            print(f"  {ds}: {msg}")

    # Summarise aggregate metrics
    summary = mlb_backtest.load_summary()
    if summary["n_records"]:
        m = summary["metrics"]
        print(f"\nAggregate across {summary['n_records']} batter-days / {summary['n_dates']} dates:")
        print(f"  DK correlation (r):  {m.get('dk_corr')}")
        print(f"  DK rank (Spearman):  {m.get('dk_spearman')}")
        print(f"  DK MAE:              {m.get('dk_mae')} pts")
        print(f"  wOBA correlation:    {m.get('woba_corr')}")
        print(f"  Edge rank (Spearman): {m.get('edge_spearman')}")

    print("\nTo commit to git and deploy to Render:")
    print("  git add data/backtest/ && git commit -m 'Add historical backtest cache' && git push")


if __name__ == "__main__":
    main()
