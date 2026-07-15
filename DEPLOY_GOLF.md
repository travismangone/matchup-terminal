# Deploying the Golf tab

The golf model is integrated as a second tab (`/golf`) alongside the baseball
engine. Everything's committed locally on `main`. Three steps to go live — all
need YOUR credentials, so they're yours to do.

## 1. Push to GitHub (triggers Render auto-deploy)
- **GitHub Desktop:** open the `matchup-terminal` repo → you'll see the commit
  "Add golf Open model as a second tab" → **Push origin**.
- **or CLI** (if you have a token set up): `git push origin main`
- Committed straight to `main`. If you'd rather preview first, push a branch and
  open a PR — Render can build a preview.

## 2. Set environment variables (Render → matchup-terminal → Environment)
| Key | Value | Why |
|-----|-------|-----|
| `DATAGOLF_KEY` | `65603d14fa9e315ebb4f86be90a8` | DataGolf SG (your Scratch Plus key) |
| `ODDS_API_KEY` | *(your Odds API key)* | sportsbook odds |
| `VIEW_ONLY` | `1` | hides the Pull button + 403s the endpoint (viewers can't spend credits) |
| `GOLF_REFRESH_MINUTES` | `60` | auto-refresh odds hourly, credit-guarded |
| `ODDS_DAILY_CREDIT_LIMIT` | `50` | *(optional)* hard daily cap |
| `ODDS_MIN_REMAINING` | `20` | *(optional)* stop when monthly balance is low |

**Never put these in the repo** — `.env` is gitignored on purpose.

## 3. That's it
Render redeploys on the push + env change. Visit your terminal → **⛳ GOLF** tab.

---

## How it behaves (worth knowing)
- **Credits:** only `pull_and_snapshot` spends Odds credits, and it's gated by the
  credit guard. With `VIEW_ONLY=1`, viewers can't trigger it at all — only the
  hourly `GOLF_REFRESH_MINUTES` timer does. ~4 credits/refresh, fully predictable.
- **Free tier caveat:** Render free services spin down after ~15 min idle. The
  auto-refresh thread only runs while the service is awake (i.e. while someone's
  viewing) — which is actually credit-efficient. First load after a spin-down is
  a ~30–60s cold start + model build. A paid always-on instance avoids this and
  refreshes on schedule around the clock.
- **First load has data immediately** — a seed snapshot (`data/lines.jsonl`,
  including your opening line for CLV) is committed, so the board isn't empty
  before the first refresh. Redeploys reset it to the seed; re-commit to advance it.
- **Baseball is untouched.** The golf import is guarded — if golf ever errors,
  the baseball engine keeps working and the golf tab just shows a message.
- **To pull manually** (you, not viewers): run locally, or temporarily set
  `VIEW_ONLY=0`. `cron_snapshot.py` is available for a manual/local refresh too.
