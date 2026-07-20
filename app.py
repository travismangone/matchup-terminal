"""
app.py
------
Flask web server for the matchup terminal.

Endpoints:
  GET  /         -> renders the dark terminal UI
  POST /run      -> kicks off a background analysis job for a given date
  GET  /status   -> JSON: current job state (idle | running | done | error)
                    and, when done, the full result payload
  POST /cancel   -> requests cancellation of the running job (best-effort)

Run locally:
  python app.py
Render:
  gunicorn app:app   (see Procfile)
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import threading
import traceback
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template, request

import matchup_tool
import mlb_backtest

# -----------------------------------------------------------------------------
# Golf Open model (second tab). Fully self-contained; the guarded import + the
# per-route try/except mean a golf problem NEVER touches the baseball engine.
# -----------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))  # no-op on Render
except Exception:
    pass

# View-only hides the credit-spending "Pull" button for terminal viewers.
# Secure-by-default for the shared terminal: view-only (pull button hidden +
# blocked) unless explicitly turned off (VIEW_ONLY=0 for local dev).
VIEW_ONLY = os.environ.get("VIEW_ONLY", "1") == "1"

try:
    from src import dashboard as golf_dashboard
    _GOLF_OK = True
except Exception as _golf_err:            # missing deps/keys -> tab degrades gracefully
    golf_dashboard = None
    _GOLF_OK = False

_golf_cache: dict = {}


def _golf_state(rebuild: bool = False) -> dict:
    if rebuild or "state" not in _golf_cache:
        _golf_cache["state"] = golf_dashboard.build_state(False)
    return _golf_cache["state"]


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("matchup-terminal")

app = Flask(__name__)

# -----------------------------------------------------------------------------
# Shared job state
# -----------------------------------------------------------------------------

_state_lock = threading.Lock()
_state: Dict[str, Any] = {
    "status": "idle",        # idle | running | done | error
    "started_at": None,
    "finished_at": None,
    "date": None,
    "log": [],
    "result": None,
    "error": None,
    "cancel_requested": False,
}
_worker_thread: Optional[threading.Thread] = None


def _append_log(line: str) -> None:
    with _state_lock:
        # Keep the log bounded to avoid runaway memory in long runs.
        if len(_state["log"]) >= 2000:
            _state["log"] = _state["log"][-1500:]
        _state["log"].append(line)
    log.info(line)


def _snapshot() -> Dict[str, Any]:
    with _state_lock:
        # Return a shallow copy so the caller can't mutate live state.
        return {
            "status": _state["status"],
            "started_at": _state["started_at"],
            "finished_at": _state["finished_at"],
            "date": _state["date"],
            "log": list(_state["log"]),
            "result": _state["result"],
            "error": _state["error"],
        }


# -----------------------------------------------------------------------------
# Background worker
# -----------------------------------------------------------------------------

def _run_analysis(date_str: str, batter_window: str = "season") -> None:
    try:
        with _state_lock:
            _state["status"] = "running"
            _state["started_at"] = _dt.datetime.utcnow().isoformat() + "Z"
            _state["finished_at"] = None
            _state["date"] = date_str
            _state["log"] = []
            _state["result"] = None
            _state["error"] = None
            _state["cancel_requested"] = False

        result = matchup_tool.analyze_slate(date_str, log_fn=_append_log, batter_window=batter_window)

        with _state_lock:
            _state["status"] = "done"
            _state["finished_at"] = _dt.datetime.utcnow().isoformat() + "Z"
            _state["result"] = result
    except Exception as e:
        tb = traceback.format_exc()
        log.error("analysis failed: %s\n%s", e, tb)
        with _state_lock:
            _state["status"] = "error"
            _state["finished_at"] = _dt.datetime.utcnow().isoformat() + "Z"
            _state["error"] = str(e)
            _state["log"].append(f"ERROR: {e}")


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------------------------------------------------------
# Backtest routes
# -----------------------------------------------------------------------------

_bt_lock  = threading.Lock()
_bt_state: Dict[str, Any] = {
    "status":   "idle",
    "start":    None,
    "end":      None,
    "log":      [],
    "result":   None,
    "error":    None,
}
_bt_thread: Optional[threading.Thread] = None


def _bt_log(line: str) -> None:
    with _bt_lock:
        if len(_bt_state["log"]) >= 2000:
            _bt_state["log"] = _bt_state["log"][-1500:]
        _bt_state["log"].append(line)
    log.info("[bt] %s", line)


def _run_backtest(start_date: str, end_date: str) -> None:
    try:
        with _bt_lock:
            _bt_state.update({"status": "running", "start": start_date,
                               "end": end_date, "log": [], "result": None, "error": None})
        result = mlb_backtest.run_range(start_date, end_date, log_fn=_bt_log)
        with _bt_lock:
            _bt_state["status"] = "done"
            _bt_state["result"] = result
    except Exception as e:
        tb = traceback.format_exc()
        log.error("backtest failed: %s\n%s", e, tb)
        with _bt_lock:
            _bt_state["status"] = "error"
            _bt_state["error"]  = str(e)
            _bt_state["log"].append(f"ERROR: {e}")


@app.route("/backtest")
def backtest_index():
    return render_template("backtest.html")


@app.route("/backtest/run", methods=["POST"])
def backtest_run():
    global _bt_thread
    data       = request.get_json(silent=True) or request.form or {}
    start_date = (data.get("start_date") if hasattr(data, "get") else None) or ""
    end_date   = (data.get("end_date")   if hasattr(data, "get") else None) or ""
    try:
        _dt.date.fromisoformat(start_date)
        _dt.date.fromisoformat(end_date)
    except Exception:
        return jsonify({"status": "error", "error": "bad date range"}), 400

    with _bt_lock:
        if _bt_state["status"] == "running":
            return jsonify({"status": "busy", "message": "backtest already running"}), 409
    with _state_lock:
        if _state["status"] == "running":
            return jsonify({"status": "busy", "message": "analysis already running"}), 409

    _bt_thread = threading.Thread(
        target=_run_backtest, args=(start_date, end_date), daemon=True)
    _bt_thread.start()
    return jsonify({"status": "started", "start": start_date, "end": end_date})


@app.route("/backtest/status")
def backtest_status():
    with _bt_lock:
        return jsonify({
            "status": _bt_state["status"],
            "start":  _bt_state["start"],
            "end":    _bt_state["end"],
            "log":    list(_bt_state["log"]),
            "result": _bt_state["result"],
            "error":  _bt_state["error"],
        })


@app.route("/backtest/results")
def backtest_results():
    return jsonify(mlb_backtest.load_summary())


@app.route("/run", methods=["POST"])
def run():
    global _worker_thread

    data = request.get_json(silent=True) or request.form or {}
    date_str = (data.get("date") if hasattr(data, "get") else None) or _dt.date.today().isoformat()
    batter_window = (data.get("batter_window") if hasattr(data, "get") else None) or "season"
    if batter_window not in ("season", "last30"):
        return jsonify({"status": "error", "error": f"bad batter_window: {batter_window}"}), 400

    # Basic validation: must parse as YYYY-MM-DD.
    try:
        _dt.date.fromisoformat(date_str)
    except Exception:
        return jsonify({"status": "error", "error": f"bad date: {date_str}"}), 400

    with _state_lock:
        currently_running = _state["status"] == "running"
    if currently_running:
        return jsonify({"status": "busy", "message": "analysis already in progress"}), 409

    _worker_thread = threading.Thread(
        target=_run_analysis, args=(date_str, batter_window), daemon=True
    )
    _worker_thread.start()
    return jsonify({"status": "started", "date": date_str})


@app.route("/status")
def status():
    return jsonify(_snapshot())


@app.route("/cancel", methods=["POST"])
def cancel():
    with _state_lock:
        _state["cancel_requested"] = True
    return jsonify({"status": "cancel_requested"})


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})


# -----------------------------------------------------------------------------
# Golf tab routes
# -----------------------------------------------------------------------------

@app.route("/golf")
def golf_index():
    return render_template("golf.html", view_only=VIEW_ONLY)


@app.route("/golf/api/state")
def golf_api_state():
    if not _GOLF_OK:
        return jsonify({"empty": True, "error": "golf module not loaded"})
    try:
        state = _golf_state(request.args.get("refresh") == "1")
        # Self-heal: if data is stale, refresh in the background for the next view
        # (this viewer still gets the cached data instantly).
        _golf_refresh_if_stale("on-view")
        return jsonify(state)
    except Exception as e:
        log.exception("golf state failed")
        return jsonify({"empty": True, "error": str(e)})


@app.route("/golf/api/pull", methods=["POST"])
def golf_api_pull():
    if VIEW_ONLY:
        return jsonify({"ok": False, "error": "pull disabled (view-only terminal)"}), 403
    if not _GOLF_OK:
        return jsonify({"ok": False, "error": "golf module not loaded"}), 500
    try:
        ts = golf_dashboard.pull_and_snapshot()   # credit-guarded
        _golf_state(rebuild=True)
        return jsonify({"ok": True, "run": ts})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/golf/healthz")
def golf_healthz():
    """Uptime probe for the golf tab. 200 healthy, 503 if the module didn't load."""
    state = _golf_cache.get("state") or {}
    body = {
        "ok": _GOLF_OK,
        "module_loaded": _GOLF_OK,
        "state_cached": "state" in _golf_cache,
        "run": state.get("run"),
        "view_only": VIEW_ONLY,
        "auto_refresh_min": GOLF_REFRESH_MINUTES,
    }
    return jsonify(body), (200 if _GOLF_OK else 503)


# -----------------------------------------------------------------------------
# Credit-safe auto-refresh — runs IN the web container (shares the snapshot
# file, unlike a separate Render cron). Opt-in via GOLF_REFRESH_MINUTES; every
# tick is gated by the CreditGuard so spend stays capped and predictable.
# -----------------------------------------------------------------------------
GOLF_REFRESH_MINUTES = int(os.environ.get("GOLF_REFRESH_MINUTES", "60"))  # 0 = off


def _golf_snapshot_age_min():
    """Minutes since the newest odds snapshot, or None if there are none."""
    try:
        from datetime import datetime, timezone
        from src import store
        run = store.closing_run()
        if not run:
            return None
        return (datetime.now(timezone.utc) -
                datetime.fromisoformat(run)).total_seconds() / 60.0
    except Exception:
        return None


# Timestamp a pull started (0 = idle). A pull older than the timeout is presumed
# hung (a stalled network call or crashed rebuild) and gets superseded — a plain
# Lock would stay held forever and freeze all future refreshes, which is exactly
# how the movement got stuck at one timestamp.
_golf_refresh = {"since": 0.0}
_GOLF_PULL_TIMEOUT = 300


def _golf_pull_once(reason: str) -> None:
    """Pull odds + rebuild state. Skips only if a *recent* pull is still running."""
    import time
    now = time.time()
    since = _golf_refresh["since"]
    if since and (now - since) < _GOLF_PULL_TIMEOUT:
        log.info("golf pull (%s) skipped — a refresh is already running", reason)
        return
    _golf_refresh["since"] = now
    try:
        ts = golf_dashboard.pull_and_snapshot()
        _golf_state(rebuild=True)
        log.info("golf pull (%s): snapshot %s", reason, ts)
    except Exception as e:
        log.warning("golf pull (%s) FAILED: %s", reason, e)   # was swallowed before
    finally:
        _golf_refresh["since"] = 0.0


def _golf_cache_behind_store() -> bool:
    """True if the store has a newer snapshot than the cached state was built from
    — i.e. a pull landed but its cache rebuild failed, so the view is stale."""
    try:
        from src import store
        latest = store.closing_run()
        cached = (_golf_cache.get("state") or {}).get("run")
        return bool(latest and cached and latest > cached)
    except Exception:
        return False


def _golf_refresh_if_stale(reason: str) -> None:
    """Self-heal (free tier): kick a background pull+rebuild if the latest snapshot
    is old OR the cache has fallen behind the store (a prior rebuild failed).
    Returns immediately — current viewer gets cached data fast; the pull's guard
    dedupes concurrent calls so a stuck one can't block forever."""
    age = _golf_snapshot_age_min()
    stale = age is None or age >= GOLF_REFRESH_MINUTES
    if not stale and not _golf_cache_behind_store():
        return
    threading.Thread(target=_golf_pull_once, args=(reason,), daemon=True).start()


def _golf_refresh_loop() -> None:
    import time
    # Boot pull: refresh ~30s after startup so every deploy gets fresh odds —
    # but SKIP if the latest snapshot is still within the refresh window, so
    # frequent Render cold-starts don't re-spend credits. Guard caps it anyway.
    time.sleep(30)
    age = _golf_snapshot_age_min()
    if age is None or age >= GOLF_REFRESH_MINUTES:
        try:
            _golf_pull_once("boot")
        except Exception as e:
            log.warning("golf boot pull failed: %s", e)
    else:
        log.info("golf boot pull skipped — snapshot only %.0f min old", age)
    # Then refresh on the interval.
    while True:
        time.sleep(GOLF_REFRESH_MINUTES * 60)
        try:
            _golf_pull_once("scheduled")
        except Exception as e:
            log.warning("golf auto-refresh failed: %s", e)


if _GOLF_OK and GOLF_REFRESH_MINUTES > 0:
    threading.Thread(target=_golf_refresh_loop, daemon=True).start()
    log.info("golf auto-refresh every %d min (+ boot pull)", GOLF_REFRESH_MINUTES)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def _print_banner(port: int) -> None:
    bar = "=" * 56
    print(bar)
    print("  Pitch-Type Matchup Tool")
    print(f"  -> http://127.0.0.1:{port}")
    print(bar)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    _print_banner(port)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
