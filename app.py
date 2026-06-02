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
