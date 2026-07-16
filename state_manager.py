"""
Tiny JSON-file state store, committed back to the repo after each run —
same persistence pattern as the rest of your pipelines (GitHub Actions
runners are ephemeral, so this file is the only memory between runs).
"""

from __future__ import annotations

import json
import os
import config


def load_state() -> dict:
    if not os.path.exists(config.STATE_FILE):
        return {}
    with open(config.STATE_FILE, "r") as f:
        return json.load(f)


def save_state(state: dict) -> None:
    with open(config.STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_entry(state: dict, symbol: str, timeframe: str) -> dict | None:
    return state.get(symbol, {}).get(timeframe)


def set_entry(state: dict, symbol: str, timeframe: str, entry: dict) -> None:
    state.setdefault(symbol, {})[timeframe] = entry


def get_open_signal(state: dict, symbol: str) -> dict | None:
    return state.get("_signals", {}).get(symbol)


def open_signal(state: dict, symbol: str, signal: dict) -> None:
    state.setdefault("_signals", {})[symbol] = signal


def close_signal(state: dict, symbol: str) -> dict | None:
    return state.get("_signals", {}).pop(symbol, None)


def append_post_log(entry: dict) -> None:
    """Append one JSONL record for the content reviewer to read later.

    Best-effort: a logging failure should never break a publishing run.
    """
    try:
        with open(config.POSTS_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as exc:
        print(f"Could not append to posts log: {exc}")
