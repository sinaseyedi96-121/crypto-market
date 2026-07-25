"""All-time hypothetical-signal track record, computed from posts_log.jsonl.

Shared by two callers: the on-close reply (a running tally appended under each
closed scenario) and the weekly standalone scorecard post. Reading the log
rather than keeping a separate counter means the track record can never drift
from what was actually published, and a fresh clone reconstructs it exactly.
"""

from __future__ import annotations

import json
import os

import config


def _load_closed_signals(path: str | None = None) -> list[dict]:
    """Every logged signal_close record, oldest first. Best-effort: a missing or
    partially-corrupt log yields whatever valid records it contains."""
    path = path or config.POSTS_LOG_FILE
    if not os.path.exists(path):
        return []
    closed = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("mode") == "signal_close":
                closed.append(entry)
    return closed


def compute_position_sizing(entry: float, stop: float,
                            risk_pct: float = config.SIGNAL_RISK_PER_TRADE_PCT,
                            max_leverage: float = config.SIGNAL_MAX_LEVERAGE) -> dict:
    """Illustrative position size/leverage for a scenario's entry-to-stop
    distance, against a fixed hypothetical example account (see
    SIGNAL_DISCLAIMERS) — not sized advice for the reader's own capital.

    Sizing follows the standard risk-based convention: position size is
    whatever notional, as a % of the hypothetical account, makes the
    entry-to-stop move cost exactly `risk_pct` of that account. When that
    would require more than `max_leverage`, the position is capped at
    `max_leverage` instead and the realized risk comes in under the target
    (surfaced via `capped`/`risk_pct` so the caption can say so).
    """
    stop_distance_pct = abs(entry - stop) / entry * 100 if entry else 0.0
    if stop_distance_pct <= 0:
        return {"position_size_pct": 0.0, "leverage": 1.0, "risk_pct": 0.0, "capped": False}
    leverage_needed = risk_pct / stop_distance_pct
    capped = leverage_needed > max_leverage
    leverage = min(leverage_needed, max_leverage)
    position_size_pct = leverage * 100
    realized_risk_pct = stop_distance_pct * leverage
    return {
        "position_size_pct": round(position_size_pct, 1),
        "leverage": round(max(leverage, 1.0), 2),
        "risk_pct": round(realized_risk_pct, 2),
        "capped": capped,
    }


def compute_scorecard(closed: list[dict], open_count: int = 0) -> dict:
    """Aggregate a list of signal_close records into a track-record summary.

    Win/loss is classified by the sign of `r_multiple` rather than the
    `outcome` label, so a scenario that partially hit targets before
    eventually stopping out still lands in the right bucket."""
    total = len(closed)
    wins = [p for p in closed if float(p.get("r_multiple", 0.0)) > 0]
    losses = [p for p in closed if float(p.get("r_multiple", 0.0)) <= 0]
    r_values = [float(p.get("r_multiple", 0.0)) for p in closed]
    best = max(closed, key=lambda p: float(p.get("r_multiple", 0.0))) if closed else None
    worst = min(closed, key=lambda p: float(p.get("r_multiple", 0.0))) if closed else None
    return {
        "closed_count": total,
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate_pct": (len(wins) / total * 100) if total else None,
        "avg_r": (sum(r_values) / total) if total else 0.0,
        "total_r": sum(r_values),
        "best": {"ticker": best.get("ticker"), "r_multiple": float(best.get("r_multiple", 0.0))} if best else None,
        "worst": {"ticker": worst.get("ticker"), "r_multiple": float(worst.get("r_multiple", 0.0))} if worst else None,
        "open_count": open_count,
    }


def load_scorecard(open_count: int = 0, path: str | None = None,
                   include: dict | None = None) -> dict:
    """Read the log and compute the scorecard. `include` lets a caller fold in a
    close that hasn't been written to the log yet (the on-close reply computes
    its tally before appending its own record), so the reply reflects itself."""
    closed = _load_closed_signals(path)
    if include is not None:
        closed = [*closed, include]
    return compute_scorecard(closed, open_count)


def format_record_line(scorecard: dict, language: str = "en") -> str:
    """One compact educational line summarizing the track record so far, suitable
    for appending under a signal's outcome reply. Localized so each channel's
    closeout reply carries the tally in its own language."""
    total = scorecard["closed_count"]
    if not total:
        if language == "fa":
            return "📊 کارنامه: این نخستین سناریوی فرضی است که بسته می‌شود."
        return "📊 Track record: this is the first hypothetical scenario to close."
    win_rate = scorecard["win_rate_pct"]
    if language == "fa":
        return (
            f"📊 کارنامه تا اینجا: {total} سناریو بسته شد · "
            f"{scorecard['win_count']} به هدف رسید ({win_rate:.0f}%) · "
            f"میانگین نتیجه {scorecard['avg_r']:+.1f}R"
        )
    return (
        f"📊 Track record so far: {total} scenarios closed · "
        f"{scorecard['win_count']} reached target ({win_rate:.0f}%) · "
        f"average outcome {scorecard['avg_r']:+.1f}R"
    )
