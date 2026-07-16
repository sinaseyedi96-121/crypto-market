"""Balanced publisher for the crypto channel.

Scheduled runs produce a daily market map, a daily derivatives pulse, a
two-asset deep-dive album, a weekday macro close, and a Sunday weekly digest.
Confirmed candles and per-slot state prevent unfinished or duplicate analysis
from being published.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import sys
import time
from zoneinfo import ZoneInfo

import pandas as pd

import chart_generator
import config
import data_fetcher
import indicators
import narrative_generator
import state_manager
import telegram_publisher


ROME = ZoneInfo("Europe/Rome")
NEW_YORK = ZoneInfo("America/New_York")


def _signal_direction(trend: str) -> str | None:
    regime = trend.split(":", 1)[0]
    if regime == "bullish":
        return "long"
    if regime == "bearish":
        return "short"
    return None


def _evaluate_signal(signal: dict, price: float) -> tuple[str, float] | None:
    """Checks a confirmed close against an open hypothetical signal's target/stop."""
    direction = signal["direction"]
    entry = signal["entry"]
    target = signal["target"]
    stop = signal["stop"]
    risk = abs(entry - stop)
    reward_actual = (price - entry) if direction == "long" else (entry - price)
    r_multiple = reward_actual / risk if risk else 0.0
    hit_target = price >= target if direction == "long" else price <= target
    hit_stop = price <= stop if direction == "long" else price >= stop
    if hit_target:
        return "target", r_multiple
    if hit_stop:
        return "stop", r_multiple
    return None


def _maybe_open_signal(state: dict, chat_id: str, asset: dict, timeframe: str,
                        analysis: dict, chart_message_id: int | None) -> None:
    """Opens one hypothetical long/short scenario per asset, as a reply under
    its own chart, if none is already open. Educational only — see README."""
    symbol = asset["symbol"]
    if state_manager.get_open_signal(state, symbol):
        return
    direction = _signal_direction(analysis["trend"])
    if not direction or not chart_message_id:
        return

    levels = analysis["levels"]
    entry = analysis["price"]
    target = levels["resistance"] if direction == "long" else levels["support"]
    stop = levels["support"] if direction == "long" else levels["resistance"]
    caption = narrative_generator.generate_signal_post(
        asset["ticker"], timeframe, direction, entry, target, stop,
        analysis["trend"], analysis["rsi"],
    )
    posted = telegram_publisher.reply_to_message(chat_id, chart_message_id, caption)
    state_manager.open_signal(state, symbol, {
        "direction": direction,
        "entry": entry,
        "target": target,
        "stop": stop,
        "timeframe": timeframe,
        "opened_at": time.time(),
        "message_id": posted.get("message_id"),
    })
    state_manager.append_post_log({
        "timestamp": time.time(),
        "mode": "signal_open",
        "ticker": asset["ticker"],
        "symbol": symbol,
        "timeframe": timeframe,
        "direction": direction,
        "entry": entry,
        "target": target,
        "stop": stop,
        "caption": caption,
        "message_id": posted.get("message_id"),
    })
    print(f"Opened hypothetical {direction} scenario for {asset['ticker']}.")


def check_followup(prev_entry: dict, current_price: float) -> str | None:
    if not prev_entry:
        return None
    support = prev_entry["levels"]["support"]
    resistance = prev_entry["levels"]["resistance"]
    pct = config.FOLLOWUP_THRESHOLD_PCT / 100
    if current_price > resistance * (1 + pct):
        return f"Price moved above the ~{resistance:.2f} resistance zone from the previous post and is now {current_price:.2f}."
    if current_price < support * (1 - pct):
        return f"Price moved below the ~{support:.2f} support zone from the previous post and is now {current_price:.2f}."
    return None


def _today() -> str:
    return datetime.now(ROME).date().isoformat()


def _slot_already_posted(state: dict, slot: str) -> bool:
    return state.get("_schedule", {}).get(slot) == _today()


def _mark_slot(state: dict, slot: str) -> None:
    state.setdefault("_schedule", {})[slot] = _today()


def _week_key() -> str:
    iso = datetime.now(ROME).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _slot_already_posted_week(state: dict, slot: str) -> bool:
    return state.get("_schedule_weekly", {}).get(slot) == _week_key()


def _mark_slot_week(state: dict, slot: str) -> None:
    state.setdefault("_schedule_weekly", {})[slot] = _week_key()


def _record_daily_snapshot(state: dict, total_mcap: float, btc_dominance: float,
                           stablecoin_mcap: float) -> None:
    """Accumulate one data point per day so the weekly digest can chart trends
    CoinGecko's free tier has no historical endpoint for (total mcap, dominance)."""
    snapshots = state.setdefault("_daily_snapshots", [])
    today = _today()
    if snapshots and snapshots[-1]["date"] == today:
        return
    snapshots.append({
        "date": today,
        "total_mcap": total_mcap,
        "btc_dominance": btc_dominance,
        "stablecoin_mcap": stablecoin_mcap,
    })
    del snapshots[: max(0, len(snapshots) - config.DAILY_SNAPSHOT_HISTORY_DAYS)]


def _snapshot_series(state: dict, field: str) -> pd.Series:
    snapshots = state.get("_daily_snapshots", [])
    if not snapshots:
        return pd.Series(dtype=float)
    index = pd.to_datetime([row["date"] for row in snapshots])
    return pd.Series([row[field] for row in snapshots], index=index, name=field)


def _asset(ticker: str) -> dict:
    return next(asset for asset in config.ASSETS if asset["ticker"] == ticker)


def _publish(chat_id: str, image_paths: list[str], caption: str) -> dict:
    return telegram_publisher.post_charts(chat_id, image_paths, caption)


def run_market_map(state: dict, chat_id: str, force: bool = False) -> None:
    try:
        global_snapshot = data_fetcher.fetch_global_snapshot()
        stablecoin_mcap = data_fetcher.fetch_stablecoin_market_cap()
        _record_daily_snapshot(
            state, global_snapshot["total_market_cap_usd"],
            global_snapshot["btc_dominance"], stablecoin_mcap,
        )
    except Exception as exc:
        print(f"Could not record today's market-structure snapshot: {exc}", file=sys.stderr)

    if not force and _slot_already_posted(state, "market_map"):
        print("Today's market map was already posted; skipping duplicate.")
        return
    snapshot = data_fetcher.fetch_market_snapshot(config.ASSETS)
    fear_greed = data_fetcher.fetch_fear_greed_index()
    chart_path = chart_generator.generate_market_map_chart(snapshot)
    caption = narrative_generator.generate_market_map(snapshot, fear_greed)
    posted = _publish(chat_id, [chart_path], caption)
    state_manager.append_post_log({
        "timestamp": time.time(),
        "mode": "market_map",
        "caption": caption,
        "message_id": posted.get("message_id"),
    })
    _mark_slot(state, "market_map")
    print("Posted daily market map.")


def run_deep_dive(state: dict, chat_id: str, force: bool = False) -> None:
    if not force and _slot_already_posted(state, "deep_dive"):
        print("Today's deep dive was already posted; skipping duplicate.")
        return

    pair = config.DEEP_DIVE_ROTATION[datetime.now(ROME).weekday()]
    timeframe = config.DEEP_DIVE_TIMEFRAME
    analyses = []
    chart_paths = []
    pending_state = []

    for ticker in pair:
        asset = _asset(ticker)
        print(f"Preparing {ticker} {timeframe} deep dive.")
        frame = data_fetcher.fetch_asset_klines(asset, timeframe, config.CANDLE_LIMIT)
        frame = indicators.enrich(frame)
        levels = indicators.find_key_levels(frame)
        current_price = float(frame["Close"].iloc[-1])
        candle_closed_at = frame["CloseTime"].iloc[-1].isoformat()
        previous = state_manager.get_entry(state, asset["symbol"], timeframe)

        followup = check_followup(previous, current_price)
        if followup and previous.get("message_id"):
            text = narrative_generator.generate_followup(ticker, timeframe, followup)
            followup_posted = telegram_publisher.reply_to_message(chat_id, previous["message_id"], text)
            state_manager.append_post_log({
                "timestamp": time.time(),
                "mode": "followup",
                "ticker": ticker,
                "timeframe": timeframe,
                "caption": text,
                "message_id": followup_posted.get("message_id"),
            })
            print(f"Posted {ticker} level follow-up.")

        chart_paths.append(chart_generator.generate_chart(frame, asset["symbol"], timeframe, levels))
        analyses.append({
            "ticker": ticker,
            "timeframe": timeframe,
            "summary": indicators.summarize_for_prompt(frame, levels),
            "trend": indicators.trend_state(frame),
            "rsi": float(frame["rsi"].iloc[-1]),
            "price": current_price,
            "levels": levels,
        })
        pending_state.append((asset, levels, current_price, candle_closed_at))
        time.sleep(1)

    fear_greed = data_fetcher.fetch_fear_greed_index()
    caption = narrative_generator.generate_comparison(analyses, fear_greed)
    posted = _publish(chat_id, chart_paths, caption)
    state_manager.append_post_log({
        "timestamp": time.time(),
        "mode": "deep_dive",
        "assets": list(pair),
        "timeframe": timeframe,
        "caption": caption,
        "message_id": posted.get("message_id"),
    })
    album_ids = posted.get("album_message_ids") or [posted.get("message_id")] * len(pair)
    for (asset, levels, current_price, candle_closed_at), analysis, chart_message_id in zip(
        pending_state, analyses, album_ids
    ):
        state_manager.set_entry(state, asset["symbol"], timeframe, {
            "levels": levels,
            "price_at_post": current_price,
            "message_id": posted.get("message_id"),
            "posted_at": posted.get("date"),
            "candle_closed_at": candle_closed_at,
        })
        _maybe_open_signal(state, chat_id, asset, timeframe, analysis, chart_message_id)
    _mark_slot(state, "deep_dive")
    print(f"Posted deep-dive album for {' + '.join(pair)}.")


def _technical_event(previous: dict | None, price: float, trend: str) -> tuple[int, str] | None:
    if not previous:
        return None
    prior_levels = previous["levels"]
    zone_width = float(prior_levels.get("zone_width", 0))
    if price > prior_levels["resistance"] + zone_width:
        return 2, f"Closed above the previous resistance zone near {prior_levels['resistance']:,.4g}"
    if price < prior_levels["support"] - zone_width:
        return 2, f"Closed below the previous support zone near {prior_levels['support']:,.4g}"

    current_regime = trend.split(":", 1)[0]
    previous_regime = previous.get("trend", "").split(":", 1)[0]
    directional = {"bullish", "bearish"}
    if current_regime in directional and previous_regime in directional and current_regime != previous_regime:
        return 1, f"EMA and price structure changed from {previous_regime} to {current_regime}"
    return None


def run_alert_scan(state: dict, chat_id: str, force: bool = False) -> None:
    """Scan confirmed 4h candles but publish only significant, capped events."""
    timeframe = "4h"
    candidates = []
    staged_entries = []
    scanned = 0

    for asset in config.ASSETS:
        try:
            frame = indicators.enrich(
                data_fetcher.fetch_asset_klines(asset, timeframe, config.CANDLE_LIMIT)
            )
            levels = indicators.find_key_levels(frame)
            price = float(frame["Close"].iloc[-1])
            close_time = frame["CloseTime"].iloc[-1].isoformat()
            trend = indicators.trend_state(frame)
            previous = state_manager.get_entry(state, asset["symbol"], timeframe)
            scanned += 1
        except Exception as exc:
            print(f"Skipping {asset['ticker']} alert scan: {exc}", file=sys.stderr)
            continue

        open_signal = state_manager.get_open_signal(state, asset["symbol"])
        if open_signal:
            result = _evaluate_signal(open_signal, price)
            if result:
                outcome, r_multiple = result
                outcome_caption = narrative_generator.generate_signal_outcome(
                    asset["ticker"], open_signal["timeframe"], open_signal["direction"],
                    open_signal["entry"], price, outcome, r_multiple, close_time,
                )
                outcome_posted = telegram_publisher.reply_to_message(
                    chat_id, open_signal["message_id"], outcome_caption
                )
                state_manager.close_signal(state, asset["symbol"])
                state_manager.append_post_log({
                    "timestamp": time.time(),
                    "mode": "signal_close",
                    "ticker": asset["ticker"],
                    "symbol": asset["symbol"],
                    "timeframe": open_signal["timeframe"],
                    "direction": open_signal["direction"],
                    "entry": open_signal["entry"],
                    "exit_price": price,
                    "outcome": outcome,
                    "r_multiple": r_multiple,
                    "caption": outcome_caption,
                    "message_id": outcome_posted.get("message_id"),
                })
                print(f"Closed hypothetical {open_signal['direction']} scenario for {asset['ticker']}: {outcome} ({r_multiple:+.2f}R).")

        if not force and previous and previous.get("candle_closed_at") == close_time:
            continue

        event = _technical_event(previous, price, trend)
        entry = {
            "levels": levels,
            "price_at_post": price,
            "message_id": previous.get("message_id") if previous else None,
            "posted_at": previous.get("posted_at") if previous else None,
            "candle_closed_at": close_time,
            "trend": trend,
        }
        staged_entries.append((asset, entry))
        if event:
            severity, description = event
            candidates.append({
                "severity": severity,
                "asset": asset,
                "frame": frame,
                "levels": levels,
                "price": price,
                "close_time": close_time,
                "description": description,
                "entry": entry,
            })

    if not scanned:
        raise RuntimeError("No assets could be scanned for confirmed 4h events")

    tracker = state.get("_alerts", {})
    used = tracker.get("count", 0) if tracker.get("date") == _today() else 0
    available = max(config.MAX_EVENT_ALERTS_PER_DAY - used, 0)
    selected = sorted(candidates, key=lambda item: item["severity"], reverse=True)[:available]

    for candidate in selected:
        asset = candidate["asset"]
        path = chart_generator.generate_chart(
            candidate["frame"], asset["symbol"], timeframe, candidate["levels"]
        )
        caption = narrative_generator.generate_event_alert(
            asset["ticker"], candidate["price"], candidate["description"],
            candidate["levels"], candidate["close_time"],
            candidate["frame"].attrs.get("source", "Binance"),
        )
        posted = _publish(chat_id, [path], caption)
        candidate["entry"]["message_id"] = posted.get("message_id")
        candidate["entry"]["posted_at"] = posted.get("date")
        state_manager.append_post_log({
            "timestamp": time.time(),
            "mode": "alert",
            "ticker": asset["ticker"],
            "timeframe": timeframe,
            "caption": caption,
            "message_id": posted.get("message_id"),
        })
        used += 1
        print(f"Posted confirmed 4h alert for {asset['ticker']}.")

    for asset, entry in staged_entries:
        state_manager.set_entry(state, asset["symbol"], timeframe, entry)
    state["_alerts"] = {"date": _today(), "count": used}
    if not selected:
        print("4h scan complete; no publishable event within today's alert cap.")


def _enrich_macro_snapshot(macro: dict) -> None:
    """Best-effort extras: VIX, yield spread, BTC correlation. Never blocks the core post."""
    try:
        macro["vix"] = float(data_fetcher.fetch_fred_series(config.FRED_VIX_SERIES).iloc[-1])
    except Exception as exc:
        print(f"VIX fetch failed: {exc}", file=sys.stderr)
    try:
        macro["yield_10y"] = float(data_fetcher.fetch_fred_series(config.FRED_10Y_SERIES).iloc[-1])
        macro["yield_2y"] = float(data_fetcher.fetch_fred_series(config.FRED_2Y_SERIES).iloc[-1])
    except Exception as exc:
        print(f"Treasury yield fetch failed: {exc}", file=sys.stderr)
    try:
        btc = data_fetcher.fetch_asset_klines(_asset("BTC"), "1d", config.CORRELATION_WINDOW + 20)
        macro["btc_sp_corr"] = indicators.rolling_correlation(
            btc["Close"], macro["sp500"], config.CORRELATION_WINDOW
        )
        macro["btc_dollar_corr"] = indicators.rolling_correlation(
            btc["Close"], macro["dollar"], config.CORRELATION_WINDOW
        )
    except Exception as exc:
        print(f"BTC correlation calculation failed: {exc}", file=sys.stderr)


def run_macro_close(state: dict, chat_id: str, force: bool = False) -> None:
    new_york_now = datetime.now(NEW_YORK)
    if not force and new_york_now.weekday() < 5 and (new_york_now.hour, new_york_now.minute) < (16, 10):
        print("US cash session has not closed; waiting for the next macro cron.")
        return
    if not force and _slot_already_posted(state, "macro_close"):
        print("Today's macro close was already posted; skipping duplicate.")
        return
    macro = data_fetcher.fetch_macro_snapshot()
    _enrich_macro_snapshot(macro)
    chart_path = chart_generator.generate_macro_chart(macro)
    caption = narrative_generator.generate_macro(macro)
    posted = _publish(chat_id, [chart_path], caption)
    state_manager.append_post_log({
        "timestamp": time.time(),
        "mode": "macro_close",
        "caption": caption,
        "message_id": posted.get("message_id"),
    })
    _mark_slot(state, "macro_close")
    print("Posted macro close.")


def run_daily_pulse(state: dict, chat_id: str, force: bool = False) -> None:
    if not force and _slot_already_posted(state, "daily_pulse"):
        print("Today's daily pulse was already posted; skipping duplicate.")
        return

    derivatives_rows = data_fetcher.fetch_derivatives_snapshot(config.PULSE_ASSETS)
    if not derivatives_rows:
        raise RuntimeError("Not enough derivatives data to publish the daily pulse")

    try:
        trending = data_fetcher.fetch_trending_coins()
    except Exception as exc:
        print(f"Trending coins fetch failed: {exc}", file=sys.stderr)
        trending = []

    chart_path = chart_generator.generate_pulse_chart(derivatives_rows)
    caption = narrative_generator.generate_daily_pulse(derivatives_rows, trending)
    _publish(chat_id, [chart_path], caption)
    _mark_slot(state, "daily_pulse")
    print("Posted daily derivatives pulse.")


def run_weekly_digest(state: dict, chat_id: str, force: bool = False) -> None:
    if not force and datetime.now(ROME).weekday() != config.WEEKLY_DIGEST_WEEKDAY:
        print("Not the weekly digest's scheduled day; skipping.")
        return
    if not force and _slot_already_posted_week(state, "weekly_digest"):
        print("This week's digest was already posted; skipping duplicate.")
        return

    weekly_rows = data_fetcher.fetch_market_snapshot(config.ASSETS)

    scoreboard_rows = []
    for asset in config.ASSETS:
        try:
            frame = indicators.enrich(
                data_fetcher.fetch_asset_klines(asset, config.DEEP_DIVE_TIMEFRAME, config.CANDLE_LIMIT)
            )
        except Exception as exc:
            print(f"Skipping {asset['ticker']} in the technical scoreboard: {exc}", file=sys.stderr)
            continue
        last = frame.iloc[-1]
        scoreboard_rows.append({
            "ticker": asset["ticker"],
            "rsi": float(last["rsi"]),
            "atr_pct": float(last["atr_pct"]),
        })
        time.sleep(0.5)
    if len(scoreboard_rows) < len(config.ASSETS) // 2:
        raise RuntimeError("Not enough assets returned candle data for the technical scoreboard")

    feargreed_history = data_fetcher.fetch_fear_greed_history(30)
    dominance_history = _snapshot_series(state, "btc_dominance")
    total_mcap_history = _snapshot_series(state, "total_mcap")
    stablecoin_history = _snapshot_series(state, "stablecoin_mcap")
    if dominance_history.empty or total_mcap_history.empty or stablecoin_history.empty:
        raise RuntimeError("No accumulated market-structure history yet; the daily market map "
                           "must run at least once before the weekly digest can chart trends")

    chart_paths = [
        chart_generator.generate_weekly_recap_chart(weekly_rows),
        chart_generator.generate_technical_scoreboard_chart(scoreboard_rows),
        chart_generator.generate_market_structure_chart(dominance_history, feargreed_history),
        chart_generator.generate_liquidity_chart(total_mcap_history, stablecoin_history),
    ]
    caption = narrative_generator.generate_weekly_digest(
        weekly_rows, scoreboard_rows, dominance_history, feargreed_history,
        total_mcap_history, stablecoin_history,
    )
    _publish(chat_id, chart_paths, caption)
    _mark_slot_week(state, "weekly_digest")
    print("Posted weekly digest.")


# Maps each GitHub Actions cron string (github.event.schedule, UTC) to its content slot.
CRON_MODE_MAP = {
    "10 */4 * * *": "alerts",
    "15 4 * * *": "market_map",
    "15 8 * * *": "daily_pulse",
    "15 12 * * *": "deep_dive",
    "15 16 * * 0": "weekly_digest",
    "15 20 * * 1-5": "macro_close",
    "15 21 * * 1-5": "macro_close",
}


def _automatic_mode() -> str:
    """Map the GitHub cron schedule to its content slot. Falls back to an hour
    bucket for local/manual runs where CRON_SCHEDULE isn't set."""
    cron = os.environ.get("CRON_SCHEDULE")
    if cron in CRON_MODE_MAP:
        return CRON_MODE_MAP[cron]
    hour = datetime.now(timezone.utc).hour
    if hour < 6:
        return "market_map"
    if hour < 10:
        return "daily_pulse"
    if hour < 17:
        return "deep_dive"
    return "macro_close"


def run(mode: str = "auto", force: bool = False) -> None:
    state = state_manager.load_state()
    chat_id = _require_env("TELEGRAM_CHANNEL")
    resolved = _automatic_mode() if mode == "auto" else mode
    modes = (
        ["market_map", "daily_pulse", "deep_dive", "macro_close", "weekly_digest"]
        if resolved == "all" else [resolved]
    )
    handlers = {
        "market_map": run_market_map,
        "deep_dive": run_deep_dive,
        "macro_close": run_macro_close,
        "alerts": run_alert_scan,
        "daily_pulse": run_daily_pulse,
        "weekly_digest": run_weekly_digest,
    }

    try:
        for selected in modes:
            try:
                handlers[selected](state, chat_id, force=force)
            except Exception as exc:
                print(f"ERROR in {selected}: {exc}", file=sys.stderr)
                if len(modes) == 1:
                    raise
            state_manager.save_state(state)
    finally:
        state_manager.save_state(state)
    print("State saved.")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("auto", "market_map", "deep_dive", "macro_close", "alerts",
                 "daily_pulse", "weekly_digest", "all"),
        default=os.environ.get("POST_MODE", "auto"),
    )
    parser.add_argument("--force", action="store_true", help="republish even if today's slot is recorded")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _arguments()
    run(arguments.mode, arguments.force)
