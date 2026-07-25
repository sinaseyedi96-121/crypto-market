"""Render a social-first technical chart from confirmed Binance candles."""

from __future__ import annotations

from datetime import date
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import mplfinance as mpf

import config


FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
FONT_FAMILY = "PT Serif"
for _font_file in ("PTSerif-Regular.ttf", "PTSerif-Bold.ttf"):
    fm.fontManager.addfont(os.path.join(FONT_DIR, _font_file))
plt.rcParams["font.family"] = FONT_FAMILY


BACKGROUND = "#08111F"
PANEL = "#0D192A"
GRID = "#26364B"
TEXT = "#E8EEF7"
MUTED = "#8EA0B8"
UP = "#00C2A8"
DOWN = "#FF5A7A"
EMA_FAST = "#31B7FF"
EMA_SLOW = "#B98CFF"
BB = "#8292A8"
SUPPORT = "#31D09D"
RESISTANCE = "#FF7A90"
CURRENT = "#F4C95D"


def _style():
    market_colors = mpf.make_marketcolors(
        up=UP,
        down=DOWN,
        edge="inherit",
        wick="inherit",
        volume="inherit",
    )
    return mpf.make_mpf_style(
        base_mpf_style="nightclouds",
        marketcolors=market_colors,
        facecolor=PANEL,
        figcolor=BACKGROUND,
        gridcolor=GRID,
        gridstyle="--",
        y_on_right=True,
        rc={
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "font.family": FONT_FAMILY,
            "font.size": 10,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        },
    )


def _rotating_choice(pool: list[str], seed: int) -> str:
    """Pick one headline from a pool, deterministically but varying day to day so
    a market that stays in the same regime for a stretch doesn't repeat the exact
    same headline every session."""
    return pool[int(seed) % len(pool)]


def _display_symbol(symbol: str, quote: str = "USDT") -> str:
    return f"{symbol[:-4]} / {quote}" if symbol.endswith("USDT") else symbol


def _compact_number(value: float, _position=None) -> str:
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}K"
    return f"{value:.0f}"


def _format_price(value: float) -> str:
    if abs(value) >= 1_000:
        return f"${value:,.0f}"
    if abs(value) >= 1:
        return f"${value:,.2f}"
    if abs(value) >= 0.01:
        return f"${value:.4f}"
    return f"${value:.6f}"


def _technical_headline(last, levels: dict) -> str:
    price = float(last["Close"])
    zone_width = float(levels.get("zone_width", 0))
    if price > levels["resistance"] + zone_width:
        return "CONFIRMED RESISTANCE BREAK"
    if price < levels["support"] - zone_width:
        return "CONFIRMED SUPPORT BREAK"
    if last["ema_fast"] > last["ema_slow"] and price > last["ema_slow"]:
        return "BULLISH MOMENTUM BUILDS"
    if last["ema_fast"] < last["ema_slow"] and price < last["ema_slow"]:
        return "BEARS STILL CONTROL THE TREND"
    return "PRICE ENTERS A DECISION ZONE"


def _price_label(ax, value: float, label: str, color: str) -> None:
    ax.text(
        0.992,
        value,
        f" {label}  {value:,.2f} ",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="center",
        color=TEXT,
        fontsize=9,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": color, "edgecolor": color, "alpha": 0.88},
        zorder=8,
    )


def generate_chart(df, symbol: str, timeframe: str, levels: dict, signal: dict | None = None) -> str:
    """Renders the standard support/resistance chart, or — when `signal` is
    given ({"direction", "entry", "stop", "targets": [prices, nearest to
    farthest]}) — the same chart with entry/stop/scaled-target lines for a
    just-opened hypothetical scenario instead."""
    os.makedirs(config.CHART_DIR, exist_ok=True)
    suffix = "_signal" if signal else ""
    out_path = os.path.join(config.CHART_DIR, f"{symbol}_{timeframe}{suffix}.png")
    display = df.tail(config.CHART_DISPLAY_CANDLES).copy()
    source = df.attrs.get("source", "Binance")
    quote = "USD" if source == "CoinGecko" else "USDT"

    add_plots = [
        mpf.make_addplot(display["bb_upper"], color=BB, width=0.7, linestyle="--", alpha=0.65),
        mpf.make_addplot(display["bb_lower"], color=BB, width=0.7, linestyle="--", alpha=0.65),
        mpf.make_addplot(display["ema_fast"], color=EMA_FAST, width=1.4),
        mpf.make_addplot(display["ema_slow"], color=EMA_SLOW, width=1.4),
    ]

    if signal:
        targets = signal["targets"]
        # Nearest target drawn thickest, farthest thinnest, so the ladder reads
        # left-to-right/near-to-far at a glance even before the labels are read.
        target_widths = [1.25 - 0.25 * (i / max(len(targets) - 1, 1)) for i in range(len(targets))]
        hlines = {
            "hlines": [signal["entry"], *targets, signal["stop"]],
            "colors": [CURRENT, *([SUPPORT] * len(targets)), RESISTANCE],
            "linestyle": ["--", *(["-"] * len(targets)), "-"],
            "linewidths": [1.3, *target_widths, 1.15],
            "alpha": 0.9,
        }
    else:
        hlines = {
            "hlines": [levels["support"], levels["resistance"]],
            "colors": [SUPPORT, RESISTANCE],
            "linestyle": ["-", "-"],
            "linewidths": [1.15, 1.15],
            "alpha": 0.9,
        }

    has_volume = bool(display["Volume"].abs().sum())
    rsi_panel = 2 if has_volume else 1
    add_plots.append(
        mpf.make_addplot(display["rsi"], panel=rsi_panel, color=EMA_SLOW, width=1.3)
    )
    plot_kwargs = {"volume": has_volume}
    plot_kwargs["panel_ratios"] = (4.2, 1.0, 1.6) if has_volume else (4.2, 1.6)
    fig, axes = mpf.plot(
        display,
        type="candle",
        style=_style(),
        addplot=add_plots,
        hlines=hlines,
        **plot_kwargs,
        figratio=(16, 10),
        figscale=1.15,
        datetime_format="%b %d",
        xrotation=0,
        returnfig=True,
        warn_too_much_data=500,
    )

    price_ax = axes[0]
    volume_ax = axes[2] if has_volume else None
    rsi_ax = axes[rsi_panel * 2]
    fig.patch.set_facecolor(BACKGROUND)
    price_ax.set_facecolor(PANEL)
    rsi_ax.set_facecolor(PANEL)
    if volume_ax is not None:
        volume_ax.set_facecolor(PANEL)

    x_values = range(len(display))
    price_ax.fill_between(
        x_values,
        display["bb_lower"].to_numpy(dtype=float),
        display["bb_upper"].to_numpy(dtype=float),
        color=BB,
        alpha=0.055,
        zorder=0,
    )

    zone_width = float(levels.get("zone_width", 0))
    if zone_width:
        price_ax.axhspan(levels["support"] - zone_width, levels["support"] + zone_width,
                         color=SUPPORT, alpha=0.055, zorder=0)
        price_ax.axhspan(levels["resistance"] - zone_width, levels["resistance"] + zone_width,
                         color=RESISTANCE, alpha=0.055, zorder=0)

    last = display.iloc[-1]
    current_price = float(last["Close"])
    if signal:
        _price_label(price_ax, signal["entry"], "ENTRY", CURRENT)
        placed = [signal["entry"]]
        targets = signal["targets"]
        y_min, y_max = price_ax.get_ylim()
        min_gap = 0.035 * (y_max - y_min if y_max > y_min else 1.0)
        for i, target_price in enumerate(targets):
            is_final = i == len(targets) - 1
            label = f"TP{i + 1}" + (" (FINAL)" if is_final else "")
            # Nearest and final targets are always labeled; a crowded middle
            # rung is skipped (the hline still renders) rather than overlap
            # an already-placed label.
            if i == 0 or is_final or all(abs(target_price - p) >= min_gap for p in placed):
                _price_label(price_ax, target_price, label, SUPPORT)
                placed.append(target_price)
        _price_label(price_ax, signal["stop"], "STOP", RESISTANCE)
    else:
        _price_label(price_ax, levels["support"], "SUPPORT", SUPPORT)
        _price_label(price_ax, levels["resistance"], "RESISTANCE", RESISTANCE)

    rsi_values = display["rsi"].to_numpy(dtype=float)
    rsi_ax.set_ylim(0, 100)
    rsi_ax.set_yticks([30, 50, 70])
    rsi_ax.fill_between(x_values, rsi_values, 0, color=EMA_SLOW, alpha=0.16, zorder=1)
    rsi_ax.axhline(70, color=DOWN, linewidth=0.9, linestyle="--", alpha=0.55)
    rsi_ax.axhline(30, color=UP, linewidth=0.9, linestyle="--", alpha=0.55)
    rsi_ax.set_ylabel(f"RSI {config.RSI_PERIOD}", color=MUTED, fontsize=9, fontweight="bold", labelpad=14)
    last_rsi = float(last["rsi"])
    rsi_ax.text(
        0.992, last_rsi, f" {last_rsi:.1f} ",
        transform=rsi_ax.get_yaxis_transform(),
        ha="right", va="center", color=TEXT, fontsize=8.5, fontweight="bold",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": EMA_SLOW, "edgecolor": EMA_SLOW, "alpha": 0.88},
        zorder=8,
    )

    legend = [
        Line2D([0], [0], color=EMA_FAST, lw=2, label=f"EMA {config.EMA_FAST}"),
        Line2D([0], [0], color=EMA_SLOW, lw=2, label=f"EMA {config.EMA_SLOW}"),
        Line2D([0], [0], color=BB, lw=1, ls="--", label="Bollinger Bands"),
    ]
    price_ax.legend(
        handles=legend,
        loc="upper left",
        ncol=3,
        frameon=False,
        labelcolor=MUTED,
        fontsize=9,
        handlelength=2.2,
        columnspacing=1.4,
    )

    price_ax.set_ylabel(f"PRICE · {quote}", color=MUTED, fontsize=9, fontweight="bold", labelpad=14)
    if volume_ax is not None:
        volume_ax.set_ylabel("VOLUME", color=MUTED, fontsize=9, fontweight="bold", labelpad=14)
        volume_ax.yaxis.set_major_formatter(FuncFormatter(_compact_number))
    for ax in filter(None, (price_ax, volume_ax, rsi_ax)):
        ax.grid(True, alpha=0.45)
        ax.tick_params(axis="both", labelsize=9)

    close_time = last.get("CloseTime", display.index[-1])
    close_time = str(close_time).replace("+00:00", "")[:16]
    stats = (
        f"{_technical_headline(last, levels)}     CLOSE  {current_price:,.2f}     "
        f"RSI  {last['rsi']:.1f}     "
        f"ATR  {last['atr_pct']:.2f}%     "
        f"BB WIDTH  {last['bb_width_pct']:.2f}%"
    )

    fig.subplots_adjust(left=0.075, right=0.93, top=0.82, bottom=0.12, hspace=0.08)
    fig.text(0.075, 0.94, f"{_display_symbol(symbol, quote)}  ·  {timeframe.upper()}",
             color=TEXT, fontsize=22, fontweight="bold", ha="left", va="center")
    fig.text(0.075, 0.895, stats, color=MUTED, fontsize=10, ha="left", va="center")
    badge = "MARKET SNAPSHOT"
    badge_color = EMA_FAST
    if signal:
        is_long = signal["direction"] == "long"
        badge = "LONG SCENARIO" if is_long else "SHORT SCENARIO"
        badge_color = SUPPORT if is_long else RESISTANCE
    fig.text(0.925, 0.94, badge, color=badge_color, fontsize=9,
             fontweight="bold", ha="right", va="center")
    fig.text(0.925, 0.895, f"CLOSED {close_time} UTC  ·  {source.upper()}",
             color=MUTED, fontsize=9, ha="right", va="center")
    fig.text(0.925, 0.035,
             f"Levels: {levels.get('lookback', len(df))} candles · pivot clusters",
             color=MUTED, fontsize=8, ha="right")

    fig.savefig(out_path, dpi=config.CHART_DPI, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _clean_axes(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.grid(axis="y", color=GRID, linestyle="--", alpha=0.55)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _ranked_change_chart(snapshot: list[dict], change_key: str, out_name: str,
                         badge: str, unit_label: str) -> str:
    """Shared horizontal-bar ranking used by both the daily map and weekly recap."""
    os.makedirs(config.CHART_DIR, exist_ok=True)
    out_path = os.path.join(config.CHART_DIR, out_name)
    rows = sorted(snapshot, key=lambda row: row[change_key])
    tickers = [row["ticker"] for row in rows]
    changes = [row[change_key] for row in rows]
    colors = [UP if change >= 0 else DOWN for change in changes]
    advancing = sum(change > 0 for change in changes)
    strongest = max(snapshot, key=lambda row: row[change_key])
    weakest = min(snapshot, key=lambda row: row[change_key])
    if advancing >= 8:
        pool = ["BROAD CRYPTO RALLY: BUYERS TAKE CONTROL",
                "GREEN ACROSS THE BOARD AS BUYERS STEP IN",
                "BROAD STRENGTH: MOST MAJORS ADVANCE"]
    elif advancing <= 2:
        pool = ["MARKET SELLOFF: RED ACROSS THE BOARD",
                "BROAD WEAKNESS: SELLERS TAKE CONTROL",
                "RISK COMES OFF ACROSS THE MAJORS"]
    else:
        pool = ["CRYPTO MARKET SPLITS: MOMENTUM IS SHIFTING",
                "A SPLIT TAPE: ROTATION UNDER THE SURFACE",
                "MIXED MARKET: LEADERS AND LAGGARDS DIVERGE"]
    headline = _rotating_choice(pool, date.today().toordinal())

    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor=BACKGROUND)
    _clean_axes(ax)
    bars = ax.barh(tickers, changes, color=colors, height=0.62, alpha=0.92)
    ax.axvline(0, color=MUTED, linewidth=0.9, alpha=0.7)
    padding = max(max(abs(value) for value in changes) * 0.035, 0.08)
    low = min(min(changes), 0)
    high = max(max(changes), 0)
    span = max(high - low, 1)
    ax.set_xlim(low - span * 0.24, high + span * 0.28)
    for bar, row in zip(bars, rows):
        value = row[change_key]
        ax.text(
            value + (padding if value >= 0 else -padding),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}%  ·  {_format_price(row['price'])}",
            ha="left" if value >= 0 else "right",
            va="center",
            color=TEXT,
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xlabel(f"{unit_label} CHANGE", color=MUTED, fontsize=9, fontweight="bold", labelpad=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.1f}%"))
    fig.subplots_adjust(left=0.10, right=0.91, top=0.79, bottom=0.13)
    fig.text(0.10, 0.92, headline, color=TEXT, fontsize=23, fontweight="bold")
    fig.text(0.10, 0.865,
             f"{advancing} UP · {len(snapshot) - advancing} DOWN · LEADER {strongest['ticker']} {strongest[change_key]:+.2f}% · WEAKEST {weakest['ticker']} {weakest[change_key]:+.2f}%",
             color=MUTED, fontsize=10, fontweight="bold")
    fig.text(0.90, 0.92, badge, color=EMA_FAST, fontsize=10,
             fontweight="bold", ha="right")
    fig.text(0.90, 0.865, "SOURCE  ·  COINGECKO", color=MUTED, fontsize=9, ha="right")
    fig.savefig(out_path, dpi=config.CHART_DPI, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_market_map_chart(snapshot: list[dict]) -> str:
    """Render one compact daily dashboard instead of ten separate posts."""
    return _ranked_change_chart(snapshot, "change_24h", "daily_market_map.png",
                                "DAILY SNAPSHOT", "24-HOUR")


def generate_weekly_recap_chart(snapshot: list[dict]) -> str:
    """Same ranking, over the trailing 7 days, for the weekly digest."""
    return _ranked_change_chart(snapshot, "change_7d", "weekly_recap.png",
                                "WEEKLY SNAPSHOT", "7-DAY")


def generate_technical_scoreboard_chart(rows: list[dict]) -> str:
    """RSI and ATR% volatility ranked across the tracked universe, from daily candles."""
    os.makedirs(config.CHART_DIR, exist_ok=True)
    out_path = os.path.join(config.CHART_DIR, "technical_scoreboard.png")

    rsi_rows = sorted(rows, key=lambda row: row["rsi"])
    atr_rows = sorted(rows, key=lambda row: row["atr_pct"])

    def _rsi_color(value: float) -> str:
        if value >= 70:
            return DOWN
        if value <= 30:
            return UP
        return EMA_FAST

    fig, (ax_rsi, ax_atr) = plt.subplots(1, 2, figsize=(12.8, 7.2), facecolor=BACKGROUND)
    for ax in (ax_rsi, ax_atr):
        _clean_axes(ax)

    ax_rsi.barh([row["ticker"] for row in rsi_rows], [row["rsi"] for row in rsi_rows],
                color=[_rsi_color(row["rsi"]) for row in rsi_rows], height=0.62, alpha=0.92)
    ax_rsi.axvline(70, color=DOWN, linewidth=0.9, linestyle="--", alpha=0.55)
    ax_rsi.axvline(30, color=UP, linewidth=0.9, linestyle="--", alpha=0.55)
    ax_rsi.set_xlim(0, 100)
    ax_rsi.set_xlabel("RSI (14)", color=MUTED, fontsize=9, fontweight="bold", labelpad=10)
    for y, row in enumerate(rsi_rows):
        ax_rsi.text(row["rsi"] + 2, y, f"{row['rsi']:.0f}", va="center",
                    color=TEXT, fontsize=9, fontweight="bold")

    max_atr = max((row["atr_pct"] for row in atr_rows), default=1) or 1
    ax_atr.barh([row["ticker"] for row in atr_rows], [row["atr_pct"] for row in atr_rows],
                color=EMA_SLOW, height=0.62, alpha=0.92)
    ax_atr.set_xlabel("ATR % OF PRICE", color=MUTED, fontsize=9, fontweight="bold", labelpad=10)
    for y, row in enumerate(atr_rows):
        ax_atr.text(row["atr_pct"] + max_atr * 0.02, y, f"{row['atr_pct']:.2f}%", va="center",
                    color=TEXT, fontsize=9, fontweight="bold")

    fig.subplots_adjust(left=0.08, right=0.96, top=0.80, bottom=0.10, wspace=0.4)
    fig.text(0.08, 0.92, "TECHNICAL SCOREBOARD", color=TEXT, fontsize=23, fontweight="bold")
    fig.text(0.08, 0.865, "RSI EXTREMES AND VOLATILITY ACROSS THE TRACKED UNIVERSE · DAILY CANDLES",
             color=MUTED, fontsize=10, fontweight="bold")
    fig.text(0.94, 0.92, "WEEKLY DIGEST", color=EMA_FAST, fontsize=10, fontweight="bold", ha="right")
    fig.text(0.94, 0.865, "SOURCE  ·  BINANCE / COINGECKO", color=MUTED, fontsize=9, ha="right")
    fig.savefig(out_path, dpi=config.CHART_DPI, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_market_structure_chart(dominance_history, fear_greed_history) -> str:
    """BTC dominance and Fear & Greed over time, built from daily snapshots since
    CoinGecko's free tier has no historical global market endpoint."""
    os.makedirs(config.CHART_DIR, exist_ok=True)
    out_path = os.path.join(config.CHART_DIR, "market_structure.png")

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2), facecolor=BACKGROUND, sharex=False)
    for ax in axes:
        _clean_axes(ax)

    axes[0].plot(dominance_history.index, dominance_history.values, color=SUPPORT, linewidth=2.1)
    axes[0].fill_between(dominance_history.index, dominance_history.values, dominance_history.min(),
                         color=SUPPORT, alpha=0.08)
    axes[0].set_ylabel("BTC DOMINANCE %", color=MUTED, fontsize=9, fontweight="bold")
    axes[0].text(0.99, 0.88, f"{dominance_history.iloc[-1]:.2f}%",
                 transform=axes[0].transAxes, ha="right", color=TEXT, fontsize=11, fontweight="bold")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    axes[1].plot(fear_greed_history.index, fear_greed_history.values, color=CURRENT, linewidth=2.1)
    axes[1].fill_between(fear_greed_history.index, fear_greed_history.values, 0, color=CURRENT, alpha=0.08)
    axes[1].axhspan(0, 25, color=DOWN, alpha=0.05, zorder=0)
    axes[1].axhspan(75, 100, color=UP, alpha=0.05, zorder=0)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("FEAR & GREED", color=MUTED, fontsize=9, fontweight="bold")
    axes[1].text(0.99, 0.88, f"{fear_greed_history.iloc[-1]:.0f}/100",
                 transform=axes[1].transAxes, ha="right", color=TEXT, fontsize=11, fontweight="bold")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    fig.subplots_adjust(left=0.09, right=0.92, top=0.80, bottom=0.12, hspace=0.30)
    fig.text(0.09, 0.92, "MARKET STRUCTURE", color=TEXT, fontsize=23, fontweight="bold")
    fig.text(0.09, 0.865, f"BTC DOMINANCE AND SENTIMENT · TRAILING {len(dominance_history)} DAYS",
             color=MUTED, fontsize=10, fontweight="bold")
    fig.text(0.91, 0.92, "WEEKLY DIGEST", color=EMA_FAST, fontsize=10, fontweight="bold", ha="right")
    fig.text(0.91, 0.865, "COINGECKO + ALTERNATIVE.ME", color=MUTED, fontsize=9, ha="right")
    fig.savefig(out_path, dpi=config.CHART_DPI, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_liquidity_chart(total_mcap_history, stablecoin_mcap_history) -> str:
    """Total and stablecoin market cap over time — capital in the market vs. on the sidelines."""
    os.makedirs(config.CHART_DIR, exist_ok=True)
    out_path = os.path.join(config.CHART_DIR, "liquidity.png")
    dollar_formatter = FuncFormatter(lambda value, _: f"${_compact_number(value)}")

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2), facecolor=BACKGROUND, sharex=False)
    for ax in axes:
        _clean_axes(ax)

    axes[0].plot(total_mcap_history.index, total_mcap_history.values, color=EMA_FAST, linewidth=2.1)
    axes[0].fill_between(total_mcap_history.index, total_mcap_history.values, total_mcap_history.min(),
                         color=EMA_FAST, alpha=0.08)
    axes[0].set_ylabel("TOTAL MARKET CAP", color=MUTED, fontsize=9, fontweight="bold")
    axes[0].yaxis.set_major_formatter(dollar_formatter)
    axes[0].text(0.99, 0.88, f"${_compact_number(total_mcap_history.iloc[-1])}",
                 transform=axes[0].transAxes, ha="right", color=TEXT, fontsize=11, fontweight="bold")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    axes[1].plot(stablecoin_mcap_history.index, stablecoin_mcap_history.values, color=EMA_SLOW, linewidth=2.1)
    axes[1].fill_between(stablecoin_mcap_history.index, stablecoin_mcap_history.values,
                         stablecoin_mcap_history.min(), color=EMA_SLOW, alpha=0.08)
    axes[1].set_ylabel("STABLECOIN MARKET CAP", color=MUTED, fontsize=9, fontweight="bold")
    axes[1].yaxis.set_major_formatter(dollar_formatter)
    axes[1].text(0.99, 0.88, f"${_compact_number(stablecoin_mcap_history.iloc[-1])}",
                 transform=axes[1].transAxes, ha="right", color=TEXT, fontsize=11, fontweight="bold")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    fig.subplots_adjust(left=0.12, right=0.92, top=0.80, bottom=0.12, hspace=0.30)
    fig.text(0.09, 0.92, "LIQUIDITY", color=TEXT, fontsize=23, fontweight="bold")
    fig.text(0.09, 0.865, f"TOTAL AND STABLECOIN MARKET CAP · TRAILING {len(total_mcap_history)} DAYS",
             color=MUTED, fontsize=10, fontweight="bold")
    fig.text(0.91, 0.92, "WEEKLY DIGEST", color=EMA_FAST, fontsize=10, fontweight="bold", ha="right")
    fig.text(0.91, 0.865, "SOURCE  ·  COINGECKO", color=MUTED, fontsize=9, ha="right")
    fig.savefig(out_path, dpi=config.CHART_DPI, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_pulse_chart(derivatives_rows: list[dict]) -> str:
    """Perpetual funding rate and open interest for the pulse assets."""
    os.makedirs(config.CHART_DIR, exist_ok=True)
    out_path = os.path.join(config.CHART_DIR, "daily_pulse.png")

    height = min(4.9 + 0.5 * len(derivatives_rows), 10.0)
    fig, (ax_funding, ax_oi) = plt.subplots(1, 2, figsize=(12.8, height), facecolor=BACKGROUND)
    for ax in (ax_funding, ax_oi):
        _clean_axes(ax)

    funding_values = [row["funding_rate"] for row in derivatives_rows]
    funding_colors = [UP if value >= 0 else DOWN for value in funding_values]
    ax_funding.barh([row["ticker"] for row in derivatives_rows], funding_values,
                    color=funding_colors, height=0.5, alpha=0.92)
    ax_funding.axvline(0, color=MUTED, linewidth=0.9, alpha=0.7)
    ax_funding.set_xlabel("FUNDING RATE %", color=MUTED, fontsize=9, fontweight="bold", labelpad=10)
    funding_pad = max(max(abs(v) for v in funding_values) * 0.10, 0.002)
    funding_low = min(min(funding_values), 0)
    funding_high = max(max(funding_values), 0)
    funding_span = max(funding_high - funding_low, 0.01)
    ax_funding.set_xlim(funding_low - funding_span * 0.35, funding_high + funding_span * 0.35)
    for y, row in enumerate(derivatives_rows):
        value = row["funding_rate"]
        ax_funding.text(value + (funding_pad if value >= 0 else -funding_pad), y, f"{value:+.4f}%",
                        va="center", ha="left" if value >= 0 else "right",
                        color=TEXT, fontsize=9, fontweight="bold")

    oi_values = [row["open_interest"] for row in derivatives_rows]
    max_oi = max(oi_values) if oi_values else 1
    ax_oi.barh([row["ticker"] for row in derivatives_rows], oi_values,
               color=EMA_SLOW, height=0.5, alpha=0.92)
    ax_oi.set_xlabel("OPEN INTEREST (USD)", color=MUTED, fontsize=9, fontweight="bold", labelpad=10)
    ax_oi.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"${_compact_number(value)}"))
    ax_oi.set_xlim(0, max_oi * 1.22)
    for y, row in enumerate(derivatives_rows):
        ax_oi.text(row["open_interest"] + max_oi * 0.02, y, f"${_compact_number(row['open_interest'])}",
                   va="center", color=TEXT, fontsize=9, fontweight="bold")

    fig.subplots_adjust(left=0.10, right=0.95, top=0.78, bottom=0.13, wspace=0.45)
    fig.text(0.10, 0.90, "DERIVATIVES PULSE", color=TEXT, fontsize=23, fontweight="bold")
    fig.text(0.10, 0.845, "PERPETUAL FUNDING AND OPEN INTEREST ACROSS MAJORS",
             color=MUTED, fontsize=10, fontweight="bold")
    fig.text(0.93, 0.90, "DAILY PULSE", color=EMA_FAST, fontsize=10, fontweight="bold", ha="right")
    fig.text(0.93, 0.845, "SOURCE  ·  COINGECKO", color=MUTED, fontsize=9, ha="right")
    fig.savefig(out_path, dpi=config.CHART_DPI, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _series_change(series, periods: int) -> float:
    if len(series) <= periods:
        return 0.0
    return float((series.iloc[-1] / series.iloc[-1 - periods] - 1) * 100)


def generate_macro_chart(macro: dict) -> str:
    """Render S&P 500 and the Fed broad-dollar index with BTC dominance."""
    os.makedirs(config.CHART_DIR, exist_ok=True)
    out_path = os.path.join(config.CHART_DIR, "macro_close.png")
    sp500 = macro["sp500"]
    dollar = macro["dollar"]
    sp_change = _series_change(sp500, 5)
    dollar_change = _series_change(dollar, 5)
    if sp_change > 0 and dollar_change < 0:
        pool = ["RISK-ON MOMENTUM BUILDS", "STOCKS FIRM AS THE DOLLAR EASES",
                "A RISK-FRIENDLY BACKDROP TAKES SHAPE"]
    elif sp_change < 0 and dollar_change > 0:
        pool = ["DOLLAR PRESSURE HITS RISK ASSETS", "A FIRMER DOLLAR WEIGHS ON RISK",
                "RISK TURNS DEFENSIVE AS THE DOLLAR CLIMBS"]
    elif sp_change > 0 and dollar_change > 0:
        pool = ["STOCKS AND THE DOLLAR CLIMB TOGETHER", "RISK AND THE DOLLAR RISE IN TANDEM"]
    elif sp_change < 0 and dollar_change < 0:
        pool = ["STOCKS AND THE DOLLAR EASE TOGETHER", "RISK AND THE DOLLAR SLIP IN TANDEM"]
    else:
        pool = ["WALL STREET AND THE DOLLAR DIVERGE", "A MIXED, RANGE-BOUND MACRO TAPE"]
    headline = _rotating_choice(pool, int(sp500.index[-1].toordinal()))

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2), facecolor=BACKGROUND, sharex=False)
    for ax in axes:
        _clean_axes(ax)

    axes[0].plot(sp500.index, sp500.values, color=EMA_FAST, linewidth=2.1)
    axes[0].fill_between(sp500.index, sp500.values, sp500.min(), color=EMA_FAST, alpha=0.08)
    axes[0].set_ylabel("S&P 500", color=MUTED, fontsize=9, fontweight="bold")
    axes[0].text(0.99, 0.88, f"{sp500.iloc[-1]:,.2f}  ·  5D {sp_change:+.2f}%",
                 transform=axes[0].transAxes, ha="right", color=TEXT, fontsize=11, fontweight="bold")
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    axes[1].plot(dollar.index, dollar.values, color="#F4C95D", linewidth=2.1)
    axes[1].fill_between(dollar.index, dollar.values, dollar.min(), color="#F4C95D", alpha=0.08)
    axes[1].set_ylabel("BROAD USD", color=MUTED, fontsize=9, fontweight="bold")
    axes[1].text(0.99, 0.88, f"{dollar.iloc[-1]:,.2f}  ·  5D {dollar_change:+.2f}%",
                 transform=axes[1].transAxes, ha="right", color=TEXT, fontsize=11, fontweight="bold")
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    fig.subplots_adjust(left=0.09, right=0.92, top=0.78, bottom=0.12, hspace=0.30)
    fig.text(0.09, 0.92, headline, color=TEXT, fontsize=23, fontweight="bold")
    fig.text(0.09, 0.865,
             f"S&P 500 {sp_change:+.2f}% (5D) · BROAD USD {dollar_change:+.2f}% (5D) · CRYPTO LIQUIDITY CONTEXT",
             color=MUTED, fontsize=10, fontweight="bold")
    fig.text(0.91, 0.92, f"BTC DOMINANCE  {macro['btc_dominance']:.2f}%",
             color=SUPPORT, fontsize=12, fontweight="bold", ha="right")
    fig.text(0.91, 0.865, "FRED + COINGECKO", color=MUTED, fontsize=9, ha="right")

    extra_stats = []
    if macro.get("vix") is not None:
        extra_stats.append(f"VIX {macro['vix']:.2f}")
    if macro.get("yield_10y") is not None and macro.get("yield_2y") is not None:
        spread = macro["yield_10y"] - macro["yield_2y"]
        extra_stats.append(f"10Y-2Y {spread:+.2f}pp")
    if macro.get("btc_sp_corr") is not None:
        extra_stats.append(f"BTC-SPX CORR {macro['btc_sp_corr']:+.2f}")
    if macro.get("btc_dollar_corr") is not None:
        extra_stats.append(f"BTC-USD CORR {macro['btc_dollar_corr']:+.2f}")
    if extra_stats:
        fig.text(0.09, 0.035, "  ·  ".join(extra_stats), color=MUTED, fontsize=9, ha="left")
        fig.text(0.91, 0.035, f"{config.CORRELATION_WINDOW}D CORRELATION WINDOW",
                 color=MUTED, fontsize=8, ha="right")

    fig.savefig(out_path, dpi=config.CHART_DPI, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_signal_scorecard_chart(scorecard: dict, closed: list[dict]) -> str:
    """Track record of the hypothetical scenarios: one bar per closed scenario,
    coloured by whether it reached target (win) or hit stop (loss), sized by its
    R multiple. Empty state when nothing has closed yet."""
    os.makedirs(config.CHART_DIR, exist_ok=True)
    out_path = os.path.join(config.CHART_DIR, "signal_scorecard.png")

    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor=BACKGROUND)
    _clean_axes(ax)

    if not closed:
        ax.axis("off")
        ax.text(0.5, 0.5, "AWAITING THE FIRST CLOSED SCENARIO",
                transform=ax.transAxes, ha="center", va="center",
                color=MUTED, fontsize=18, fontweight="bold")
        subtitle = f"{scorecard['open_count']} HYPOTHETICAL SCENARIO(S) CURRENTLY OPEN"
    else:
        tickers = [row["ticker"] for row in closed]
        r_values = [float(row.get("r_multiple", 0.0)) for row in closed]
        colors = [UP if row.get("outcome") == "target" else DOWN for row in closed]
        y_values = list(range(len(closed)))
        ax.barh(y_values, r_values, color=colors, height=0.6, alpha=0.92)
        ax.set_yticks(y_values)
        ax.set_yticklabels(tickers)
        ax.axvline(0, color=MUTED, linewidth=0.9, alpha=0.7)
        pad = max(max(abs(v) for v in r_values) * 0.04, 0.05)
        low = min(min(r_values), 0)
        high = max(max(r_values), 0)
        span = max(high - low, 1)
        ax.set_xlim(low - span * 0.20, high + span * 0.26)
        for y, value in zip(y_values, r_values):
            ax.text(value + (pad if value >= 0 else -pad), y, f"{value:+.2f}R",
                    va="center", ha="left" if value >= 0 else "right",
                    color=TEXT, fontsize=10, fontweight="bold")
        ax.set_xlabel("RESULT IN R  ·  REWARD REALIZED PER UNIT OF RISK",
                      color=MUTED, fontsize=9, fontweight="bold", labelpad=12)
        win_rate = scorecard["win_rate_pct"] or 0.0
        subtitle = (f"{scorecard['closed_count']} CLOSED · {scorecard['win_count']} REACHED TARGET "
                    f"({win_rate:.0f}%) · AVG {scorecard['avg_r']:+.2f}R · {scorecard['open_count']} OPEN")

    fig.subplots_adjust(left=0.10, right=0.94, top=0.80, bottom=0.13)
    fig.text(0.10, 0.92, "HYPOTHETICAL SIGNAL SCORECARD", color=TEXT, fontsize=23, fontweight="bold")
    fig.text(0.10, 0.865, subtitle, color=MUTED, fontsize=10, fontweight="bold")
    fig.text(0.90, 0.92, "EDUCATIONAL TRACK RECORD", color=EMA_FAST, fontsize=10, fontweight="bold", ha="right")
    fig.text(0.90, 0.865, "HYPOTHETICAL SCENARIOS · NOT ADVICE", color=MUTED, fontsize=9, ha="right")
    fig.savefig(out_path, dpi=config.CHART_DPI, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_watchlist_chart(rows: list[dict]) -> str:
    """Which assets sit closest to a key level right now — the shorter the bar,
    the nearer the decision point. Coloured by whether the near level is
    resistance overhead or support beneath."""
    os.makedirs(config.CHART_DIR, exist_ok=True)
    out_path = os.path.join(config.CHART_DIR, "watchlist.png")

    ordered = sorted(rows, key=lambda row: abs(row["distance_pct"]), reverse=True)
    tickers = [row["ticker"] for row in ordered]
    distances = [abs(row["distance_pct"]) for row in ordered]
    colors = [RESISTANCE if row["level_type"] == "resistance" else SUPPORT for row in ordered]

    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor=BACKGROUND)
    _clean_axes(ax)
    ax.barh(tickers, distances, color=colors, height=0.62, alpha=0.92)
    max_distance = max(distances) if distances else 1
    ax.set_xlim(0, max_distance * 1.4)
    ax.set_xlabel("DISTANCE TO NEAREST KEY LEVEL (%)", color=MUTED, fontsize=9, fontweight="bold", labelpad=12)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}%"))
    for y, row in enumerate(ordered):
        side = "resistance overhead" if row["level_type"] == "resistance" else "support beneath"
        ax.text(abs(row["distance_pct"]) + max_distance * 0.02, y,
                f"{side}  ·  RSI {row['rsi']:.0f}  ·  {row['trend'].split(':', 1)[0]}",
                va="center", color=TEXT, fontsize=9, fontweight="bold")

    fig.subplots_adjust(left=0.10, right=0.95, top=0.80, bottom=0.13)
    fig.text(0.10, 0.92, "WHAT TO WATCH THIS WEEK", color=TEXT, fontsize=23, fontweight="bold")
    fig.text(0.10, 0.865, "ASSETS CLOSEST TO A DECISION LEVEL · CONFIRMED DAILY CANDLES",
             color=MUTED, fontsize=10, fontweight="bold")
    fig.text(0.93, 0.92, "WEEK AHEAD", color=EMA_FAST, fontsize=10, fontweight="bold", ha="right")
    fig.text(0.93, 0.865, "SOURCE · BINANCE / COINGECKO", color=MUTED, fontsize=9, ha="right")
    fig.savefig(out_path, dpi=config.CHART_DPI, facecolor=BACKGROUND, bbox_inches="tight")
    plt.close(fig)
    return out_path
