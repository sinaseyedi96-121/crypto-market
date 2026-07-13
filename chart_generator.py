"""
Renders a real candlestick chart from real OHLC data using mplfinance.
No AI, no image generation model — this is plain data plotting, so every
number on the chart is exactly what the market did.
"""

import os
import mplfinance as mpf
import config


def generate_chart(df, symbol: str, timeframe: str, levels: dict) -> str:
    os.makedirs(config.CHART_DIR, exist_ok=True)
    out_path = os.path.join(config.CHART_DIR, f"{symbol}_{timeframe}.png")

    add_plots = [
        mpf.make_addplot(df["ema_fast"], color="#3498db", width=1.1),
        mpf.make_addplot(df["ema_slow"], color="#9b59b6", width=1.1),
    ]

    hlines = dict(
        hlines=[levels["support"], levels["resistance"]],
        colors=["#2ecc71", "#e74c3c"],
        linestyle="-.",
        linewidths=1.2,
    )

    mpf.plot(
        df,
        type="candle",
        style="charles",
        title=f"{symbol} — {timeframe}",
        addplot=add_plots,
        hlines=hlines,
        volume=True,
        figratio=(12, 7),
        savefig=dict(fname=out_path, dpi=150, bbox_inches="tight"),
    )
    return out_path
