"""
All adjustable settings live here. Change values below to tune behavior —
nothing else in the codebase should need editing for day-to-day tweaks.
"""

# ---- Markets ----
# Current top non-stablecoin universe. CoinGecko IDs provide a free fallback
# for assets whose USDT pair is unavailable from Binance.
ASSETS = [
    {"symbol": "BTCUSDT", "ticker": "BTC", "name": "Bitcoin", "coingecko_id": "bitcoin"},
    {"symbol": "ETHUSDT", "ticker": "ETH", "name": "Ethereum", "coingecko_id": "ethereum"},
    {"symbol": "BNBUSDT", "ticker": "BNB", "name": "BNB", "coingecko_id": "binancecoin"},
    {"symbol": "XRPUSDT", "ticker": "XRP", "name": "XRP", "coingecko_id": "ripple"},
    {"symbol": "SOLUSDT", "ticker": "SOL", "name": "Solana", "coingecko_id": "solana"},
    {"symbol": "TRXUSDT", "ticker": "TRX", "name": "TRON", "coingecko_id": "tron"},
    {"symbol": "HYPEUSDT", "ticker": "HYPE", "name": "Hyperliquid", "coingecko_id": "hyperliquid"},
    {"symbol": "DOGEUSDT", "ticker": "DOGE", "name": "Dogecoin", "coingecko_id": "dogecoin"},
    {"symbol": "LEOUSDT", "ticker": "LEO", "name": "UNUS SED LEO", "coingecko_id": "leo-token"},
    {"symbol": "ZECUSDT", "ticker": "ZEC", "name": "Zcash", "coingecko_id": "zcash"},
]

# Monday through Sunday. Each deep-dive is one Telegram album containing two
# charts, so the channel stays useful without becoming noisy.
DEEP_DIVE_ROTATION = [
    ("BTC", "SOL"),
    ("ETH", "XRP"),
    ("BNB", "DOGE"),
    ("TRX", "HYPE"),
    ("LEO", "ZEC"),
    ("BTC", "ETH"),
    ("SOL", "HYPE"),
]
DEEP_DIVE_TIMEFRAME = "1d"                 # long view: weeks/months
# Near-term companion chart posted alongside the daily one per asset, so a
# reader sees both "the next few hours/days" and "the next few weeks/months"
# in the same album. Reuses the interval the alert scanner already runs on.
DEEP_DIVE_SHORT_TIMEFRAME = "4h"
CANDLE_LIMIT = 500                         # enough history for indicators, long-term levels, and a
                                            # deep signal-target search (see find_extended_targets)

# Tickers checked for funding rate / open interest in the daily pulse post.
# Same tracked universe as ASSETS, minus LEO (an exchange token with no
# perpetual futures market, so CoinGecko's /derivatives has no data for it).
PULSE_ASSETS = ["BTC", "ETH", "BNB", "XRP", "SOL", "TRX", "HYPE", "DOGE", "ZEC"]
TRENDING_COINS_LIMIT = 5

# Day of week (Monday=0 .. Sunday=6) each weekly post is allowed to publish.
WEEKLY_DIGEST_WEEKDAY = 6                   # Sunday: backward-looking recap
SIGNAL_SCORECARD_WEEKDAY = 5               # Saturday: hypothetical-signal track record
WHAT_TO_WATCH_WEEKDAY = 0                  # Monday: forward-looking week-ahead post

# ---- Indicators ----
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70                        # RSI at/above this reads as stretched to the upside
RSI_OVERSOLD = 30                          # RSI at/below this reads as stretched to the downside
EMA_FAST = 20
EMA_SLOW = 50
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
VOLATILITY_LOOKBACK = 100                  # history used to contextualize ATR and BB width
TREND_SLOPE_LOOKBACK = 5                   # candles used to confirm EMA direction

# Support/resistance is based on repeated pivot touches, not only the highest
# and lowest wick in a short window. 180 candles is ~30 days on 4h and ~6
# months on 1d.
LEVEL_LOOKBACK = 180
PIVOT_WINDOW = 3                           # candles required on each side of a pivot
MIN_LEVEL_TOUCHES = 2
LEVEL_CLUSTER_ATR_MULTIPLIER = 0.40        # merge nearby pivots into one level

# ---- Follow-up posts ----
# If price moves this many % beyond a previously-flagged zone, post a follow-up
# reply on the original message describing what happened.
FOLLOWUP_THRESHOLD_PCT = 1.5
MAX_EVENT_ALERTS_PER_DAY = 2

# ---- What to watch (weekly forward-looking post, see README) ----
# An asset is flagged as "one to watch" when the confirmed close sits within
# this % of a key support/resistance level (a decision point that a break or
# bounce would resolve) or when RSI is stretched. Built entirely from data the
# pipeline already fetches — no forward calendar dependency.
WATCH_LEVEL_PROXIMITY_PCT = 4.0
WATCH_MAX_ITEMS = 6

# ---- Hypothetical trade posts (educational only, see README) ----
# One open hypothetical scenario per asset at a time; entry is the confirmed
# deep-dive close, the initial stop is the nearby support/resistance level, and
# targets are the pivot levels found across all fetched history — nearest to
# farthest, capped at SIGNAL_MAX_TARGETS — whose farthest member still clears
# MIN_SIGNAL_RISK_REWARD (see indicators.find_extended_targets). If nothing
# clears it, no signal is opened for that asset that day. As each target is
# reached the position is treated as partially closed and the stop ladders up
# (to breakeven after the first target, then to the prior target after each
# one after that) so the remaining runner can never give back more than it has
# already banked.
MAX_OPEN_SIGNALS_PER_ASSET = 1
MIN_SIGNAL_RISK_REWARD = 3.0
SIGNAL_MAX_TARGETS = 4
# Illustrative position sizing shown alongside each scenario: risking this % of
# a fixed hypothetical example account against the entry-to-stop distance, with
# the resulting leverage capped here. A worked example of a standard
# risk-based sizing convention, not sized advice for the reader — see
# SIGNAL_DISCLAIMERS.
SIGNAL_RISK_PER_TRADE_PCT = 1.0
SIGNAL_MAX_LEVERAGE = 5.0
SIGNAL_DISCLAIMERS = {
    "en": (
        "\n\n⚠️ Hypothetical educational scenario, not a trade recommendation. "
        "Position size/leverage figures are worked against a fixed hypothetical "
        "example account, not a recommendation for your own capital. "
        "Not financial advice."
    ),
    "fa": (
        "\n\n⚠️ سناریوی آموزشی و فرضی؛ نه توصیه‌ی معاملاتی است و نه مشاوره‌ی مالی. "
        "اعداد اندازه‌ی پوزیشن و اهرم بر پایه‌ی یک حساب فرضی و ثابت محاسبه شده‌اند و "
        "توصیه‌ای برای سرمایه‌ی شما نیستند."
    ),
}
SIGNAL_DISCLAIMER = SIGNAL_DISCLAIMERS["en"]   # default/back-compat (English)

# ---- Content review (weekly OpenAI critique, see README) ----
POSTS_LOG_FILE = "posts_log.jsonl"
REVIEW_LOOKBACK_DAYS = 7
REVIEW_REPORT_FILE = "review_report.md"
REVIEWER_MAX_TOKENS = 8000
# The reviewer may only propose full-file replacements for these two files —
# prompt wording and tunable constants, never the trading/indicator logic.
REVIEWER_EDITABLE_FILES = ["narrative_generator.py", "config.py"]

# ---- OpenAI (text generation) ----
OPENAI_MODEL = "gpt-5.6-luna"
# Luna is a reasoning model: hidden reasoning tokens are drawn from the SAME
# completion budget as the visible caption. Keep reasoning light for these
# short captions and give the budget generous headroom, so reasoning can never
# starve the visible text into the deterministic fallback (the old 500 cap with
# default "medium" effort could silently truncate a caption to nothing).
OPENAI_REASONING_EFFORT = "low"
OPENAI_MAX_TOKENS = 2000

# ---- Telegram ----
TELEGRAM_CAPTION_LIMIT = 1024              # Telegram's hard cap on photo captions

# The same bot publishes every post to each channel below. Charts are shared
# (numbers are numbers); only the caption text is localized per channel.
# The English channel is primary and required; the Farsi channel is optional —
# if its env var is unset (e.g. the secret hasn't been added yet) it is skipped
# with a warning so the primary channel keeps posting. The bot must be an admin
# of every channel it posts to.
CHANNELS = [
    {
        "language": "en",
        "env": "TELEGRAM_CHANNEL",
        "name": "To the Moon 🚀",
        "url": "https://t.me/cryptomarket_bit",
        "required": True,
    },
    {
        "language": "fa",
        "env": "TELEGRAM_CHANNEL_FA",
        "name": "تحلیل بازار کریپتو",
        "url": "https://t.me/crypto_market_farsi",
        "required": False,
    },
]
CHANNEL_URL = CHANNELS[0]["url"]           # primary channel, kept for back-compat

# ---- Chart ----
CHART_DISPLAY_CANDLES = 120                # calculate on all candles, render only the recent view
CHART_DPI = 160

# ---- Sentiment (optional, free, no key needed) ----
FEAR_GREED_URL = "https://api.alternative.me/fng/"
COINGECKO_BASE_URL = "https://api.coingecko.com/api/v3"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
MACRO_LOOKBACK = 140
FRED_VIX_SERIES = "VIXCLS"
FRED_10Y_SERIES = "DGS10"
FRED_2Y_SERIES = "DGS2"
CORRELATION_WINDOW = 30                    # trading days used for BTC vs macro rolling correlation

# ---- Weekly digest history (built up daily, no paid historical API needed) ----
DAILY_SNAPSHOT_HISTORY_DAYS = 90

# ---- Paths ----
CHART_DIR = "charts"
STATE_FILE = "post_history.json"

# ---- Mandatory footer appended in code (not left to the model to remember) ----
# A clickable link back to the channel, so a forwarded/reposted caption still
# drives people back to the source. The link text is the channel's display name.
# Rendered via Telegram's HTML parse mode (see telegram_publisher.py), so caption
# bodies are HTML-escaped first. Channel names carry no HTML-special characters.
DISCLAIMERS = {
    channel["language"]: f'\n\n<a href="{channel["url"]}">{channel["name"]}</a>'
    for channel in CHANNELS
}
DISCLAIMER = DISCLAIMERS["en"]             # default/back-compat (English)
