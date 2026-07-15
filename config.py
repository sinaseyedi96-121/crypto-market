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
DEEP_DIVE_TIMEFRAME = "1d"
CANDLE_LIMIT = 300                         # enough history for indicators and long-term levels

# Tickers checked for funding rate / long-short ratio in the daily pulse post.
PULSE_ASSETS = ["BTC", "ETH", "SOL"]
TRENDING_COINS_LIMIT = 5

# Day of week (Monday=0 .. Sunday=6) the weekly digest is allowed to publish.
WEEKLY_DIGEST_WEEKDAY = 6

# ---- Indicators ----
RSI_PERIOD = 14
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

# ---- DeepSeek (text generation) ----
DEEPSEEK_MODEL = "deepseek-v4-pro"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MAX_TOKENS = 500

# ---- Telegram ----
TELEGRAM_CAPTION_LIMIT = 1024              # Telegram's hard cap on photo captions

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
DISCLAIMER = "\n\n⚠️ Educational content, not financial advice."
