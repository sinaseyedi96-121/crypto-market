"""
All adjustable settings live here. Change values below to tune behavior —
nothing else in the codebase should need editing for day-to-day tweaks.
"""

# ---- Markets ----
SYMBOLS = ["BTCUSDT", "ETHUSDT"]          # Binance symbol names
TIMEFRAMES = ["4h", "1d"]                  # Binance kline intervals to post
CANDLE_LIMIT = 200                         # candles pulled per request (need enough for EMA50/BB20 warmup)

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
SWING_LOOKBACK = 20                        # candles used to compute recent support/resistance zone

# ---- Follow-up posts ----
# If price moves this many % beyond a previously-flagged zone, post a follow-up
# reply on the original message describing what happened.
FOLLOWUP_THRESHOLD_PCT = 1.5

# ---- DeepSeek (text generation) ----
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MAX_TOKENS = 350

# ---- Telegram ----
TELEGRAM_CAPTION_LIMIT = 1024              # Telegram's hard cap on photo captions

# ---- Sentiment (optional, free, no key needed) ----
FEAR_GREED_URL = "https://api.alternative.me/fng/"

# ---- Paths ----
CHART_DIR = "charts"
STATE_FILE = "post_history.json"

# ---- Mandatory footer appended in code (not left to the model to remember) ----
DISCLAIMER = "\n\n⚠️ Educational content, not financial advice."
