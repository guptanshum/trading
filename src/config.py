"""
Configuration settings for Gold & Silver ETF Trading App
"""
from dataclasses import dataclass, field
from typing import Optional
import os

# ETF Tickers
GOLD_TICKER = "GLD"  # SPDR Gold Shares
SILVER_TICKER = "SIVR"  # abrdn Physical Silver Shares ETF

# Alternative tickers if needed
ALT_GOLD_TICKERS = ["IAU", "GLDM", "SGOL"]
ALT_SILVER_TICKERS = ["SLV", "PSLV"]

# Data settings
HISTORICAL_PERIOD = "2y"  # 2 years of historical data
INTRADAY_PERIOD = "5d"  # 5 days for intraday data
INTRADAY_INTERVAL = "1m"  # 1-minute candles for day trading
DAILY_INTERVAL = "1d"

# Refresh settings
LIVE_REFRESH_SECONDS = 60  # Refresh every 1 minute

# Trading hours (US Eastern Time)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Shadow trading settings
INITIAL_CAPITAL = 100_000.0  # $100,000 virtual capital
MAX_POSITION_SIZE_PCT = 0.25  # Max 25% of portfolio in single position
STOP_LOSS_PCT = 0.02  # 2% stop loss
TAKE_PROFIT_PCT = 0.05  # 5% take profit

# Risk management
MAX_DAILY_LOSS_PCT = 0.03  # Max 3% daily loss
MAX_DRAWDOWN_PCT = 0.10  # Max 10% drawdown

# Technical indicator settings
@dataclass
class IndicatorSettings:
    # Moving averages
    sma_fast: int = 9
    sma_slow: int = 21
    ema_fast: int = 12
    ema_slow: int = 26

    # RSI
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR for volatility
    atr_period: int = 14

# Strategy settings
@dataclass
class StrategySettings:
    # Trend following
    trend_ma_cross_enabled: bool = True
    trend_breakout_enabled: bool = True
    trend_momentum_enabled: bool = True

    # Mean reversion
    reversion_rsi_enabled: bool = True
    reversion_bb_enabled: bool = True
    reversion_ratio_enabled: bool = True

    # Gold/Silver ratio thresholds
    gs_ratio_mean: float = 80.0  # Historical average around 80
    gs_ratio_std: float = 10.0  # Standard deviation
    gs_ratio_buy_silver: float = 90.0  # Buy silver when ratio > 90
    gs_ratio_buy_gold: float = 70.0  # Buy gold when ratio < 70

# Database settings
DATABASE_PATH = "data/trading.db"

# Avanza API settings (fill in when ready)
@dataclass
class AvanzaSettings:
    username: Optional[str] = None
    password: Optional[str] = None
    totp_secret: Optional[str] = None
    account_id: Optional[str] = None  # ISK account ID
    enabled: bool = False

    @classmethod
    def from_env(cls):
        """Load Avanza credentials from environment variables"""
        return cls(
            username=os.getenv("AVANZA_USERNAME"),
            password=os.getenv("AVANZA_PASSWORD"),
            totp_secret=os.getenv("AVANZA_TOTP_SECRET"),
            account_id=os.getenv("AVANZA_ACCOUNT_ID"),
            enabled=os.getenv("AVANZA_ENABLED", "false").lower() == "true"
        )

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Dashboard settings
DASHBOARD_TITLE = "Gold & Silver ETF Trading"
DASHBOARD_REFRESH_SECONDS = 60
CHART_HEIGHT = 400
