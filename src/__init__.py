"""
Gold & Silver ETF Trading Application

A comprehensive trading analysis and shadow trading system for GLD and SIVR ETFs.
"""

from .config import (
    GOLD_TICKER,
    SILVER_TICKER,
    INITIAL_CAPITAL,
    IndicatorSettings,
    StrategySettings,
    AvanzaSettings
)

from .data_fetcher import DataFetcher, MarketData, fetch_and_prepare_data
from .indicators import TechnicalIndicators, add_indicators_to_data
from .analysis import StatisticalAnalysis, analyze_market_data
from .strategies import (
    Signal,
    SignalType,
    Asset,
    BaseStrategy,
    MACrossoverStrategy,
    RSIMeanReversionStrategy,
    BollingerBandStrategy,
    MACDStrategy,
    GoldSilverRatioStrategy,
    CompositeStrategy,
    create_default_strategies
)
from .trading_engine import ShadowTradingEngine, Trade, Position
from .avanza_api import AvanzaClient, get_avanza_client

__version__ = "1.0.0"
__all__ = [
    # Config
    "GOLD_TICKER",
    "SILVER_TICKER",
    "INITIAL_CAPITAL",
    "IndicatorSettings",
    "StrategySettings",
    "AvanzaSettings",
    # Data
    "DataFetcher",
    "MarketData",
    "fetch_and_prepare_data",
    # Indicators
    "TechnicalIndicators",
    "add_indicators_to_data",
    # Analysis
    "StatisticalAnalysis",
    "analyze_market_data",
    # Strategies
    "Signal",
    "SignalType",
    "Asset",
    "BaseStrategy",
    "MACrossoverStrategy",
    "RSIMeanReversionStrategy",
    "BollingerBandStrategy",
    "MACDStrategy",
    "GoldSilverRatioStrategy",
    "CompositeStrategy",
    "create_default_strategies",
    # Trading
    "ShadowTradingEngine",
    "Trade",
    "Position",
    # Avanza
    "AvanzaClient",
    "get_avanza_client",
]
