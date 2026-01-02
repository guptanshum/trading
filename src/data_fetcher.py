"""
Data fetcher module for Gold & Silver ETF data from Yahoo Finance
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
import pandas as pd
import yfinance as yf
from dataclasses import dataclass

from config import (
    GOLD_TICKER, SILVER_TICKER,
    HISTORICAL_PERIOD, INTRADAY_PERIOD,
    INTRADAY_INTERVAL, DAILY_INTERVAL
)

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Container for market data"""
    gold: pd.DataFrame
    silver: pd.DataFrame
    timestamp: datetime


class DataFetcher:
    """Fetches and manages ETF data from Yahoo Finance"""

    def __init__(
        self,
        gold_ticker: str = GOLD_TICKER,
        silver_ticker: str = SILVER_TICKER
    ):
        self.gold_ticker = gold_ticker
        self.silver_ticker = silver_ticker
        self._gold_info: Optional[dict] = None
        self._silver_info: Optional[dict] = None

    def fetch_historical_daily(
        self,
        period: str = HISTORICAL_PERIOD
    ) -> MarketData:
        """
        Fetch historical daily data for both ETFs

        Args:
            period: Time period (e.g., '1y', '2y', '5y', 'max')

        Returns:
            MarketData with daily OHLCV data
        """
        logger.info(f"Fetching {period} historical daily data...")

        gold_data = yf.download(
            self.gold_ticker,
            period=period,
            interval=DAILY_INTERVAL,
            progress=False
        )

        silver_data = yf.download(
            self.silver_ticker,
            period=period,
            interval=DAILY_INTERVAL,
            progress=False
        )

        # Flatten multi-index columns if present
        gold_data = self._flatten_columns(gold_data)
        silver_data = self._flatten_columns(silver_data)

        logger.info(
            f"Fetched {len(gold_data)} gold and {len(silver_data)} silver daily bars"
        )

        return MarketData(
            gold=gold_data,
            silver=silver_data,
            timestamp=datetime.now()
        )

    def fetch_intraday(
        self,
        period: str = INTRADAY_PERIOD,
        interval: str = INTRADAY_INTERVAL
    ) -> MarketData:
        """
        Fetch intraday data for day trading

        Args:
            period: Time period (e.g., '1d', '5d', '7d')
            interval: Candle interval (e.g., '1m', '5m', '15m')

        Returns:
            MarketData with intraday OHLCV data
        """
        logger.info(f"Fetching {period} intraday data at {interval} interval...")

        gold_data = yf.download(
            self.gold_ticker,
            period=period,
            interval=interval,
            progress=False
        )

        silver_data = yf.download(
            self.silver_ticker,
            period=period,
            interval=interval,
            progress=False
        )

        # Flatten multi-index columns if present
        gold_data = self._flatten_columns(gold_data)
        silver_data = self._flatten_columns(silver_data)

        logger.info(
            f"Fetched {len(gold_data)} gold and {len(silver_data)} silver intraday bars"
        )

        return MarketData(
            gold=gold_data,
            silver=silver_data,
            timestamp=datetime.now()
        )

    def fetch_latest_price(self) -> Tuple[float, float]:
        """
        Fetch the latest prices for both ETFs

        Returns:
            Tuple of (gold_price, silver_price)
        """
        gold = yf.Ticker(self.gold_ticker)
        silver = yf.Ticker(self.silver_ticker)

        gold_price = gold.info.get('regularMarketPrice', 0)
        silver_price = silver.info.get('regularMarketPrice', 0)

        return gold_price, silver_price

    def fetch_live_data(self) -> MarketData:
        """
        Fetch the most recent data point for live tracking

        Returns:
            MarketData with latest bars
        """
        return self.fetch_intraday(period="1d", interval="1m")

    def get_ticker_info(self, ticker: str) -> dict:
        """Get detailed info about a ticker"""
        return yf.Ticker(ticker).info

    def get_gold_info(self) -> dict:
        """Get cached gold ETF info"""
        if self._gold_info is None:
            self._gold_info = self.get_ticker_info(self.gold_ticker)
        return self._gold_info

    def get_silver_info(self) -> dict:
        """Get cached silver ETF info"""
        if self._silver_info is None:
            self._silver_info = self.get_ticker_info(self.silver_ticker)
        return self._silver_info

    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Flatten multi-index columns from yfinance"""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

    def calculate_gold_silver_ratio(self, data: MarketData) -> pd.Series:
        """
        Calculate the Gold/Silver ratio from price data

        The ratio shows how many ounces of silver it takes to buy one ounce of gold.
        Historical average is around 60-80.

        Args:
            data: MarketData with aligned gold and silver prices

        Returns:
            Series with Gold/Silver ratio
        """
        # Align the data
        aligned = pd.DataFrame({
            'gold': data.gold['Close'],
            'silver': data.silver['Close']
        }).dropna()

        # Calculate ratio (Note: ETF prices aren't actual oz prices,
        # but ratio still shows relative value)
        ratio = aligned['gold'] / aligned['silver']
        ratio.name = 'GS_Ratio'

        return ratio

    def get_combined_data(self, data: MarketData) -> pd.DataFrame:
        """
        Combine gold and silver data into a single DataFrame

        Args:
            data: MarketData object

        Returns:
            DataFrame with prefixed columns (gold_*, silver_*)
        """
        gold_df = data.gold.copy()
        silver_df = data.silver.copy()

        # Rename columns with prefixes
        gold_df.columns = [f'gold_{col.lower()}' for col in gold_df.columns]
        silver_df.columns = [f'silver_{col.lower()}' for col in silver_df.columns]

        # Combine on index
        combined = pd.concat([gold_df, silver_df], axis=1)

        # Calculate ratio
        if 'gold_close' in combined.columns and 'silver_close' in combined.columns:
            combined['gs_ratio'] = combined['gold_close'] / combined['silver_close']

        return combined.dropna()


def fetch_and_prepare_data(
    gold_ticker: str = GOLD_TICKER,
    silver_ticker: str = SILVER_TICKER,
    period: str = HISTORICAL_PERIOD
) -> pd.DataFrame:
    """
    Convenience function to fetch and prepare data for analysis

    Args:
        gold_ticker: Gold ETF ticker symbol
        silver_ticker: Silver ETF ticker symbol
        period: Historical period to fetch

    Returns:
        Combined DataFrame ready for analysis
    """
    fetcher = DataFetcher(gold_ticker, silver_ticker)
    data = fetcher.fetch_historical_daily(period)
    return fetcher.get_combined_data(data)


if __name__ == "__main__":
    # Test the data fetcher
    logging.basicConfig(level=logging.INFO)

    fetcher = DataFetcher()

    # Fetch historical data
    print("Fetching historical data...")
    hist_data = fetcher.fetch_historical_daily("1y")
    print(f"Gold data shape: {hist_data.gold.shape}")
    print(f"Silver data shape: {hist_data.silver.shape}")

    # Get combined data
    combined = fetcher.get_combined_data(hist_data)
    print(f"\nCombined data shape: {combined.shape}")
    print(f"Columns: {list(combined.columns)}")
    print(f"\nLast 5 rows:\n{combined.tail()}")

    # Fetch intraday data
    print("\nFetching intraday data...")
    intraday = fetcher.fetch_intraday("1d", "1m")
    print(f"Gold intraday bars: {len(intraday.gold)}")
    print(f"Silver intraday bars: {len(intraday.silver)}")
