"""
Data fetcher module for Gold & Silver ETF data from Yahoo Finance
Supports both bull (long) and bear (inverse) ETFs
"""
import logging
import os
import glob
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
import time

from config import (
    GOLD_TICKER, SILVER_TICKER,
    GOLD_BEAR_TICKER, SILVER_BEAR_TICKER,
    HISTORICAL_PERIOD, INTRADAY_PERIOD,
    INTRADAY_INTERVAL, DAILY_INTERVAL
)

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Container for market data - supports both bull and bear ETFs"""
    gold: pd.DataFrame
    silver: pd.DataFrame
    gold_bear: pd.DataFrame = field(default_factory=pd.DataFrame)
    silver_bear: pd.DataFrame = field(default_factory=pd.DataFrame)
    timestamp: datetime = field(default_factory=datetime.now)


class DataFetcher:
    """Fetches and manages ETF data from Yahoo Finance"""

    def __init__(
        self,
        gold_ticker: str = GOLD_TICKER,
        silver_ticker: str = SILVER_TICKER,
        gold_bear_ticker: str = GOLD_BEAR_TICKER,
        silver_bear_ticker: str = SILVER_BEAR_TICKER,
        include_bear: bool = True
    ):
        self.gold_ticker = gold_ticker
        self.silver_ticker = silver_ticker
        self.gold_bear_ticker = gold_bear_ticker
        self.silver_bear_ticker = silver_bear_ticker
        self.include_bear = include_bear
        self._ticker_info_cache: Dict[str, dict] = {}

    def _fetch_ticker_data(
        self,
        ticker: str,
        period: str,
        interval: str
    ) -> pd.DataFrame:
        """Fetch data for a single ticker with rate limit handling"""
        try:
            data = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False
            )
            return self._flatten_columns(data)
        except Exception as e:
            logger.warning(f"Failed to fetch {ticker}: {e}")
            return pd.DataFrame()

    def fetch_historical_daily(
        self,
        period: str = HISTORICAL_PERIOD
    ) -> MarketData:
        """
        Fetch historical daily data for all ETFs (bull and bear)

        Args:
            period: Time period (e.g., '1y', '2y', '5y', 'max')

        Returns:
            MarketData with daily OHLCV data for all assets
        """
        logger.info(f"Fetching {period} historical daily data for all ETFs...")

        # Fetch bull ETFs
        gold_data = self._fetch_ticker_data(self.gold_ticker, period, DAILY_INTERVAL)
        time.sleep(0.5)  # Small delay to avoid rate limiting

        silver_data = self._fetch_ticker_data(self.silver_ticker, period, DAILY_INTERVAL)

        # Fetch bear ETFs if enabled
        gold_bear_data = pd.DataFrame()
        silver_bear_data = pd.DataFrame()

        if self.include_bear:
            time.sleep(0.5)
            gold_bear_data = self._fetch_ticker_data(self.gold_bear_ticker, period, DAILY_INTERVAL)
            time.sleep(0.5)
            silver_bear_data = self._fetch_ticker_data(self.silver_bear_ticker, period, DAILY_INTERVAL)

        logger.info(
            f"Fetched: Gold={len(gold_data)}, Silver={len(silver_data)}, "
            f"GoldBear={len(gold_bear_data)}, SilverBear={len(silver_bear_data)} bars"
        )

        return MarketData(
            gold=gold_data,
            silver=silver_data,
            gold_bear=gold_bear_data,
            silver_bear=silver_bear_data,
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
            MarketData with intraday OHLCV data for all assets
        """
        logger.info(f"Fetching {period} intraday data at {interval} interval...")

        gold_data = self._fetch_ticker_data(self.gold_ticker, period, interval)
        time.sleep(0.3)
        silver_data = self._fetch_ticker_data(self.silver_ticker, period, interval)

        gold_bear_data = pd.DataFrame()
        silver_bear_data = pd.DataFrame()

        if self.include_bear:
            time.sleep(0.3)
            gold_bear_data = self._fetch_ticker_data(self.gold_bear_ticker, period, interval)
            time.sleep(0.3)
            silver_bear_data = self._fetch_ticker_data(self.silver_bear_ticker, period, interval)

        logger.info(
            f"Fetched intraday: Gold={len(gold_data)}, Silver={len(silver_data)}, "
            f"GoldBear={len(gold_bear_data)}, SilverBear={len(silver_bear_data)} bars"
        )

        return MarketData(
            gold=gold_data,
            silver=silver_data,
            gold_bear=gold_bear_data,
            silver_bear=silver_bear_data,
            timestamp=datetime.now()
        )

    def fetch_latest_price(self) -> Dict[str, float]:
        """
        Fetch the latest prices for all ETFs

        Returns:
            Dict with prices for all assets
        """
        prices = {}

        try:
            prices['gold'] = yf.Ticker(self.gold_ticker).info.get('regularMarketPrice', 0)
            prices['silver'] = yf.Ticker(self.silver_ticker).info.get('regularMarketPrice', 0)

            if self.include_bear:
                prices['gold_bear'] = yf.Ticker(self.gold_bear_ticker).info.get('regularMarketPrice', 0)
                prices['silver_bear'] = yf.Ticker(self.silver_bear_ticker).info.get('regularMarketPrice', 0)
        except Exception as e:
            logger.warning(f"Error fetching latest prices: {e}")

        return prices

    def fetch_live_data(self) -> MarketData:
        """
        Fetch the most recent data point for live tracking

        Returns:
            MarketData with latest bars
        """
        return self.fetch_intraday(period="1d", interval="1m")

    def get_ticker_info(self, ticker: str) -> dict:
        """Get detailed info about a ticker (cached)"""
        if ticker not in self._ticker_info_cache:
            try:
                self._ticker_info_cache[ticker] = yf.Ticker(ticker).info
            except Exception as e:
                logger.warning(f"Failed to get info for {ticker}: {e}")
                self._ticker_info_cache[ticker] = {}
        return self._ticker_info_cache[ticker]

    def get_gold_info(self) -> dict:
        """Get cached gold ETF info"""
        return self.get_ticker_info(self.gold_ticker)

    def get_silver_info(self) -> dict:
        """Get cached silver ETF info"""
        return self.get_ticker_info(self.silver_ticker)

    def get_gold_bear_info(self) -> dict:
        """Get cached gold bear ETF info"""
        return self.get_ticker_info(self.gold_bear_ticker)

    def get_silver_bear_info(self) -> dict:
        """Get cached silver bear ETF info"""
        return self.get_ticker_info(self.silver_bear_ticker)

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
        Combine all ETF data into a single DataFrame

        Args:
            data: MarketData object

        Returns:
            DataFrame with prefixed columns (gold_*, silver_*, gold_bear_*, silver_bear_*)
        """
        dfs_to_combine = []

        # Bull ETFs
        if len(data.gold) > 0:
            gold_df = data.gold.copy()
            gold_df.columns = [f'gold_{col.lower()}' for col in gold_df.columns]
            dfs_to_combine.append(gold_df)

        if len(data.silver) > 0:
            silver_df = data.silver.copy()
            silver_df.columns = [f'silver_{col.lower()}' for col in silver_df.columns]
            dfs_to_combine.append(silver_df)

        # Bear ETFs
        if len(data.gold_bear) > 0:
            gold_bear_df = data.gold_bear.copy()
            gold_bear_df.columns = [f'gold_bear_{col.lower()}' for col in gold_bear_df.columns]
            dfs_to_combine.append(gold_bear_df)

        if len(data.silver_bear) > 0:
            silver_bear_df = data.silver_bear.copy()
            silver_bear_df.columns = [f'silver_bear_{col.lower()}' for col in silver_bear_df.columns]
            dfs_to_combine.append(silver_bear_df)

        if not dfs_to_combine:
            return pd.DataFrame()

        # Combine on index
        combined = pd.concat(dfs_to_combine, axis=1)

        # Calculate ratios
        if 'gold_close' in combined.columns and 'silver_close' in combined.columns:
            combined['gs_ratio'] = combined['gold_close'] / combined['silver_close']

        # Calculate bull vs bear ratios (useful for divergence analysis)
        if 'gold_close' in combined.columns and 'gold_bear_close' in combined.columns:
            combined['gold_bull_bear_ratio'] = combined['gold_close'] / combined['gold_bear_close']

        if 'silver_close' in combined.columns and 'silver_bear_close' in combined.columns:
            combined['silver_bull_bear_ratio'] = combined['silver_close'] / combined['silver_bear_close']

        return combined.dropna()


def get_data_file_path() -> str:
    """Get the path to the stored historical data file"""
    # Look for most recent historical data file
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    # Find the most recent historical data file
    files = glob.glob(os.path.join(data_dir, 'historical_data_*.csv'))
    if files:
        return max(files)  # Most recent by filename
    return os.path.join(data_dir, 'historical_data.csv')


def load_stored_data() -> pd.DataFrame:
    """
    Load historical data from stored CSV file

    Returns:
        DataFrame with historical data, or empty DataFrame if not found
    """
    file_path = get_data_file_path()
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
    except Exception as e:
        logger.warning(f"Failed to load stored data: {e}")
    return pd.DataFrame()


def save_data_to_file(df: pd.DataFrame, filename: str = None) -> str:
    """
    Save DataFrame to CSV file

    Args:
        df: DataFrame to save
        filename: Optional filename, defaults to historical_data_YYYYMMDD.csv

    Returns:
        Path to saved file
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    if filename is None:
        filename = f"historical_data_{datetime.now().strftime('%Y%m%d')}.csv"

    file_path = os.path.join(data_dir, filename)
    df.to_csv(file_path, index=True, index_label='Date')
    logger.info(f"Saved {len(df)} rows to {file_path}")
    return file_path


def fetch_and_prepare_data(
    gold_ticker: str = GOLD_TICKER,
    silver_ticker: str = SILVER_TICKER,
    gold_bear_ticker: str = GOLD_BEAR_TICKER,
    silver_bear_ticker: str = SILVER_BEAR_TICKER,
    period: str = HISTORICAL_PERIOD,
    include_bear: bool = True,
    use_stored_data: bool = True
) -> pd.DataFrame:
    """
    Fetch and prepare data using hybrid approach:
    1. Load stored historical data as base
    2. Fetch only recent data from Yahoo to update
    3. Fall back to stored data if rate limited

    Args:
        gold_ticker: Gold ETF ticker symbol
        silver_ticker: Silver ETF ticker symbol
        gold_bear_ticker: Gold bear ETF ticker symbol
        silver_bear_ticker: Silver bear ETF ticker symbol
        period: Historical period to fetch (used as fallback)
        include_bear: Whether to include bear ETFs
        use_stored_data: Whether to use hybrid approach with stored data

    Returns:
        Combined DataFrame ready for analysis with all assets
    """
    stored_df = pd.DataFrame()

    # Step 1: Load stored historical data
    if use_stored_data:
        stored_df = load_stored_data()
        if not stored_df.empty:
            logger.info(f"Loaded stored data: {stored_df.index[0]} to {stored_df.index[-1]}")

    # Step 2: Try to fetch recent data from Yahoo
    fetcher = DataFetcher(
        gold_ticker=gold_ticker,
        silver_ticker=silver_ticker,
        gold_bear_ticker=gold_bear_ticker,
        silver_bear_ticker=silver_bear_ticker,
        include_bear=include_bear
    )

    try:
        # If we have stored data, only fetch last 7 days to update
        if not stored_df.empty:
            fetch_period = "7d"
            logger.info(f"Fetching recent {fetch_period} data to update stored data...")
        else:
            fetch_period = period
            logger.info(f"No stored data, fetching full {fetch_period} from Yahoo...")

        data = fetcher.fetch_historical_daily(fetch_period)
        fresh_df = fetcher.get_combined_data(data)

        if fresh_df.empty:
            raise ValueError("Empty data received from Yahoo")

        logger.info(f"Fetched fresh data: {len(fresh_df)} rows")

        # Step 3: Merge stored and fresh data
        if not stored_df.empty and not fresh_df.empty:
            # Combine: use stored data + update with fresh data
            # Fresh data takes priority for overlapping dates
            combined = pd.concat([stored_df, fresh_df])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
            logger.info(f"Combined data: {len(combined)} rows ({combined.index[0]} to {combined.index[-1]})")
            return combined
        else:
            return fresh_df

    except Exception as e:
        error_msg = str(e).lower()
        if "rate" in error_msg or "429" in error_msg or "too many" in error_msg:
            logger.warning(f"Rate limited by Yahoo Finance: {e}")
        else:
            logger.error(f"Error fetching data: {e}")

        # Step 4: Fall back to stored data
        if not stored_df.empty:
            logger.info("Falling back to stored historical data")
            return stored_df
        else:
            logger.error("No stored data available as fallback")
            return pd.DataFrame()


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
