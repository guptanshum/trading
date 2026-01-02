"""
Technical indicators module for trading analysis
"""
import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass

from config import IndicatorSettings


@dataclass
class IndicatorResult:
    """Container for indicator calculation results"""
    name: str
    values: pd.Series
    signal: Optional[pd.Series] = None  # Buy/Sell signals
    upper_band: Optional[pd.Series] = None
    lower_band: Optional[pd.Series] = None


class TechnicalIndicators:
    """Calculate technical indicators for trading analysis"""

    def __init__(self, settings: Optional[IndicatorSettings] = None):
        self.settings = settings or IndicatorSettings()

    def sma(self, data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()

    def ema(self, data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()

    def rsi(self, data: pd.Series, period: Optional[int] = None) -> pd.Series:
        """
        Relative Strength Index

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        period = period or self.settings.rsi_period

        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def macd(
        self,
        data: pd.Series,
        fast: Optional[int] = None,
        slow: Optional[int] = None,
        signal: Optional[int] = None
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        fast = fast or self.settings.macd_fast
        slow = slow or self.settings.macd_slow
        signal = signal or self.settings.macd_signal

        ema_fast = self.ema(data, fast)
        ema_slow = self.ema(data, slow)

        macd_line = ema_fast - ema_slow
        signal_line = self.ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def bollinger_bands(
        self,
        data: pd.Series,
        period: Optional[int] = None,
        std_dev: Optional[float] = None
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands

        Returns:
            Tuple of (Middle Band/SMA, Upper Band, Lower Band)
        """
        period = period or self.settings.bb_period
        std_dev = std_dev or self.settings.bb_std

        middle = self.sma(data, period)
        std = data.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return middle, upper, lower

    def atr(self, high: pd.Series, low: pd.Series, close: pd.Series,
            period: Optional[int] = None) -> pd.Series:
        """
        Average True Range - volatility indicator

        True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        """
        period = period or self.settings.atr_period

        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    def stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator

        %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = SMA of %K

        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k = ((close - lowest_low) / (highest_high - lowest_low)) * 100
        d = k.rolling(window=d_period).mean()

        return k, d

    def williams_r(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Williams %R

        %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        wr = ((highest_high - close) / (highest_high - lowest_low)) * -100

        return wr

    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume

        OBV = Previous OBV + Current Volume (if close > prev close)
        OBV = Previous OBV - Current Volume (if close < prev close)
        """
        direction = np.sign(close.diff())
        direction.iloc[0] = 0

        obv = (direction * volume).cumsum()

        return obv

    def vwap(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Volume Weighted Average Price

        VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
        """
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()

        return vwap

    def calculate_all(self, df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
        """
        Calculate all indicators for a DataFrame with OHLCV data

        Args:
            df: DataFrame with Open, High, Low, Close, Volume columns
            prefix: Column prefix (e.g., 'gold_' or 'silver_')

        Returns:
            DataFrame with all indicator columns added
        """
        result = df.copy()

        # Get column names
        close_col = f'{prefix}close' if prefix else 'Close'
        high_col = f'{prefix}high' if prefix else 'High'
        low_col = f'{prefix}low' if prefix else 'Low'
        volume_col = f'{prefix}volume' if prefix else 'Volume'

        close = result[close_col]
        high = result[high_col]
        low = result[low_col]

        # Moving Averages
        result[f'{prefix}sma_fast'] = self.sma(close, self.settings.sma_fast)
        result[f'{prefix}sma_slow'] = self.sma(close, self.settings.sma_slow)
        result[f'{prefix}ema_fast'] = self.ema(close, self.settings.ema_fast)
        result[f'{prefix}ema_slow'] = self.ema(close, self.settings.ema_slow)

        # RSI
        result[f'{prefix}rsi'] = self.rsi(close)

        # MACD
        macd_line, signal_line, histogram = self.macd(close)
        result[f'{prefix}macd'] = macd_line
        result[f'{prefix}macd_signal'] = signal_line
        result[f'{prefix}macd_hist'] = histogram

        # Bollinger Bands
        bb_mid, bb_upper, bb_lower = self.bollinger_bands(close)
        result[f'{prefix}bb_mid'] = bb_mid
        result[f'{prefix}bb_upper'] = bb_upper
        result[f'{prefix}bb_lower'] = bb_lower
        result[f'{prefix}bb_width'] = (bb_upper - bb_lower) / bb_mid

        # ATR
        result[f'{prefix}atr'] = self.atr(high, low, close)

        # Stochastic
        stoch_k, stoch_d = self.stochastic(high, low, close)
        result[f'{prefix}stoch_k'] = stoch_k
        result[f'{prefix}stoch_d'] = stoch_d

        # Williams %R
        result[f'{prefix}williams_r'] = self.williams_r(high, low, close)

        # Volume indicators (if volume available)
        if volume_col in result.columns:
            volume = result[volume_col]
            result[f'{prefix}obv'] = self.obv(close, volume)
            result[f'{prefix}vwap'] = self.vwap(high, low, close, volume)

        return result


def add_indicators_to_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add all indicators to combined ETF data

    Args:
        df: DataFrame with gold_*, silver_*, gold_bear_*, silver_bear_* columns

    Returns:
        DataFrame with indicators added for all assets
    """
    indicators = TechnicalIndicators()

    # Bull ETFs
    if 'gold_close' in df.columns:
        df = indicators.calculate_all(df, prefix='gold_')

    if 'silver_close' in df.columns:
        df = indicators.calculate_all(df, prefix='silver_')

    # Bear ETFs
    if 'gold_bear_close' in df.columns:
        df = indicators.calculate_all(df, prefix='gold_bear_')

    if 'silver_bear_close' in df.columns:
        df = indicators.calculate_all(df, prefix='silver_bear_')

    return df


if __name__ == "__main__":
    # Test the indicators
    import yfinance as yf

    # Fetch test data
    print("Fetching test data...")
    gold = yf.download("GLD", period="6mo", progress=False)

    if isinstance(gold.columns, pd.MultiIndex):
        gold.columns = gold.columns.get_level_values(0)

    # Calculate indicators
    indicators = TechnicalIndicators()

    print("\nCalculating indicators...")
    result = indicators.calculate_all(gold, prefix='')

    print(f"\nResult shape: {result.shape}")
    print(f"Columns: {list(result.columns)}")
    print(f"\nLast row indicators:")
    last = result.iloc[-1]
    for col in ['Close', 'sma_fast', 'sma_slow', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'atr']:
        if col in last.index:
            print(f"  {col}: {last[col]:.2f}")
