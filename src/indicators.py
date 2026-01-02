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

    def adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index - measures trend strength

        Returns:
            Tuple of (ADX, +DI, -DI)
        """
        # Calculate True Range
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)

        # Smoothed averages
        atr = true_range.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx, plus_di, minus_di

    def mfi(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Money Flow Index - volume-weighted RSI

        MFI = 100 - (100 / (1 + Money Flow Ratio))
        """
        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume

        # Positive and negative money flow
        price_change = typical_price.diff()
        positive_flow = np.where(price_change > 0, raw_money_flow, 0)
        negative_flow = np.where(price_change < 0, raw_money_flow, 0)

        positive_flow = pd.Series(positive_flow, index=close.index)
        negative_flow = pd.Series(negative_flow, index=close.index)

        # Rolling sums
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        # Money Flow Ratio and MFI
        mf_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + mf_ratio))

        return mfi

    def cci(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Commodity Channel Index - identifies cyclical trends

        CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
        """
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_dev = typical_price.rolling(window=period).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        cci = (typical_price - sma) / (0.015 * mean_dev)

        return cci

    def parabolic_sar(
        self,
        high: pd.Series,
        low: pd.Series,
        af_start: float = 0.02,
        af_increment: float = 0.02,
        af_max: float = 0.2
    ) -> pd.Series:
        """
        Parabolic SAR - trend following indicator

        Returns SAR values
        """
        length = len(high)
        sar = pd.Series(index=high.index, dtype=float)
        trend = pd.Series(index=high.index, dtype=int)
        af = af_start
        ep = low.iloc[0]  # Extreme point

        # Initialize
        sar.iloc[0] = high.iloc[0]
        trend.iloc[0] = -1  # Start with downtrend

        for i in range(1, length):
            if trend.iloc[i-1] == 1:  # Uptrend
                sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
                sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])

                if low.iloc[i] < sar.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = ep
                    ep = low.iloc[i]
                    af = af_start
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep:
                        ep = high.iloc[i]
                        af = min(af + af_increment, af_max)
            else:  # Downtrend
                sar.iloc[i] = sar.iloc[i-1] + af * (ep - sar.iloc[i-1])
                sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])

                if high.iloc[i] > sar.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = ep
                    ep = high.iloc[i]
                    af = af_start
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep:
                        ep = low.iloc[i]
                        af = min(af + af_increment, af_max)

        return sar

    def ichimoku(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        tenkan_period: int = 9,
        kijun_period: int = 26,
        senkou_b_period: int = 52
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Ichimoku Cloud indicator

        Returns:
            Tuple of (Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, Chikou Span)
        """
        # Tenkan-sen (Conversion Line)
        tenkan = (high.rolling(window=tenkan_period).max() +
                  low.rolling(window=tenkan_period).min()) / 2

        # Kijun-sen (Base Line)
        kijun = (high.rolling(window=kijun_period).max() +
                 low.rolling(window=kijun_period).min()) / 2

        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)

        # Senkou Span B (Leading Span B)
        senkou_b = ((high.rolling(window=senkou_b_period).max() +
                     low.rolling(window=senkou_b_period).min()) / 2).shift(kijun_period)

        # Chikou Span (Lagging Span)
        chikou = close.shift(-kijun_period)

        return tenkan, kijun, senkou_a, senkou_b, chikou

    def roc(self, data: pd.Series, period: int = 12) -> pd.Series:
        """
        Rate of Change - momentum indicator

        ROC = ((Price - Price n periods ago) / Price n periods ago) * 100
        """
        return ((data - data.shift(period)) / data.shift(period)) * 100

    def momentum(self, data: pd.Series, period: int = 10) -> pd.Series:
        """
        Momentum indicator

        Momentum = Price - Price n periods ago
        """
        return data - data.shift(period)

    def pivot_points(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Calculate Pivot Points and Support/Resistance levels

        Returns:
            Tuple of (Pivot, R1, R2, S1, S2)
        """
        pivot = (high.shift(1) + low.shift(1) + close.shift(1)) / 3
        r1 = 2 * pivot - low.shift(1)
        s1 = 2 * pivot - high.shift(1)
        r2 = pivot + (high.shift(1) - low.shift(1))
        s2 = pivot - (high.shift(1) - low.shift(1))

        return pivot, r1, r2, s1, s2

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
            result[f'{prefix}mfi'] = self.mfi(high, low, close, volume)

        # ADX - Trend Strength
        adx, plus_di, minus_di = self.adx(high, low, close)
        result[f'{prefix}adx'] = adx
        result[f'{prefix}plus_di'] = plus_di
        result[f'{prefix}minus_di'] = minus_di

        # CCI - Commodity Channel Index
        result[f'{prefix}cci'] = self.cci(high, low, close)

        # Parabolic SAR
        result[f'{prefix}psar'] = self.parabolic_sar(high, low)

        # Ichimoku Cloud
        tenkan, kijun, senkou_a, senkou_b, chikou = self.ichimoku(high, low, close)
        result[f'{prefix}ichimoku_tenkan'] = tenkan
        result[f'{prefix}ichimoku_kijun'] = kijun
        result[f'{prefix}ichimoku_senkou_a'] = senkou_a
        result[f'{prefix}ichimoku_senkou_b'] = senkou_b

        # Momentum indicators
        result[f'{prefix}roc'] = self.roc(close)
        result[f'{prefix}momentum'] = self.momentum(close)

        # Pivot Points
        pivot, r1, r2, s1, s2 = self.pivot_points(high, low, close)
        result[f'{prefix}pivot'] = pivot
        result[f'{prefix}r1'] = r1
        result[f'{prefix}r2'] = r2
        result[f'{prefix}s1'] = s1
        result[f'{prefix}s2'] = s2

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
