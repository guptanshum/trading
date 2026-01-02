"""
Trading strategies module for Gold/Silver ETF trading
"""
import pandas as pd
import numpy as np
from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from config import IndicatorSettings, StrategySettings


class SignalType(Enum):
    """Trading signal types"""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


class Asset(Enum):
    """Tradeable assets"""
    GOLD = "gold"
    SILVER = "silver"


@dataclass
class Signal:
    """Trading signal with metadata"""
    timestamp: datetime
    asset: Asset
    signal_type: SignalType
    strategy: str
    price: float
    confidence: float  # 0-1 confidence score
    reason: str
    indicators: dict = field(default_factory=dict)


@dataclass
class StrategyResult:
    """Result from strategy execution"""
    signals: List[Signal]
    data: pd.DataFrame
    metrics: dict


class BaseStrategy:
    """Base class for trading strategies"""

    def __init__(self, name: str):
        self.name = name

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """Generate trading signals - to be implemented by subclasses"""
        raise NotImplementedError

    def backtest(self, df: pd.DataFrame) -> dict:
        """Backtest the strategy on historical data"""
        signals = self.generate_signals(df)
        # Simple backtest logic
        return {
            'total_signals': len(signals),
            'buy_signals': len([s for s in signals if s.signal_type.value > 0]),
            'sell_signals': len([s for s in signals if s.signal_type.value < 0])
        }


class MACrossoverStrategy(BaseStrategy):
    """Moving Average Crossover Strategy"""

    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 21,
        asset: Asset = Asset.GOLD
    ):
        super().__init__("MA_Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.asset = asset

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        signals = []
        prefix = f"{self.asset.value}_"

        sma_fast_col = f"{prefix}sma_fast"
        sma_slow_col = f"{prefix}sma_slow"
        close_col = f"{prefix}close"

        if sma_fast_col not in df.columns or sma_slow_col not in df.columns:
            return signals

        # Calculate crossover points
        df = df.copy()
        df['fast_above'] = df[sma_fast_col] > df[sma_slow_col]
        df['crossover'] = df['fast_above'].diff()

        for idx in df.index:
            row = df.loc[idx]

            if pd.isna(row['crossover']):
                continue

            if row['crossover'] == True:  # Bullish crossover
                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=self.asset,
                    signal_type=SignalType.BUY,
                    strategy=self.name,
                    price=row[close_col],
                    confidence=0.7,
                    reason=f"Fast MA ({self.fast_period}) crossed above Slow MA ({self.slow_period})",
                    indicators={
                        'fast_ma': row[sma_fast_col],
                        'slow_ma': row[sma_slow_col]
                    }
                ))
            elif row['crossover'] == False:  # Bearish crossover
                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=self.asset,
                    signal_type=SignalType.SELL,
                    strategy=self.name,
                    price=row[close_col],
                    confidence=0.7,
                    reason=f"Fast MA ({self.fast_period}) crossed below Slow MA ({self.slow_period})",
                    indicators={
                        'fast_ma': row[sma_fast_col],
                        'slow_ma': row[sma_slow_col]
                    }
                ))

        return signals


class RSIMeanReversionStrategy(BaseStrategy):
    """RSI-based Mean Reversion Strategy"""

    def __init__(
        self,
        rsi_period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
        asset: Asset = Asset.GOLD
    ):
        super().__init__("RSI_MeanReversion")
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold
        self.asset = asset

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        signals = []
        prefix = f"{self.asset.value}_"

        rsi_col = f"{prefix}rsi"
        close_col = f"{prefix}close"

        if rsi_col not in df.columns:
            return signals

        prev_rsi = None
        for idx in df.index:
            row = df.loc[idx]
            rsi = row[rsi_col]

            if pd.isna(rsi):
                continue

            # Oversold bounce (RSI crosses above oversold level)
            if prev_rsi is not None and prev_rsi < self.oversold and rsi >= self.oversold:
                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=self.asset,
                    signal_type=SignalType.BUY,
                    strategy=self.name,
                    price=row[close_col],
                    confidence=0.6 + (self.oversold - prev_rsi) / 100,
                    reason=f"RSI bounced from oversold ({prev_rsi:.1f} -> {rsi:.1f})",
                    indicators={'rsi': rsi}
                ))

            # Overbought reversal (RSI crosses below overbought level)
            if prev_rsi is not None and prev_rsi > self.overbought and rsi <= self.overbought:
                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=self.asset,
                    signal_type=SignalType.SELL,
                    strategy=self.name,
                    price=row[close_col],
                    confidence=0.6 + (prev_rsi - self.overbought) / 100,
                    reason=f"RSI fell from overbought ({prev_rsi:.1f} -> {rsi:.1f})",
                    indicators={'rsi': rsi}
                ))

            prev_rsi = rsi

        return signals


class BollingerBandStrategy(BaseStrategy):
    """Bollinger Band Bounce/Breakout Strategy"""

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        asset: Asset = Asset.GOLD
    ):
        super().__init__("BollingerBand")
        self.period = period
        self.std_dev = std_dev
        self.asset = asset

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        signals = []
        prefix = f"{self.asset.value}_"

        bb_upper = f"{prefix}bb_upper"
        bb_lower = f"{prefix}bb_lower"
        bb_mid = f"{prefix}bb_mid"
        close_col = f"{prefix}close"

        if bb_upper not in df.columns:
            return signals

        prev_close = None
        for idx in df.index:
            row = df.loc[idx]
            close = row[close_col]
            upper = row[bb_upper]
            lower = row[bb_lower]

            if pd.isna(upper) or pd.isna(lower):
                continue

            # Price bounces off lower band
            if prev_close is not None and prev_close <= lower and close > lower:
                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=self.asset,
                    signal_type=SignalType.BUY,
                    strategy=self.name,
                    price=close,
                    confidence=0.65,
                    reason=f"Price bounced off lower Bollinger Band ({lower:.2f})",
                    indicators={
                        'bb_upper': upper,
                        'bb_lower': lower,
                        'bb_mid': row[bb_mid]
                    }
                ))

            # Price bounces off upper band
            if prev_close is not None and prev_close >= upper and close < upper:
                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=self.asset,
                    signal_type=SignalType.SELL,
                    strategy=self.name,
                    price=close,
                    confidence=0.65,
                    reason=f"Price rejected at upper Bollinger Band ({upper:.2f})",
                    indicators={
                        'bb_upper': upper,
                        'bb_lower': lower,
                        'bb_mid': row[bb_mid]
                    }
                ))

            prev_close = close

        return signals


class MACDStrategy(BaseStrategy):
    """MACD Crossover Strategy"""

    def __init__(self, asset: Asset = Asset.GOLD):
        super().__init__("MACD")
        self.asset = asset

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        signals = []
        prefix = f"{self.asset.value}_"

        macd_col = f"{prefix}macd"
        signal_col = f"{prefix}macd_signal"
        hist_col = f"{prefix}macd_hist"
        close_col = f"{prefix}close"

        if macd_col not in df.columns:
            return signals

        df = df.copy()
        df['macd_above'] = df[macd_col] > df[signal_col]
        df['macd_cross'] = df['macd_above'].diff()

        for idx in df.index:
            row = df.loc[idx]

            if pd.isna(row['macd_cross']):
                continue

            if row['macd_cross'] == True:  # Bullish crossover
                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=self.asset,
                    signal_type=SignalType.BUY,
                    strategy=self.name,
                    price=row[close_col],
                    confidence=0.7,
                    reason="MACD crossed above signal line",
                    indicators={
                        'macd': row[macd_col],
                        'signal': row[signal_col],
                        'histogram': row[hist_col]
                    }
                ))
            elif row['macd_cross'] == False:  # Bearish crossover
                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=self.asset,
                    signal_type=SignalType.SELL,
                    strategy=self.name,
                    price=row[close_col],
                    confidence=0.7,
                    reason="MACD crossed below signal line",
                    indicators={
                        'macd': row[macd_col],
                        'signal': row[signal_col],
                        'histogram': row[hist_col]
                    }
                ))

        return signals


class GoldSilverRatioStrategy(BaseStrategy):
    """Gold/Silver Ratio Mean Reversion Strategy"""

    def __init__(
        self,
        ratio_high: float = 90,
        ratio_low: float = 70,
        lookback: int = 60
    ):
        super().__init__("GS_Ratio")
        self.ratio_high = ratio_high
        self.ratio_low = ratio_low
        self.lookback = lookback

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        signals = []

        if 'gs_ratio' not in df.columns:
            return signals

        df = df.copy()
        df['ratio_ma'] = df['gs_ratio'].rolling(self.lookback).mean()
        df['ratio_std'] = df['gs_ratio'].rolling(self.lookback).std()
        df['ratio_zscore'] = (df['gs_ratio'] - df['ratio_ma']) / df['ratio_std']

        prev_zscore = None
        for idx in df.index:
            row = df.loc[idx]
            zscore = row['ratio_zscore']
            ratio = row['gs_ratio']

            if pd.isna(zscore):
                continue

            # Ratio very high - silver undervalued
            if prev_zscore is not None and prev_zscore > 2 and zscore <= 2:
                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=Asset.SILVER,
                    signal_type=SignalType.BUY,
                    strategy=self.name,
                    price=row['silver_close'],
                    confidence=min(0.5 + abs(zscore) * 0.1, 0.9),
                    reason=f"G/S ratio ({ratio:.1f}) reverting from high - silver undervalued",
                    indicators={'gs_ratio': ratio, 'zscore': zscore}
                ))

            # Ratio very low - gold undervalued
            if prev_zscore is not None and prev_zscore < -2 and zscore >= -2:
                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=Asset.GOLD,
                    signal_type=SignalType.BUY,
                    strategy=self.name,
                    price=row['gold_close'],
                    confidence=min(0.5 + abs(zscore) * 0.1, 0.9),
                    reason=f"G/S ratio ({ratio:.1f}) reverting from low - gold undervalued",
                    indicators={'gs_ratio': ratio, 'zscore': zscore}
                ))

            prev_zscore = zscore

        return signals


class CompositeStrategy(BaseStrategy):
    """Combines multiple strategies and aggregates signals"""

    def __init__(self, strategies: List[BaseStrategy] = None):
        super().__init__("Composite")
        self.strategies = strategies or []

    def add_strategy(self, strategy: BaseStrategy):
        self.strategies.append(strategy)

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        all_signals = []
        for strategy in self.strategies:
            signals = strategy.generate_signals(df)
            all_signals.extend(signals)

        # Sort by timestamp
        all_signals.sort(key=lambda s: s.timestamp)

        return all_signals

    def get_consensus_signal(
        self,
        df: pd.DataFrame,
        asset: Asset
    ) -> tuple[SignalType, float, List[Signal]]:
        """
        Get consensus signal from all strategies

        Returns:
            Tuple of (consensus_signal, confidence, contributing_signals)
        """
        signals = self.generate_signals(df)

        # Filter to most recent signals for this asset
        asset_signals = [s for s in signals if s.asset == asset]

        if not asset_signals:
            return SignalType.HOLD, 0.0, []

        # Get signals from last bar
        last_time = df.index[-1]
        recent_signals = [s for s in asset_signals
                        if s.timestamp == last_time or
                        (hasattr(last_time, 'date') and
                         hasattr(s.timestamp, 'date') and
                         s.timestamp.date() == last_time.date())]

        if not recent_signals:
            # Fall back to last 5 signals
            recent_signals = asset_signals[-5:] if len(asset_signals) >= 5 else asset_signals

        # Aggregate signals
        total_score = sum(s.signal_type.value * s.confidence for s in recent_signals)
        total_weight = sum(s.confidence for s in recent_signals)

        if total_weight == 0:
            return SignalType.HOLD, 0.0, recent_signals

        avg_score = total_score / total_weight

        # Map score to signal type
        if avg_score >= 1.5:
            consensus = SignalType.STRONG_BUY
        elif avg_score >= 0.5:
            consensus = SignalType.BUY
        elif avg_score <= -1.5:
            consensus = SignalType.STRONG_SELL
        elif avg_score <= -0.5:
            consensus = SignalType.SELL
        else:
            consensus = SignalType.HOLD

        confidence = min(abs(avg_score) / 2, 1.0)

        return consensus, confidence, recent_signals


def create_default_strategies() -> CompositeStrategy:
    """Create default composite strategy with all sub-strategies"""
    composite = CompositeStrategy()

    # Add strategies for both gold and silver
    for asset in [Asset.GOLD, Asset.SILVER]:
        composite.add_strategy(MACrossoverStrategy(asset=asset))
        composite.add_strategy(RSIMeanReversionStrategy(asset=asset))
        composite.add_strategy(BollingerBandStrategy(asset=asset))
        composite.add_strategy(MACDStrategy(asset=asset))

    # Add ratio strategy
    composite.add_strategy(GoldSilverRatioStrategy())

    return composite


if __name__ == "__main__":
    # Test strategies
    from data_fetcher import fetch_and_prepare_data
    from indicators import add_indicators_to_data

    print("Fetching and preparing data...")
    df = fetch_and_prepare_data(period="6mo")
    df = add_indicators_to_data(df)

    print(f"Data shape: {df.shape}")

    # Create composite strategy
    strategy = create_default_strategies()

    # Generate all signals
    print("\nGenerating signals...")
    signals = strategy.generate_signals(df)
    print(f"Total signals: {len(signals)}")

    # Show recent signals
    print("\nRecent signals:")
    for signal in signals[-10:]:
        print(f"  {signal.timestamp}: {signal.asset.value} {signal.signal_type.name} "
              f"@ ${signal.price:.2f} ({signal.strategy}) - {signal.reason[:50]}...")

    # Get consensus
    print("\nConsensus signals:")
    for asset in [Asset.GOLD, Asset.SILVER]:
        consensus, confidence, contrib = strategy.get_consensus_signal(df, asset)
        print(f"  {asset.value}: {consensus.name} (confidence: {confidence:.2f})")
