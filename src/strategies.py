"""
Trading strategies module for Gold/Silver ETF trading
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from scipy import stats

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
    GOLD_BEAR = "gold_bear"      # GLL - 2x Inverse Gold
    SILVER_BEAR = "silver_bear"  # ZSL - 2x Inverse Silver

    def is_bear(self) -> bool:
        """Check if this is a bear/inverse ETF"""
        return self in [Asset.GOLD_BEAR, Asset.SILVER_BEAR]

    def is_bull(self) -> bool:
        """Check if this is a bull/long ETF"""
        return self in [Asset.GOLD, Asset.SILVER]

    def get_inverse(self) -> 'Asset':
        """Get the inverse asset (bull->bear or bear->bull)"""
        mapping = {
            Asset.GOLD: Asset.GOLD_BEAR,
            Asset.SILVER: Asset.SILVER_BEAR,
            Asset.GOLD_BEAR: Asset.GOLD,
            Asset.SILVER_BEAR: Asset.SILVER
        }
        return mapping.get(self, self)

    def get_display_name(self) -> str:
        """Get human-readable display name"""
        names = {
            Asset.GOLD: "Gold (GLD)",
            Asset.SILVER: "Silver (SLV)",
            Asset.GOLD_BEAR: "Gold Bear (GLL)",
            Asset.SILVER_BEAR: "Silver Bear (ZSL)"
        }
        return names.get(self, self.value)


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

    def get_current_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """
        Get signal based on CURRENT state of indicators (latest row only).
        This should be used for live trading instead of generate_signals().
        Override in subclasses.
        """
        return None

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
                fast_ma = row[sma_fast_col]
                slow_ma = row[sma_slow_col]
                spread_pct = ((fast_ma - slow_ma) / slow_ma) * 100
                price = row[close_col]
                price_vs_ma = ((price - slow_ma) / slow_ma) * 100

                reason = (
                    f"BULLISH CROSSOVER: {self.fast_period}-SMA (${fast_ma:,.2f}) crossed above "
                    f"{self.slow_period}-SMA (${slow_ma:,.2f}). "
                    f"Spread: +{spread_pct:.2f}%. "
                    f"Price ${price:,.2f} is {price_vs_ma:+.2f}% from slow MA. "
                    f"Momentum shifting bullish - trend reversal signal."
                )

                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=self.asset,
                    signal_type=SignalType.BUY,
                    strategy=self.name,
                    price=price,
                    confidence=0.7,
                    reason=reason,
                    indicators={
                        'fast_ma': fast_ma,
                        'slow_ma': slow_ma,
                        'spread_pct': spread_pct,
                        'price_vs_slow_ma_pct': price_vs_ma
                    }
                ))
            elif row['crossover'] == False:  # Bearish crossover
                fast_ma = row[sma_fast_col]
                slow_ma = row[sma_slow_col]
                spread_pct = ((fast_ma - slow_ma) / slow_ma) * 100
                price = row[close_col]
                price_vs_ma = ((price - slow_ma) / slow_ma) * 100

                reason = (
                    f"BEARISH CROSSOVER: {self.fast_period}-SMA (${fast_ma:,.2f}) crossed below "
                    f"{self.slow_period}-SMA (${slow_ma:,.2f}). "
                    f"Spread: {spread_pct:.2f}%. "
                    f"Price ${price:,.2f} is {price_vs_ma:+.2f}% from slow MA. "
                    f"Momentum shifting bearish - trend reversal signal."
                )

                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=self.asset,
                    signal_type=SignalType.SELL,
                    strategy=self.name,
                    price=price,
                    confidence=0.7,
                    reason=reason,
                    indicators={
                        'fast_ma': fast_ma,
                        'slow_ma': slow_ma,
                        'spread_pct': spread_pct,
                        'price_vs_slow_ma_pct': price_vs_ma
                    }
                ))

        return signals

    def get_current_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Get signal based on current MA positions (not crossover events)"""
        prefix = f"{self.asset.value}_"
        sma_fast_col = f"{prefix}sma_fast"
        sma_slow_col = f"{prefix}sma_slow"
        close_col = f"{prefix}close"

        if sma_fast_col not in df.columns or sma_slow_col not in df.columns:
            return None

        row = df.iloc[-1]
        fast_ma = row[sma_fast_col]
        slow_ma = row[sma_slow_col]
        price = row[close_col]

        if pd.isna(fast_ma) or pd.isna(slow_ma):
            return None

        spread_pct = ((fast_ma - slow_ma) / slow_ma) * 100
        price_vs_ma = ((price - slow_ma) / slow_ma) * 100

        # Determine signal based on current state
        if fast_ma > slow_ma:
            # Bullish: fast MA above slow MA
            strength = "strong" if spread_pct > 1.5 else "moderate"
            reason = (
                f"BULLISH TREND: {self.fast_period}-SMA (${fast_ma:,.2f}) is above "
                f"{self.slow_period}-SMA (${slow_ma:,.2f}). "
                f"Spread: +{spread_pct:.2f}% ({strength}). "
                f"Price ${price:,.2f} is {price_vs_ma:+.2f}% from slow MA."
            )
            signal_type = SignalType.STRONG_BUY if spread_pct > 2 else SignalType.BUY
            confidence = min(0.5 + spread_pct / 5, 0.9)
        elif fast_ma < slow_ma:
            # Bearish: fast MA below slow MA
            strength = "strong" if abs(spread_pct) > 1.5 else "moderate"
            reason = (
                f"BEARISH TREND: {self.fast_period}-SMA (${fast_ma:,.2f}) is below "
                f"{self.slow_period}-SMA (${slow_ma:,.2f}). "
                f"Spread: {spread_pct:.2f}% ({strength}). "
                f"Price ${price:,.2f} is {price_vs_ma:+.2f}% from slow MA."
            )
            signal_type = SignalType.STRONG_SELL if spread_pct < -2 else SignalType.SELL
            confidence = min(0.5 + abs(spread_pct) / 5, 0.9)
        else:
            return None  # Exactly equal - no signal

        return Signal(
            timestamp=df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
            asset=self.asset,
            signal_type=signal_type,
            strategy=self.name,
            price=price,
            confidence=confidence,
            reason=reason,
            indicators={
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'spread_pct': spread_pct,
                'price_vs_slow_ma_pct': price_vs_ma
            }
        )


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
                rsi_change = rsi - prev_rsi
                oversold_depth = self.oversold - prev_rsi
                confidence = min(0.6 + (self.oversold - prev_rsi) / 100, 0.95)
                price = row[close_col]

                depth_desc = "extreme" if oversold_depth > 10 else "significant" if oversold_depth > 5 else "moderate"

                reason = (
                    f"RSI REVERSAL FROM OVERSOLD: RSI rose from {prev_rsi:.1f} to {rsi:.1f} "
                    f"(+{rsi_change:.1f} pts), crossing above {self.oversold} threshold. "
                    f"Oversold depth was {depth_desc} ({oversold_depth:.1f} pts below threshold). "
                    f"Price: ${price:,.2f}. "
                    f"High probability mean reversion setup - selling pressure exhausted."
                )

                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=self.asset,
                    signal_type=SignalType.BUY,
                    strategy=self.name,
                    price=price,
                    confidence=confidence,
                    reason=reason,
                    indicators={
                        'rsi': rsi,
                        'prev_rsi': prev_rsi,
                        'rsi_change': rsi_change,
                        'oversold_depth': oversold_depth
                    }
                ))

            # Overbought reversal (RSI crosses below overbought level)
            if prev_rsi is not None and prev_rsi > self.overbought and rsi <= self.overbought:
                rsi_change = prev_rsi - rsi
                overbought_excess = prev_rsi - self.overbought
                confidence = min(0.6 + (prev_rsi - self.overbought) / 100, 0.95)
                price = row[close_col]

                excess_desc = "extreme" if overbought_excess > 10 else "significant" if overbought_excess > 5 else "moderate"

                reason = (
                    f"RSI REVERSAL FROM OVERBOUGHT: RSI fell from {prev_rsi:.1f} to {rsi:.1f} "
                    f"(-{rsi_change:.1f} pts), crossing below {self.overbought} threshold. "
                    f"Overbought excess was {excess_desc} ({overbought_excess:.1f} pts above threshold). "
                    f"Price: ${price:,.2f}. "
                    f"Mean reversion signal - buying pressure exhausted."
                )

                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=self.asset,
                    signal_type=SignalType.SELL,
                    strategy=self.name,
                    price=price,
                    confidence=confidence,
                    reason=reason,
                    indicators={
                        'rsi': rsi,
                        'prev_rsi': prev_rsi,
                        'rsi_change': rsi_change,
                        'overbought_excess': overbought_excess
                    }
                ))

            prev_rsi = rsi

        return signals

    def get_current_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Get signal based on current RSI level (not crossover events)"""
        prefix = f"{self.asset.value}_"
        rsi_col = f"{prefix}rsi"
        close_col = f"{prefix}close"

        if rsi_col not in df.columns:
            return None

        row = df.iloc[-1]
        rsi = row[rsi_col]
        price = row[close_col]

        if pd.isna(rsi):
            return None

        # Determine signal based on current RSI zone
        if rsi < self.oversold:
            # Oversold - potential buy
            depth = self.oversold - rsi
            depth_desc = "extreme" if depth > 10 else "significant" if depth > 5 else "moderate"
            confidence = min(0.5 + depth / 50, 0.9)

            reason = (
                f"RSI OVERSOLD: Current RSI is {rsi:.1f}, which is {depth:.1f} pts below "
                f"the {self.oversold} oversold threshold ({depth_desc}). "
                f"Price: ${price:,.2f}. "
                f"Asset may be undervalued - watch for reversal confirmation."
            )
            signal_type = SignalType.STRONG_BUY if depth > 10 else SignalType.BUY

        elif rsi > self.overbought:
            # Overbought - potential sell
            excess = rsi - self.overbought
            excess_desc = "extreme" if excess > 10 else "significant" if excess > 5 else "moderate"
            confidence = min(0.5 + excess / 50, 0.9)

            reason = (
                f"RSI OVERBOUGHT: Current RSI is {rsi:.1f}, which is {excess:.1f} pts above "
                f"the {self.overbought} overbought threshold ({excess_desc}). "
                f"Price: ${price:,.2f}. "
                f"Asset may be overvalued - watch for reversal confirmation."
            )
            signal_type = SignalType.STRONG_SELL if excess > 10 else SignalType.SELL

        elif rsi >= 45 and rsi <= 55:
            # Neutral zone - hold
            return None
        elif rsi > 55:
            # Upper neutral zone - slight bullish bias
            confidence = 0.4
            reason = (
                f"RSI NEUTRAL-BULLISH: Current RSI is {rsi:.1f}, above midline but below "
                f"overbought. Price: ${price:,.2f}. Momentum slightly positive."
            )
            signal_type = SignalType.HOLD
            return None  # Don't generate signal for neutral zones
        else:
            # Lower neutral zone - slight bearish bias
            confidence = 0.4
            reason = (
                f"RSI NEUTRAL-BEARISH: Current RSI is {rsi:.1f}, below midline but above "
                f"oversold. Price: ${price:,.2f}. Momentum slightly negative."
            )
            signal_type = SignalType.HOLD
            return None  # Don't generate signal for neutral zones

        return Signal(
            timestamp=df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
            asset=self.asset,
            signal_type=signal_type,
            strategy=self.name,
            price=price,
            confidence=confidence,
            reason=reason,
            indicators={
                'rsi': rsi,
                'oversold_threshold': self.oversold,
                'overbought_threshold': self.overbought
            }
        )


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
                mid = row[bb_mid]
                band_width = ((upper - lower) / mid) * 100
                pct_from_lower = ((close - lower) / lower) * 100
                pct_to_mid = ((mid - close) / close) * 100

                width_desc = "tight" if band_width < 3 else "normal" if band_width < 6 else "wide"

                reason = (
                    f"LOWER BB BOUNCE: Price touched ${lower:,.2f} (lower band) and recovered to ${close:,.2f}. "
                    f"Band width: {band_width:.1f}% ({width_desc} volatility). "
                    f"Price now {pct_from_lower:.2f}% above lower band, {pct_to_mid:.2f}% below middle band (${mid:,.2f}). "
                    f"Mean reversion opportunity - price reverted from 2-sigma extreme."
                )

                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=self.asset,
                    signal_type=SignalType.BUY,
                    strategy=self.name,
                    price=close,
                    confidence=0.65,
                    reason=reason,
                    indicators={
                        'bb_upper': upper,
                        'bb_lower': lower,
                        'bb_mid': mid,
                        'band_width_pct': band_width,
                        'pct_from_lower': pct_from_lower
                    }
                ))

            # Price bounces off upper band
            if prev_close is not None and prev_close >= upper and close < upper:
                mid = row[bb_mid]
                band_width = ((upper - lower) / mid) * 100
                pct_from_upper = ((upper - close) / close) * 100
                pct_from_mid = ((close - mid) / mid) * 100

                width_desc = "tight" if band_width < 3 else "normal" if band_width < 6 else "wide"

                reason = (
                    f"UPPER BB REJECTION: Price touched ${upper:,.2f} (upper band) and pulled back to ${close:,.2f}. "
                    f"Band width: {band_width:.1f}% ({width_desc} volatility). "
                    f"Price now {pct_from_upper:.2f}% below upper band, {pct_from_mid:.2f}% above middle band (${mid:,.2f}). "
                    f"Mean reversion signal - price rejected at 2-sigma extreme."
                )

                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=self.asset,
                    signal_type=SignalType.SELL,
                    strategy=self.name,
                    price=close,
                    confidence=0.65,
                    reason=reason,
                    indicators={
                        'bb_upper': upper,
                        'bb_lower': lower,
                        'bb_mid': mid,
                        'band_width_pct': band_width,
                        'pct_from_upper': pct_from_upper
                    }
                ))

            prev_close = close

        return signals

    def get_current_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Get signal based on current price position relative to Bollinger Bands"""
        prefix = f"{self.asset.value}_"
        bb_upper = f"{prefix}bb_upper"
        bb_lower = f"{prefix}bb_lower"
        bb_mid = f"{prefix}bb_mid"
        close_col = f"{prefix}close"

        if bb_upper not in df.columns:
            return None

        row = df.iloc[-1]
        close = row[close_col]
        upper = row[bb_upper]
        lower = row[bb_lower]
        mid = row[bb_mid]

        if pd.isna(upper) or pd.isna(lower) or pd.isna(mid):
            return None

        band_width = ((upper - lower) / mid) * 100
        width_desc = "tight" if band_width < 3 else "normal" if band_width < 6 else "wide"

        # Calculate position within bands (0 = lower, 1 = upper)
        band_position = (close - lower) / (upper - lower)
        pct_from_mid = ((close - mid) / mid) * 100

        # Determine signal based on current position
        if close <= lower:
            # Price at or below lower band - oversold
            pct_below = ((lower - close) / lower) * 100
            confidence = min(0.5 + pct_below / 5, 0.85)
            reason = (
                f"BB LOWER BAND TOUCH: Price ${close:,.2f} is at/below lower band ${lower:,.2f}. "
                f"Band width: {band_width:.1f}% ({width_desc} volatility). "
                f"Position: {band_position:.1%} within bands. "
                f"Price at 2-sigma extreme - potential mean reversion opportunity."
            )
            signal_type = SignalType.BUY

        elif close >= upper:
            # Price at or above upper band - overbought
            pct_above = ((close - upper) / upper) * 100
            confidence = min(0.5 + pct_above / 5, 0.85)
            reason = (
                f"BB UPPER BAND TOUCH: Price ${close:,.2f} is at/above upper band ${upper:,.2f}. "
                f"Band width: {band_width:.1f}% ({width_desc} volatility). "
                f"Position: {band_position:.1%} within bands. "
                f"Price at 2-sigma extreme - potential reversal zone."
            )
            signal_type = SignalType.SELL

        elif band_position < 0.2:
            # Price near lower band but not touching
            confidence = 0.5
            reason = (
                f"BB NEAR LOWER: Price ${close:,.2f} is in lower 20% of bands. "
                f"Lower: ${lower:,.2f}, Mid: ${mid:,.2f}, Upper: ${upper:,.2f}. "
                f"Band position: {band_position:.1%}. Mild bullish bias."
            )
            signal_type = SignalType.BUY

        elif band_position > 0.8:
            # Price near upper band but not touching
            confidence = 0.5
            reason = (
                f"BB NEAR UPPER: Price ${close:,.2f} is in upper 20% of bands. "
                f"Lower: ${lower:,.2f}, Mid: ${mid:,.2f}, Upper: ${upper:,.2f}. "
                f"Band position: {band_position:.1%}. Mild bearish bias."
            )
            signal_type = SignalType.SELL

        else:
            # Price in middle zone - neutral
            return None

        return Signal(
            timestamp=df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
            asset=self.asset,
            signal_type=signal_type,
            strategy=self.name,
            price=close,
            confidence=confidence,
            reason=reason,
            indicators={
                'bb_upper': upper,
                'bb_lower': lower,
                'bb_mid': mid,
                'band_width_pct': band_width,
                'band_position': band_position
            }
        )


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
                macd = row[macd_col]
                signal = row[signal_col]
                hist = row[hist_col]
                price = row[close_col]

                # Determine if crossover is above or below zero line
                zone = "above zero (bullish territory)" if macd > 0 else "below zero (recovery from bearish)"

                reason = (
                    f"BULLISH MACD CROSSOVER: MACD ({macd:.3f}) crossed above Signal ({signal:.3f}). "
                    f"Histogram turned positive ({hist:+.3f}). "
                    f"Crossover occurred {zone}. "
                    f"Price: ${price:,.2f}. "
                    f"Momentum shifting bullish - trend strength increasing."
                )

                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=self.asset,
                    signal_type=SignalType.BUY,
                    strategy=self.name,
                    price=price,
                    confidence=0.7,
                    reason=reason,
                    indicators={
                        'macd': macd,
                        'signal': signal,
                        'histogram': hist,
                        'above_zero': macd > 0
                    }
                ))
            elif row['macd_cross'] == False:  # Bearish crossover
                macd = row[macd_col]
                signal = row[signal_col]
                hist = row[hist_col]
                price = row[close_col]

                # Determine if crossover is above or below zero line
                zone = "below zero (bearish territory)" if macd < 0 else "above zero (reversal from bullish)"

                reason = (
                    f"BEARISH MACD CROSSOVER: MACD ({macd:.3f}) crossed below Signal ({signal:.3f}). "
                    f"Histogram turned negative ({hist:.3f}). "
                    f"Crossover occurred {zone}. "
                    f"Price: ${price:,.2f}. "
                    f"Momentum shifting bearish - trend strength weakening."
                )

                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=self.asset,
                    signal_type=SignalType.SELL,
                    strategy=self.name,
                    price=price,
                    confidence=0.7,
                    reason=reason,
                    indicators={
                        'macd': macd,
                        'signal': signal,
                        'histogram': hist,
                        'above_zero': macd > 0
                    }
                ))

        return signals

    def get_current_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Get signal based on current MACD vs Signal line position"""
        prefix = f"{self.asset.value}_"
        macd_col = f"{prefix}macd"
        signal_col = f"{prefix}macd_signal"
        hist_col = f"{prefix}macd_hist"
        close_col = f"{prefix}close"

        if macd_col not in df.columns:
            return None

        row = df.iloc[-1]
        macd = row[macd_col]
        signal = row[signal_col]
        hist = row[hist_col]
        price = row[close_col]

        if pd.isna(macd) or pd.isna(signal):
            return None

        # Calculate momentum strength
        hist_strength = abs(hist) * 100  # Scale histogram for comparison

        # Determine signal based on current MACD position
        if macd > signal:
            # MACD above signal line - bullish
            spread = macd - signal
            if macd > 0:
                # MACD above signal AND above zero - strong bullish
                zone = "above zero (bullish territory)"
                signal_type = SignalType.STRONG_BUY if hist > 0.01 else SignalType.BUY
                confidence = min(0.6 + abs(hist) * 10, 0.9)
            else:
                # MACD above signal but below zero - recovering
                zone = "below zero (recovery phase)"
                signal_type = SignalType.BUY
                confidence = min(0.5 + abs(hist) * 10, 0.8)

            reason = (
                f"MACD BULLISH: MACD ({macd:.3f}) is above Signal ({signal:.3f}). "
                f"Histogram: {hist:+.3f}. Position: {zone}. "
                f"Price: ${price:,.2f}. Bullish momentum active."
            )

        elif macd < signal:
            # MACD below signal line - bearish
            spread = signal - macd
            if macd < 0:
                # MACD below signal AND below zero - strong bearish
                zone = "below zero (bearish territory)"
                signal_type = SignalType.STRONG_SELL if hist < -0.01 else SignalType.SELL
                confidence = min(0.6 + abs(hist) * 10, 0.9)
            else:
                # MACD below signal but above zero - weakening
                zone = "above zero (weakening phase)"
                signal_type = SignalType.SELL
                confidence = min(0.5 + abs(hist) * 10, 0.8)

            reason = (
                f"MACD BEARISH: MACD ({macd:.3f}) is below Signal ({signal:.3f}). "
                f"Histogram: {hist:.3f}. Position: {zone}. "
                f"Price: ${price:,.2f}. Bearish momentum active."
            )

        else:
            return None  # Exactly equal - no signal

        return Signal(
            timestamp=df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
            asset=self.asset,
            signal_type=signal_type,
            strategy=self.name,
            price=price,
            confidence=confidence,
            reason=reason,
            indicators={
                'macd': macd,
                'signal': signal,
                'histogram': hist,
                'above_zero': macd > 0
            }
        )


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
                mean_ratio = row['ratio_ma']
                std_ratio = row['ratio_std']
                confidence = min(0.5 + abs(zscore) * 0.1, 0.9)
                silver_price = row['silver_close']
                gold_price = row['gold_close']

                # Calculate percentile
                percentile = 50 + (zscore * 34)  # Approximate percentile from z-score
                percentile = min(max(percentile, 0), 100)

                reason = (
                    f"G/S RATIO MEAN REVERSION (BUY SILVER): Current ratio {ratio:.1f} "
                    f"(z-score: {zscore:+.2f}) reverting toward mean of {mean_ratio:.1f}. "
                    f"Ratio was in ~{percentile:.0f}th percentile - historically extreme. "
                    f"Silver (${silver_price:,.2f}) undervalued relative to Gold (${gold_price:,.2f}). "
                    f"Expect ratio compression as silver outperforms."
                )

                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=Asset.SILVER,
                    signal_type=SignalType.BUY,
                    strategy=self.name,
                    price=silver_price,
                    confidence=confidence,
                    reason=reason,
                    indicators={
                        'gs_ratio': ratio,
                        'zscore': zscore,
                        'mean_ratio': mean_ratio,
                        'std_ratio': std_ratio,
                        'percentile': percentile
                    }
                ))

            # Ratio very low - gold undervalued
            if prev_zscore is not None and prev_zscore < -2 and zscore >= -2:
                mean_ratio = row['ratio_ma']
                std_ratio = row['ratio_std']
                confidence = min(0.5 + abs(zscore) * 0.1, 0.9)
                silver_price = row['silver_close']
                gold_price = row['gold_close']

                # Calculate percentile
                percentile = 50 + (zscore * 34)  # Approximate percentile from z-score
                percentile = min(max(percentile, 0), 100)

                reason = (
                    f"G/S RATIO MEAN REVERSION (BUY GOLD): Current ratio {ratio:.1f} "
                    f"(z-score: {zscore:+.2f}) reverting toward mean of {mean_ratio:.1f}. "
                    f"Ratio was in ~{percentile:.0f}th percentile - historically extreme. "
                    f"Gold (${gold_price:,.2f}) undervalued relative to Silver (${silver_price:,.2f}). "
                    f"Expect ratio expansion as gold outperforms."
                )

                signals.append(Signal(
                    timestamp=idx if isinstance(idx, datetime) else datetime.now(),
                    asset=Asset.GOLD,
                    signal_type=SignalType.BUY,
                    strategy=self.name,
                    price=gold_price,
                    confidence=confidence,
                    reason=reason,
                    indicators={
                        'gs_ratio': ratio,
                        'zscore': zscore,
                        'mean_ratio': mean_ratio,
                        'std_ratio': std_ratio,
                        'percentile': percentile
                    }
                ))

            prev_zscore = zscore

        return signals

    def get_current_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Get signal based on current Gold/Silver ratio z-score"""
        if 'gs_ratio' not in df.columns:
            return None

        # Calculate z-score for current ratio
        ratio_series = df['gs_ratio']
        ratio_ma = ratio_series.rolling(self.lookback).mean()
        ratio_std = ratio_series.rolling(self.lookback).std()

        row = df.iloc[-1]
        ratio = row['gs_ratio']
        mean_ratio = ratio_ma.iloc[-1]
        std_ratio = ratio_std.iloc[-1]

        if pd.isna(ratio) or pd.isna(mean_ratio) or pd.isna(std_ratio) or std_ratio == 0:
            return None

        zscore = (ratio - mean_ratio) / std_ratio
        percentile = 50 + (zscore * 34)
        percentile = min(max(percentile, 0), 100)

        silver_price = row['silver_close']
        gold_price = row['gold_close']

        # Determine signal based on current z-score
        if zscore > 2:
            # Ratio extremely high - silver undervalued
            confidence = min(0.5 + abs(zscore) * 0.1, 0.9)
            reason = (
                f"G/S RATIO HIGH (BUY SILVER): Ratio {ratio:.1f} is {zscore:.2f} std devs above mean ({mean_ratio:.1f}). "
                f"~{percentile:.0f}th percentile - silver appears undervalued. "
                f"Silver: ${silver_price:,.2f}, Gold: ${gold_price:,.2f}."
            )
            signal_type = SignalType.STRONG_BUY if zscore > 2.5 else SignalType.BUY
            asset = Asset.SILVER
            price = silver_price

        elif zscore > 1:
            # Ratio moderately high
            confidence = 0.5
            reason = (
                f"G/S RATIO ELEVATED: Ratio {ratio:.1f} is {zscore:.2f} std devs above mean ({mean_ratio:.1f}). "
                f"~{percentile:.0f}th percentile - mild silver bias."
            )
            signal_type = SignalType.BUY
            asset = Asset.SILVER
            price = silver_price

        elif zscore < -2:
            # Ratio extremely low - gold undervalued
            confidence = min(0.5 + abs(zscore) * 0.1, 0.9)
            reason = (
                f"G/S RATIO LOW (BUY GOLD): Ratio {ratio:.1f} is {zscore:.2f} std devs below mean ({mean_ratio:.1f}). "
                f"~{percentile:.0f}th percentile - gold appears undervalued. "
                f"Gold: ${gold_price:,.2f}, Silver: ${silver_price:,.2f}."
            )
            signal_type = SignalType.STRONG_BUY if zscore < -2.5 else SignalType.BUY
            asset = Asset.GOLD
            price = gold_price

        elif zscore < -1:
            # Ratio moderately low
            confidence = 0.5
            reason = (
                f"G/S RATIO DEPRESSED: Ratio {ratio:.1f} is {zscore:.2f} std devs below mean ({mean_ratio:.1f}). "
                f"~{percentile:.0f}th percentile - mild gold bias."
            )
            signal_type = SignalType.BUY
            asset = Asset.GOLD
            price = gold_price

        else:
            # Ratio in normal range
            return None

        return Signal(
            timestamp=df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
            asset=asset,
            signal_type=signal_type,
            strategy=self.name,
            price=price,
            confidence=confidence,
            reason=reason,
            indicators={
                'gs_ratio': ratio,
                'zscore': zscore,
                'mean_ratio': mean_ratio,
                'std_ratio': std_ratio,
                'percentile': percentile
            }
        )


class ADXTrendStrategy(BaseStrategy):
    """ADX-based Trend Strength Strategy"""

    def __init__(self, adx_threshold: float = 25, asset: Asset = Asset.GOLD):
        super().__init__("ADX_Trend")
        self.adx_threshold = adx_threshold
        self.asset = asset

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        return []  # Use get_current_signal for live trading

    def get_current_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Get signal based on ADX trend strength and DI crossover"""
        prefix = f"{self.asset.value}_"
        adx_col = f"{prefix}adx"
        plus_di_col = f"{prefix}plus_di"
        minus_di_col = f"{prefix}minus_di"
        close_col = f"{prefix}close"

        if adx_col not in df.columns:
            return None

        row = df.iloc[-1]
        adx = row[adx_col]
        plus_di = row[plus_di_col]
        minus_di = row[minus_di_col]
        price = row[close_col]

        if pd.isna(adx) or pd.isna(plus_di) or pd.isna(minus_di):
            return None

        # ADX indicates trend strength, DI crossover indicates direction
        if adx > self.adx_threshold:
            # Strong trend
            if plus_di > minus_di:
                di_spread = plus_di - minus_di
                confidence = min(0.5 + (adx - self.adx_threshold) / 50 + di_spread / 100, 0.95)
                reason = (
                    f"ADX STRONG UPTREND: ADX={adx:.1f} (>{self.adx_threshold}), "
                    f"+DI={plus_di:.1f} > -DI={minus_di:.1f}. "
                    f"Strong bullish trend confirmed. Price: ${price:,.2f}."
                )
                signal_type = SignalType.STRONG_BUY if adx > 40 else SignalType.BUY
            else:
                di_spread = minus_di - plus_di
                confidence = min(0.5 + (adx - self.adx_threshold) / 50 + di_spread / 100, 0.95)
                reason = (
                    f"ADX STRONG DOWNTREND: ADX={adx:.1f} (>{self.adx_threshold}), "
                    f"-DI={minus_di:.1f} > +DI={plus_di:.1f}. "
                    f"Strong bearish trend confirmed. Price: ${price:,.2f}."
                )
                signal_type = SignalType.STRONG_SELL if adx > 40 else SignalType.SELL
        else:
            # Weak trend - no signal
            return None

        return Signal(
            timestamp=df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
            asset=self.asset,
            signal_type=signal_type,
            strategy=self.name,
            price=price,
            confidence=confidence,
            reason=reason,
            indicators={'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}
        )


class StochasticStrategy(BaseStrategy):
    """Stochastic Oscillator Strategy"""

    def __init__(self, overbought: float = 80, oversold: float = 20, asset: Asset = Asset.GOLD):
        super().__init__("Stochastic")
        self.overbought = overbought
        self.oversold = oversold
        self.asset = asset

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        return []

    def get_current_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Get signal based on Stochastic %K and %D"""
        prefix = f"{self.asset.value}_"
        k_col = f"{prefix}stoch_k"
        d_col = f"{prefix}stoch_d"
        close_col = f"{prefix}close"

        if k_col not in df.columns:
            return None

        row = df.iloc[-1]
        k = row[k_col]
        d = row[d_col]
        price = row[close_col]

        if pd.isna(k) or pd.isna(d):
            return None

        if k < self.oversold and d < self.oversold:
            # Oversold - potential buy
            depth = self.oversold - min(k, d)
            confidence = min(0.5 + depth / 40, 0.85)
            reason = (
                f"STOCHASTIC OVERSOLD: %K={k:.1f}, %D={d:.1f} (both <{self.oversold}). "
                f"Price: ${price:,.2f}. Potential reversal zone."
            )
            signal_type = SignalType.BUY
        elif k > self.overbought and d > self.overbought:
            # Overbought - potential sell
            excess = max(k, d) - self.overbought
            confidence = min(0.5 + excess / 40, 0.85)
            reason = (
                f"STOCHASTIC OVERBOUGHT: %K={k:.1f}, %D={d:.1f} (both >{self.overbought}). "
                f"Price: ${price:,.2f}. Potential reversal zone."
            )
            signal_type = SignalType.SELL
        elif k > d and k < 50:
            # %K crossing above %D in lower half - bullish
            confidence = 0.5
            reason = f"STOCHASTIC BULLISH CROSS: %K={k:.1f} > %D={d:.1f} in lower zone."
            signal_type = SignalType.BUY
        elif k < d and k > 50:
            # %K crossing below %D in upper half - bearish
            confidence = 0.5
            reason = f"STOCHASTIC BEARISH CROSS: %K={k:.1f} < %D={d:.1f} in upper zone."
            signal_type = SignalType.SELL
        else:
            return None

        return Signal(
            timestamp=df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
            asset=self.asset,
            signal_type=signal_type,
            strategy=self.name,
            price=price,
            confidence=confidence,
            reason=reason,
            indicators={'stoch_k': k, 'stoch_d': d}
        )


class CCIStrategy(BaseStrategy):
    """Commodity Channel Index Strategy"""

    def __init__(self, overbought: float = 100, oversold: float = -100, asset: Asset = Asset.GOLD):
        super().__init__("CCI")
        self.overbought = overbought
        self.oversold = oversold
        self.asset = asset

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        return []

    def get_current_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Get signal based on CCI levels"""
        prefix = f"{self.asset.value}_"
        cci_col = f"{prefix}cci"
        close_col = f"{prefix}close"

        if cci_col not in df.columns:
            return None

        row = df.iloc[-1]
        cci = row[cci_col]
        price = row[close_col]

        if pd.isna(cci):
            return None

        if cci < self.oversold:
            # Oversold
            depth = abs(cci - self.oversold)
            confidence = min(0.5 + depth / 200, 0.9)
            signal_type = SignalType.STRONG_BUY if cci < -200 else SignalType.BUY
            reason = (
                f"CCI OVERSOLD: CCI={cci:.1f} (<{self.oversold}). "
                f"Price: ${price:,.2f}. Potential bullish reversal."
            )
        elif cci > self.overbought:
            # Overbought
            excess = cci - self.overbought
            confidence = min(0.5 + excess / 200, 0.9)
            signal_type = SignalType.STRONG_SELL if cci > 200 else SignalType.SELL
            reason = (
                f"CCI OVERBOUGHT: CCI={cci:.1f} (>{self.overbought}). "
                f"Price: ${price:,.2f}. Potential bearish reversal."
            )
        elif cci > 0 and cci < 50:
            # Mild bullish
            confidence = 0.4
            signal_type = SignalType.BUY
            reason = f"CCI MILD BULLISH: CCI={cci:.1f}. Neutral-positive momentum."
        elif cci < 0 and cci > -50:
            # Mild bearish
            confidence = 0.4
            signal_type = SignalType.SELL
            reason = f"CCI MILD BEARISH: CCI={cci:.1f}. Neutral-negative momentum."
        else:
            return None

        return Signal(
            timestamp=df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
            asset=self.asset,
            signal_type=signal_type,
            strategy=self.name,
            price=price,
            confidence=confidence,
            reason=reason,
            indicators={'cci': cci}
        )


class IchimokuStrategy(BaseStrategy):
    """Ichimoku Cloud Strategy"""

    def __init__(self, asset: Asset = Asset.GOLD):
        super().__init__("Ichimoku")
        self.asset = asset

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        return []

    def get_current_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Get signal based on Ichimoku Cloud"""
        prefix = f"{self.asset.value}_"
        tenkan_col = f"{prefix}ichimoku_tenkan"
        kijun_col = f"{prefix}ichimoku_kijun"
        senkou_a_col = f"{prefix}ichimoku_senkou_a"
        senkou_b_col = f"{prefix}ichimoku_senkou_b"
        close_col = f"{prefix}close"

        if tenkan_col not in df.columns:
            return None

        row = df.iloc[-1]
        tenkan = row[tenkan_col]
        kijun = row[kijun_col]
        senkou_a = row[senkou_a_col]
        senkou_b = row[senkou_b_col]
        price = row[close_col]

        if pd.isna(tenkan) or pd.isna(kijun):
            return None

        cloud_top = max(senkou_a, senkou_b) if not pd.isna(senkou_a) and not pd.isna(senkou_b) else None
        cloud_bottom = min(senkou_a, senkou_b) if not pd.isna(senkou_a) and not pd.isna(senkou_b) else None

        signals_bullish = 0
        signals_bearish = 0

        # TK Cross
        if tenkan > kijun:
            signals_bullish += 1
        else:
            signals_bearish += 1

        # Price vs Cloud
        if cloud_top and cloud_bottom:
            if price > cloud_top:
                signals_bullish += 2  # Strong bullish
            elif price < cloud_bottom:
                signals_bearish += 2  # Strong bearish

        # Determine signal
        if signals_bullish > signals_bearish:
            confidence = min(0.4 + signals_bullish * 0.15, 0.9)
            if price > cloud_top if cloud_top else price > kijun:
                signal_type = SignalType.STRONG_BUY
                reason = (
                    f"ICHIMOKU BULLISH: Price ${price:,.2f} above cloud. "
                    f"Tenkan={tenkan:.2f} > Kijun={kijun:.2f}. Strong uptrend."
                )
            else:
                signal_type = SignalType.BUY
                reason = (
                    f"ICHIMOKU MILD BULLISH: Tenkan={tenkan:.2f} > Kijun={kijun:.2f}. "
                    f"Price ${price:,.2f}."
                )
        elif signals_bearish > signals_bullish:
            confidence = min(0.4 + signals_bearish * 0.15, 0.9)
            if price < cloud_bottom if cloud_bottom else price < kijun:
                signal_type = SignalType.STRONG_SELL
                reason = (
                    f"ICHIMOKU BEARISH: Price ${price:,.2f} below cloud. "
                    f"Tenkan={tenkan:.2f} < Kijun={kijun:.2f}. Strong downtrend."
                )
            else:
                signal_type = SignalType.SELL
                reason = (
                    f"ICHIMOKU MILD BEARISH: Tenkan={tenkan:.2f} < Kijun={kijun:.2f}. "
                    f"Price ${price:,.2f}."
                )
        else:
            return None

        return Signal(
            timestamp=df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
            asset=self.asset,
            signal_type=signal_type,
            strategy=self.name,
            price=price,
            confidence=confidence,
            reason=reason,
            indicators={'tenkan': tenkan, 'kijun': kijun, 'senkou_a': senkou_a, 'senkou_b': senkou_b}
        )


class MFIStrategy(BaseStrategy):
    """Money Flow Index Strategy"""

    def __init__(self, overbought: float = 80, oversold: float = 20, asset: Asset = Asset.GOLD):
        super().__init__("MFI")
        self.overbought = overbought
        self.oversold = oversold
        self.asset = asset

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        return []

    def get_current_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Get signal based on MFI (volume-weighted RSI)"""
        prefix = f"{self.asset.value}_"
        mfi_col = f"{prefix}mfi"
        close_col = f"{prefix}close"

        if mfi_col not in df.columns:
            return None

        row = df.iloc[-1]
        mfi = row[mfi_col]
        price = row[close_col]

        if pd.isna(mfi):
            return None

        if mfi < self.oversold:
            depth = self.oversold - mfi
            confidence = min(0.5 + depth / 40, 0.9)
            signal_type = SignalType.STRONG_BUY if mfi < 10 else SignalType.BUY
            reason = (
                f"MFI OVERSOLD: MFI={mfi:.1f} (<{self.oversold}). "
                f"Volume confirms selling exhaustion. Price: ${price:,.2f}."
            )
        elif mfi > self.overbought:
            excess = mfi - self.overbought
            confidence = min(0.5 + excess / 40, 0.9)
            signal_type = SignalType.STRONG_SELL if mfi > 90 else SignalType.SELL
            reason = (
                f"MFI OVERBOUGHT: MFI={mfi:.1f} (>{self.overbought}). "
                f"Volume confirms buying exhaustion. Price: ${price:,.2f}."
            )
        else:
            return None

        return Signal(
            timestamp=df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
            asset=self.asset,
            signal_type=signal_type,
            strategy=self.name,
            price=price,
            confidence=confidence,
            reason=reason,
            indicators={'mfi': mfi}
        )


class PivotPointStrategy(BaseStrategy):
    """Pivot Point Support/Resistance Strategy"""

    def __init__(self, asset: Asset = Asset.GOLD):
        super().__init__("PivotPoint")
        self.asset = asset

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        return []

    def get_current_signal(self, df: pd.DataFrame) -> Optional[Signal]:
        """Get signal based on price relative to pivot points"""
        prefix = f"{self.asset.value}_"
        pivot_col = f"{prefix}pivot"
        r1_col = f"{prefix}r1"
        s1_col = f"{prefix}s1"
        close_col = f"{prefix}close"

        if pivot_col not in df.columns:
            return None

        row = df.iloc[-1]
        pivot = row[pivot_col]
        r1 = row[r1_col]
        s1 = row[s1_col]
        price = row[close_col]

        if pd.isna(pivot) or pd.isna(r1) or pd.isna(s1):
            return None

        # Calculate price position
        if price > r1:
            # Above R1 - strong bullish breakout
            confidence = 0.7
            signal_type = SignalType.BUY
            reason = (
                f"PIVOT BREAKOUT: Price ${price:,.2f} above R1 (${r1:,.2f}). "
                f"Pivot: ${pivot:,.2f}. Bullish momentum."
            )
        elif price < s1:
            # Below S1 - strong bearish breakdown
            confidence = 0.7
            signal_type = SignalType.SELL
            reason = (
                f"PIVOT BREAKDOWN: Price ${price:,.2f} below S1 (${s1:,.2f}). "
                f"Pivot: ${pivot:,.2f}. Bearish momentum."
            )
        elif price > pivot:
            # Above pivot - mild bullish
            pct_above = ((price - pivot) / pivot) * 100
            confidence = min(0.4 + pct_above / 5, 0.6)
            signal_type = SignalType.BUY
            reason = (
                f"ABOVE PIVOT: Price ${price:,.2f} > Pivot ${pivot:,.2f}. "
                f"R1: ${r1:,.2f}, S1: ${s1:,.2f}. Mild bullish bias."
            )
        elif price < pivot:
            # Below pivot - mild bearish
            pct_below = ((pivot - price) / pivot) * 100
            confidence = min(0.4 + pct_below / 5, 0.6)
            signal_type = SignalType.SELL
            reason = (
                f"BELOW PIVOT: Price ${price:,.2f} < Pivot ${pivot:,.2f}. "
                f"R1: ${r1:,.2f}, S1: ${s1:,.2f}. Mild bearish bias."
            )
        else:
            return None

        return Signal(
            timestamp=df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now(),
            asset=self.asset,
            signal_type=signal_type,
            strategy=self.name,
            price=price,
            confidence=confidence,
            reason=reason,
            indicators={'pivot': pivot, 'r1': r1, 's1': s1}
        )


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
        Get consensus signal from all strategies based on CURRENT indicator state.

        This method calls get_current_signal() on each strategy to evaluate the
        current market state, rather than accumulating historical crossover events.

        Returns:
            Tuple of (consensus_signal, confidence, contributing_signals)
        """
        current_signals = []

        # Get current signal from each strategy
        for strategy in self.strategies:
            signal = strategy.get_current_signal(df)
            if signal is not None and signal.asset == asset:
                current_signals.append(signal)

        if not current_signals:
            return SignalType.HOLD, 0.0, []

        # Aggregate signals based on current state
        total_score = sum(s.signal_type.value * s.confidence for s in current_signals)
        total_weight = sum(s.confidence for s in current_signals)

        if total_weight == 0:
            return SignalType.HOLD, 0.0, current_signals

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

        return consensus, confidence, current_signals


def generate_signal_context(
    df: pd.DataFrame,
    asset: Asset,
    signals: List[Signal]
) -> Dict[str, Any]:
    """
    Generate comprehensive market context for signals.

    Returns dict with:
    - confluence: How many strategies agree
    - market_regime: Trending/ranging, volatility state
    - key_levels: Support/resistance
    - risk_context: ATR-based stop suggestions
    """
    prefix = f"{asset.value}_"
    close_col = f"{prefix}close"

    if close_col not in df.columns or len(df) < 20:
        return {}

    current_price = df[close_col].iloc[-1]
    prices = df[close_col]

    # Confluence analysis
    buy_signals = [s for s in signals if s.signal_type.value > 0]
    sell_signals = [s for s in signals if s.signal_type.value < 0]
    total_strategies = len(set(s.strategy for s in signals)) if signals else 0

    confluence = {
        'bullish_count': len(buy_signals),
        'bearish_count': len(sell_signals),
        'total_strategies': max(total_strategies, 1),
        'agreement_pct': 0
    }

    if signals:
        dominant = max(len(buy_signals), len(sell_signals))
        confluence['agreement_pct'] = (dominant / len(signals)) * 100

    # Market regime detection
    sma_slow_col = f"{prefix}sma_slow"
    atr_col = f"{prefix}atr"

    regime = {
        'trend': 'NEUTRAL',
        'trend_strength': 0,
        'volatility': 'NORMAL',
        'volatility_pct': 0
    }

    if sma_slow_col in df.columns:
        sma_slow = df[sma_slow_col].iloc[-1]
        if not pd.isna(sma_slow):
            price_vs_sma = ((current_price - sma_slow) / sma_slow) * 100
            regime['trend_strength'] = abs(price_vs_sma)

            if price_vs_sma > 2:
                regime['trend'] = 'UPTREND'
            elif price_vs_sma < -2:
                regime['trend'] = 'DOWNTREND'
            else:
                regime['trend'] = 'RANGING'

    # Volatility regime
    if atr_col in df.columns:
        current_atr = df[atr_col].iloc[-1]
        if not pd.isna(current_atr):
            atr_pct = (current_atr / current_price) * 100
            regime['volatility_pct'] = atr_pct

            # Compare to historical ATR
            hist_atr = df[atr_col].rolling(60).mean().iloc[-1]
            if not pd.isna(hist_atr):
                atr_ratio = current_atr / hist_atr
                if atr_ratio > 1.5:
                    regime['volatility'] = 'HIGH'
                elif atr_ratio < 0.7:
                    regime['volatility'] = 'LOW'

    # Support/Resistance detection
    key_levels = {'support': [], 'resistance': []}

    # Simple pivot-based support/resistance
    rolling_min = prices.rolling(20).min()
    rolling_max = prices.rolling(20).max()

    # Find recent support levels (local minima)
    support_candidates = prices[prices == rolling_min].tail(5).values
    if len(support_candidates) > 0:
        key_levels['support'] = sorted(set(round(s, 2) for s in support_candidates if s < current_price))[-3:]

    # Find recent resistance levels (local maxima)
    resistance_candidates = prices[prices == rolling_max].tail(5).values
    if len(resistance_candidates) > 0:
        key_levels['resistance'] = sorted(set(round(r, 2) for r in resistance_candidates if r > current_price))[:3]

    # Risk context - ATR-based stop suggestions
    risk_context = {
        'suggested_stop_long': None,
        'suggested_stop_short': None,
        'atr_value': None,
        'risk_reward_note': ''
    }

    if atr_col in df.columns:
        current_atr = df[atr_col].iloc[-1]
        if not pd.isna(current_atr):
            risk_context['atr_value'] = current_atr
            risk_context['suggested_stop_long'] = round(current_price - (2 * current_atr), 2)
            risk_context['suggested_stop_short'] = round(current_price + (2 * current_atr), 2)

            # Risk/reward note based on nearest support/resistance
            if key_levels['resistance'] and key_levels['support']:
                nearest_resistance = min(key_levels['resistance']) if key_levels['resistance'] else current_price * 1.05
                nearest_support = max(key_levels['support']) if key_levels['support'] else current_price * 0.95

                upside = nearest_resistance - current_price
                downside = current_price - nearest_support

                if downside > 0:
                    rr_ratio = upside / downside
                    risk_context['risk_reward_note'] = f"Risk/Reward to nearest levels: {rr_ratio:.2f}:1"

    return {
        'confluence': confluence,
        'regime': regime,
        'key_levels': key_levels,
        'risk_context': risk_context,
        'current_price': current_price
    }


def create_default_strategies() -> CompositeStrategy:
    """Create default composite strategy with all sub-strategies"""
    composite = CompositeStrategy()

    # Add strategies for all assets (bull and bear)
    all_assets = [Asset.GOLD, Asset.SILVER, Asset.GOLD_BEAR, Asset.SILVER_BEAR]

    for asset in all_assets:
        # Core strategies
        composite.add_strategy(MACrossoverStrategy(asset=asset))
        composite.add_strategy(RSIMeanReversionStrategy(asset=asset))
        composite.add_strategy(BollingerBandStrategy(asset=asset))
        composite.add_strategy(MACDStrategy(asset=asset))

        # New advanced strategies
        composite.add_strategy(ADXTrendStrategy(asset=asset))
        composite.add_strategy(StochasticStrategy(asset=asset))
        composite.add_strategy(CCIStrategy(asset=asset))
        composite.add_strategy(IchimokuStrategy(asset=asset))
        composite.add_strategy(MFIStrategy(asset=asset))
        composite.add_strategy(PivotPointStrategy(asset=asset))

    # Add ratio strategy (for bull assets only)
    composite.add_strategy(GoldSilverRatioStrategy())

    return composite


def create_intraday_strategies() -> CompositeStrategy:
    """
    Create intraday-focused strategy with faster settings.

    Best strategies for intraday:
    1. RSI (fast) - Quick overbought/oversold signals
    2. Stochastic (fast) - Short-term momentum reversals
    3. MACD - Momentum confirmation
    4. Bollinger Bands - Volatility breakouts
    5. Pivot Points - Intraday S/R levels (specifically designed for intraday)
    """
    composite = CompositeStrategy()

    all_assets = [Asset.GOLD, Asset.SILVER, Asset.GOLD_BEAR, Asset.SILVER_BEAR]

    for asset in all_assets:
        # Fast RSI (7-period instead of 14, tighter bands)
        composite.add_strategy(RSIMeanReversionStrategy(
            rsi_period=7,
            overbought=65,  # Tighter than default 70
            oversold=35,    # Tighter than default 30
            asset=asset
        ))

        # Fast Stochastic (5,3,3 instead of 14,3,3)
        composite.add_strategy(StochasticStrategy(
            overbought=75,  # Tighter than default 80
            oversold=25,    # Tighter than default 20
            asset=asset
        ))

        # MACD - standard settings work well for intraday
        composite.add_strategy(MACDStrategy(asset=asset))

        # Bollinger Bands with tighter bands for quicker signals
        composite.add_strategy(BollingerBandStrategy(
            period=10,      # Faster than default 20
            std_dev=1.5,    # Tighter than default 2.0
            asset=asset
        ))

        # Pivot Points - best for intraday S/R
        composite.add_strategy(PivotPointStrategy(asset=asset))

    return composite


@dataclass
class IntradayLevels:
    """Entry/exit levels for intraday trading"""
    asset: Asset
    current_price: float
    signal: SignalType
    confidence: float

    # Entry levels
    entry_price: float
    entry_reason: str

    # Exit levels
    target_1: float  # First profit target
    target_2: float  # Second profit target
    stop_loss: float

    # Key levels
    pivot: float
    r1: float
    r2: float
    s1: float
    s2: float

    # Bollinger levels
    bb_upper: float
    bb_lower: float
    bb_mid: float

    # Risk metrics
    risk_amount: float  # Distance to stop
    reward_1: float     # Distance to target 1
    risk_reward_1: float
    reward_2: float     # Distance to target 2
    risk_reward_2: float


def calculate_intraday_levels(
    df: pd.DataFrame,
    asset: Asset,
    signal: SignalType,
    confidence: float
) -> Optional[IntradayLevels]:
    """
    Calculate exact entry/exit levels for intraday trading.

    Guarantees:
    - For LONG: targets > entry > stop_loss
    - For SHORT: targets < entry < stop_loss
    - Minimum 1:1 risk/reward ratio
    - All levels are distinct
    """
    prefix = f"{asset.value}_"

    # Required columns
    close_col = f"{prefix}close"
    pivot_col = f"{prefix}pivot"
    r1_col = f"{prefix}r1"
    r2_col = f"{prefix}r2"
    s1_col = f"{prefix}s1"
    s2_col = f"{prefix}s2"
    bb_upper_col = f"{prefix}bb_upper"
    bb_lower_col = f"{prefix}bb_lower"
    bb_mid_col = f"{prefix}bb_mid"
    atr_col = f"{prefix}atr"

    if close_col not in df.columns:
        return None

    row = df.iloc[-1]
    current_price = float(row[close_col])

    # Get ATR first - this is critical for all calculations
    atr = row.get(atr_col, current_price * 0.005)
    atr = float(current_price * 0.005 if pd.isna(atr) else atr)

    # Minimum ATR of 0.1% of price to avoid tiny movements
    atr = max(atr, current_price * 0.001)

    # Get pivot levels
    pivot = float(row.get(pivot_col, current_price) if not pd.isna(row.get(pivot_col)) else current_price)
    r1 = float(row.get(r1_col, current_price + atr) if not pd.isna(row.get(r1_col)) else current_price + atr)
    r2 = float(row.get(r2_col, current_price + 2*atr) if not pd.isna(row.get(r2_col)) else current_price + 2*atr)
    s1 = float(row.get(s1_col, current_price - atr) if not pd.isna(row.get(s1_col)) else current_price - atr)
    s2 = float(row.get(s2_col, current_price - 2*atr) if not pd.isna(row.get(s2_col)) else current_price - 2*atr)

    # Get Bollinger Bands
    bb_upper = float(row.get(bb_upper_col, current_price + 2*atr) if not pd.isna(row.get(bb_upper_col)) else current_price + 2*atr)
    bb_lower = float(row.get(bb_lower_col, current_price - 2*atr) if not pd.isna(row.get(bb_lower_col)) else current_price - 2*atr)
    bb_mid = float(row.get(bb_mid_col, current_price) if not pd.isna(row.get(bb_mid_col)) else current_price)

    if signal in [SignalType.BUY, SignalType.STRONG_BUY]:
        # LONG position
        entry_price = current_price
        entry_reason = "Enter LONG at market"

        # Stop loss: 1.5x ATR below entry (guaranteed distance)
        stop_loss = entry_price - (1.5 * atr)
        risk_amount = entry_price - stop_loss  # Always positive

        # Find targets ABOVE current price
        # Collect all resistance levels above entry
        resistance_levels = []
        if r1 > entry_price + (0.3 * atr):  # Must be meaningfully above
            resistance_levels.append(r1)
        if r2 > entry_price + (0.3 * atr):
            resistance_levels.append(r2)
        if bb_upper > entry_price + (0.3 * atr):
            resistance_levels.append(bb_upper)

        # Sort ascending
        resistance_levels = sorted(resistance_levels)

        # Target 1: First resistance above entry, or 1x ATR (minimum 1:1 R:R)
        if resistance_levels:
            target_1 = resistance_levels[0]
        else:
            target_1 = entry_price + (1.0 * atr)  # Fallback: 1x ATR = 1:1.5 R:R

        # Ensure minimum R:R of 1:1
        min_target_1 = entry_price + risk_amount
        target_1 = max(target_1, min_target_1)

        # Target 2: Second resistance above entry, or 2x ATR
        if len(resistance_levels) >= 2:
            target_2 = resistance_levels[1]
        elif len(resistance_levels) == 1:
            target_2 = target_1 + (1.0 * atr)
        else:
            target_2 = entry_price + (2.0 * atr)

        # Ensure target_2 > target_1
        target_2 = max(target_2, target_1 + (0.5 * atr))

    elif signal in [SignalType.SELL, SignalType.STRONG_SELL]:
        # SHORT position
        entry_price = current_price
        entry_reason = "Enter SHORT at market"

        # Stop loss: 1.5x ATR above entry (guaranteed distance)
        stop_loss = entry_price + (1.5 * atr)
        risk_amount = stop_loss - entry_price  # Always positive

        # Find targets BELOW current price
        # Collect all support levels below entry
        support_levels = []
        if s1 < entry_price - (0.3 * atr):  # Must be meaningfully below
            support_levels.append(s1)
        if s2 < entry_price - (0.3 * atr):
            support_levels.append(s2)
        if bb_lower < entry_price - (0.3 * atr):
            support_levels.append(bb_lower)

        # Sort descending (nearest first)
        support_levels = sorted(support_levels, reverse=True)

        # Target 1: First support below entry, or 1x ATR down
        if support_levels:
            target_1 = support_levels[0]
        else:
            target_1 = entry_price - (1.0 * atr)

        # Ensure minimum R:R of 1:1
        max_target_1 = entry_price - risk_amount
        target_1 = min(target_1, max_target_1)

        # Target 2: Second support below entry
        if len(support_levels) >= 2:
            target_2 = support_levels[1]
        elif len(support_levels) == 1:
            target_2 = target_1 - (1.0 * atr)
        else:
            target_2 = entry_price - (2.0 * atr)

        # Ensure target_2 < target_1
        target_2 = min(target_2, target_1 - (0.5 * atr))

    else:
        # HOLD - no entry
        return None

    # Calculate risk/reward
    reward_1 = abs(target_1 - entry_price)
    reward_2 = abs(target_2 - entry_price)

    risk_reward_1 = reward_1 / risk_amount if risk_amount > 0 else 0
    risk_reward_2 = reward_2 / risk_amount if risk_amount > 0 else 0

    return IntradayLevels(
        asset=asset,
        current_price=round(current_price, 2),
        signal=signal,
        confidence=confidence,
        entry_price=round(entry_price, 2),
        entry_reason=entry_reason,
        target_1=round(target_1, 2),
        target_2=round(target_2, 2),
        stop_loss=round(stop_loss, 2),
        pivot=round(pivot, 2),
        r1=round(r1, 2),
        r2=round(r2, 2),
        s1=round(s1, 2),
        s2=round(s2, 2),
        bb_upper=round(bb_upper, 2),
        bb_lower=round(bb_lower, 2),
        bb_mid=round(bb_mid, 2),
        risk_amount=round(risk_amount, 2),
        reward_1=round(reward_1, 2),
        risk_reward_1=round(risk_reward_1, 2),
        reward_2=round(reward_2, 2),
        risk_reward_2=round(risk_reward_2, 2)
    )


def get_intraday_signals_with_levels(
    df: pd.DataFrame,
    strategy: CompositeStrategy = None
) -> Dict[Asset, Dict]:
    """
    Get intraday signals with exact entry/exit levels for all assets.

    Returns dict with:
    - signal: SignalType
    - confidence: float
    - levels: IntradayLevels
    - strategies_agreeing: list of strategy names
    """
    if strategy is None:
        strategy = create_intraday_strategies()

    results = {}

    for asset in [Asset.GOLD, Asset.SILVER, Asset.GOLD_BEAR, Asset.SILVER_BEAR]:
        consensus, confidence, signals = strategy.get_consensus_signal(df, asset)

        # Get agreeing strategies
        buy_strategies = [s.strategy for s in signals if s.signal_type.value > 0]
        sell_strategies = [s.strategy for s in signals if s.signal_type.value < 0]

        if consensus.value > 0:
            agreeing = buy_strategies
        elif consensus.value < 0:
            agreeing = sell_strategies
        else:
            agreeing = []

        # Calculate entry/exit levels
        levels = calculate_intraday_levels(df, asset, consensus, confidence)

        results[asset] = {
            'signal': consensus,
            'confidence': confidence,
            'levels': levels,
            'strategies_agreeing': agreeing,
            'all_signals': signals
        }

    return results


def get_mirrored_signal(bull_signal: SignalType) -> SignalType:
    """
    Get the mirrored signal for bear ETFs.
    When bull shows SELL, bear should show BUY (and vice versa).
    """
    mirror_map = {
        SignalType.STRONG_BUY: SignalType.STRONG_SELL,
        SignalType.BUY: SignalType.SELL,
        SignalType.HOLD: SignalType.HOLD,
        SignalType.SELL: SignalType.BUY,
        SignalType.STRONG_SELL: SignalType.STRONG_BUY
    }
    return mirror_map.get(bull_signal, SignalType.HOLD)


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
