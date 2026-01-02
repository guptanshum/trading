"""
Statistical analysis module for Gold/Silver trading
"""
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from scipy import stats

from config import StrategySettings


@dataclass
class RatioAnalysis:
    """Gold/Silver ratio analysis results"""
    current_ratio: float
    mean_ratio: float
    std_ratio: float
    z_score: float
    percentile: float
    signal: str  # 'BUY_SILVER', 'BUY_GOLD', 'NEUTRAL'


@dataclass
class CorrelationAnalysis:
    """Correlation analysis results"""
    correlation: float
    rolling_correlation: pd.Series
    beta: float  # Silver beta to gold
    r_squared: float


@dataclass
class SeasonalPattern:
    """Seasonal pattern analysis"""
    monthly_returns: pd.Series
    best_months: list
    worst_months: list
    day_of_week_returns: pd.Series


@dataclass
class VolatilityAnalysis:
    """Volatility clustering analysis"""
    current_volatility: float
    historical_volatility: float
    volatility_percentile: float
    regime: str  # 'HIGH', 'NORMAL', 'LOW'


class StatisticalAnalysis:
    """Statistical analysis for Gold/Silver trading"""

    def __init__(self, settings: Optional[StrategySettings] = None):
        self.settings = settings or StrategySettings()

    def analyze_gold_silver_ratio(
        self,
        gold_prices: pd.Series,
        silver_prices: pd.Series,
        lookback: int = 252  # ~1 year of trading days
    ) -> RatioAnalysis:
        """
        Analyze the Gold/Silver price ratio

        The ratio indicates how many units of silver equal one unit of gold.
        Mean reversion strategies can trade when ratio deviates significantly.
        """
        # Calculate ratio
        ratio = gold_prices / silver_prices
        ratio = ratio.dropna()

        if len(ratio) < lookback:
            lookback = len(ratio)

        # Calculate statistics over lookback period
        recent_ratio = ratio.iloc[-lookback:]
        current = ratio.iloc[-1]
        mean = recent_ratio.mean()
        std = recent_ratio.std()

        # Z-score shows how many std devs from mean
        z_score = (current - mean) / std if std > 0 else 0

        # Percentile ranking
        percentile = stats.percentileofscore(recent_ratio, current)

        # Generate signal based on z-score
        if z_score > 2:  # Ratio very high - silver undervalued
            signal = 'BUY_SILVER'
        elif z_score < -2:  # Ratio very low - gold undervalued
            signal = 'BUY_GOLD'
        else:
            signal = 'NEUTRAL'

        return RatioAnalysis(
            current_ratio=current,
            mean_ratio=mean,
            std_ratio=std,
            z_score=z_score,
            percentile=percentile,
            signal=signal
        )

    def analyze_correlation(
        self,
        gold_returns: pd.Series,
        silver_returns: pd.Series,
        window: int = 30
    ) -> CorrelationAnalysis:
        """
        Analyze correlation between gold and silver returns

        Args:
            gold_returns: Gold daily returns
            silver_returns: Silver daily returns
            window: Rolling window for correlation
        """
        # Align data
        aligned = pd.DataFrame({
            'gold': gold_returns,
            'silver': silver_returns
        }).dropna()

        # Overall correlation
        correlation = aligned['gold'].corr(aligned['silver'])

        # Rolling correlation
        rolling_corr = aligned['gold'].rolling(window).corr(aligned['silver'])

        # Linear regression: Silver = alpha + beta * Gold
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            aligned['gold'], aligned['silver']
        )

        return CorrelationAnalysis(
            correlation=correlation,
            rolling_correlation=rolling_corr,
            beta=slope,
            r_squared=r_value ** 2
        )

    def analyze_mean_reversion(
        self,
        prices: pd.Series,
        window: int = 20
    ) -> Tuple[float, float, str]:
        """
        Analyze mean reversion potential

        Returns:
            Tuple of (z_score, half_life, signal)
        """
        # Calculate z-score from rolling mean
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()

        z_score = (prices.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]

        # Estimate half-life of mean reversion using OLS
        # Price change vs deviation from mean
        price_diff = prices.diff().dropna()
        deviation = (prices - rolling_mean).shift(1).dropna()

        # Align series
        aligned_idx = price_diff.index.intersection(deviation.index)
        price_diff = price_diff.loc[aligned_idx]
        deviation = deviation.loc[aligned_idx]

        if len(deviation) > 10 and deviation.std() > 0:
            slope, _, _, _, _ = stats.linregress(deviation, price_diff)
            if slope < 0:
                half_life = -np.log(2) / slope
            else:
                half_life = np.inf
        else:
            half_life = np.inf

        # Signal
        if z_score > 2:
            signal = 'SELL'  # Overbought
        elif z_score < -2:
            signal = 'BUY'  # Oversold
        else:
            signal = 'HOLD'

        return z_score, half_life, signal

    def analyze_seasonality(
        self,
        prices: pd.Series
    ) -> SeasonalPattern:
        """
        Analyze seasonal patterns in returns

        Returns monthly and day-of-week average returns
        """
        returns = prices.pct_change().dropna()

        # Monthly returns
        monthly = returns.groupby(returns.index.month).mean() * 100
        monthly.index = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(monthly)]

        # Best and worst months
        sorted_months = monthly.sort_values(ascending=False)
        best_months = sorted_months.head(3).index.tolist()
        worst_months = sorted_months.tail(3).index.tolist()

        # Day of week returns (0=Monday, 4=Friday)
        dow = returns.groupby(returns.index.dayofweek).mean() * 100
        dow.index = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'][:len(dow)]

        return SeasonalPattern(
            monthly_returns=monthly,
            best_months=best_months,
            worst_months=worst_months,
            day_of_week_returns=dow
        )

    def analyze_volatility(
        self,
        prices: pd.Series,
        window: int = 20,
        annual_factor: int = 252
    ) -> VolatilityAnalysis:
        """
        Analyze volatility regime

        Returns current volatility, historical context, and regime
        """
        returns = prices.pct_change().dropna()

        # Rolling volatility (annualized)
        rolling_vol = returns.rolling(window).std() * np.sqrt(annual_factor)

        current_vol = rolling_vol.iloc[-1]
        hist_vol = rolling_vol.mean()

        # Percentile of current volatility
        vol_percentile = stats.percentileofscore(rolling_vol.dropna(), current_vol)

        # Determine regime
        if vol_percentile > 80:
            regime = 'HIGH'
        elif vol_percentile < 20:
            regime = 'LOW'
        else:
            regime = 'NORMAL'

        return VolatilityAnalysis(
            current_volatility=current_vol * 100,  # As percentage
            historical_volatility=hist_vol * 100,
            volatility_percentile=vol_percentile,
            regime=regime
        )

    def detect_support_resistance(
        self,
        prices: pd.Series,
        window: int = 20,
        num_levels: int = 3
    ) -> Tuple[list, list]:
        """
        Detect support and resistance levels

        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        # Find local minima and maxima
        rolling_min = prices.rolling(window, center=True).min()
        rolling_max = prices.rolling(window, center=True).max()

        # Points where price equals rolling min/max
        is_support = prices == rolling_min
        is_resistance = prices == rolling_max

        support_prices = prices[is_support].values
        resistance_prices = prices[is_resistance].values

        # Cluster nearby levels and get most significant
        support_levels = self._cluster_levels(support_prices, num_levels)
        resistance_levels = self._cluster_levels(resistance_prices, num_levels)

        return support_levels, resistance_levels

    def _cluster_levels(self, prices: np.ndarray, num_levels: int) -> list:
        """Cluster nearby price levels and return most significant"""
        if len(prices) < num_levels:
            return sorted(prices.tolist())

        # Simple clustering: histogram approach
        hist, bin_edges = np.histogram(prices, bins=num_levels * 2)
        significant_bins = np.argsort(hist)[-num_levels:]

        levels = []
        for bin_idx in significant_bins:
            # Use bin center as level
            level = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
            levels.append(level)

        return sorted(levels)

    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate percentage returns"""
        return prices.pct_change().dropna()

    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns"""
        return np.log(prices / prices.shift(1)).dropna()


def analyze_market_data(df: pd.DataFrame) -> dict:
    """
    Comprehensive analysis of gold/silver market data

    Args:
        df: DataFrame with gold_close and silver_close columns

    Returns:
        Dictionary with all analysis results
    """
    analyzer = StatisticalAnalysis()

    results = {}

    if 'gold_close' in df.columns and 'silver_close' in df.columns:
        # Gold/Silver ratio analysis
        results['ratio'] = analyzer.analyze_gold_silver_ratio(
            df['gold_close'], df['silver_close']
        )

        # Correlation analysis
        gold_returns = analyzer.calculate_returns(df['gold_close'])
        silver_returns = analyzer.calculate_returns(df['silver_close'])
        results['correlation'] = analyzer.analyze_correlation(
            gold_returns, silver_returns
        )

    # Individual asset analysis
    for asset in ['gold', 'silver']:
        close_col = f'{asset}_close'
        if close_col in df.columns:
            prices = df[close_col]

            # Mean reversion
            z_score, half_life, signal = analyzer.analyze_mean_reversion(prices)
            results[f'{asset}_mean_reversion'] = {
                'z_score': z_score,
                'half_life': half_life,
                'signal': signal
            }

            # Seasonality
            results[f'{asset}_seasonality'] = analyzer.analyze_seasonality(prices)

            # Volatility
            results[f'{asset}_volatility'] = analyzer.analyze_volatility(prices)

            # Support/Resistance
            support, resistance = analyzer.detect_support_resistance(prices)
            results[f'{asset}_levels'] = {
                'support': support,
                'resistance': resistance
            }

    return results


if __name__ == "__main__":
    # Test the analysis module
    from data_fetcher import fetch_and_prepare_data

    print("Fetching data for analysis...")
    df = fetch_and_prepare_data(period="1y")

    print("\nRunning analysis...")
    results = analyze_market_data(df)

    print("\n=== Gold/Silver Ratio Analysis ===")
    ratio = results['ratio']
    print(f"Current Ratio: {ratio.current_ratio:.2f}")
    print(f"Mean Ratio: {ratio.mean_ratio:.2f}")
    print(f"Z-Score: {ratio.z_score:.2f}")
    print(f"Percentile: {ratio.percentile:.1f}%")
    print(f"Signal: {ratio.signal}")

    print("\n=== Correlation Analysis ===")
    corr = results['correlation']
    print(f"Correlation: {corr.correlation:.3f}")
    print(f"Silver Beta to Gold: {corr.beta:.3f}")
    print(f"R-Squared: {corr.r_squared:.3f}")

    print("\n=== Gold Volatility ===")
    vol = results['gold_volatility']
    print(f"Current Vol: {vol.current_volatility:.1f}%")
    print(f"Historical Vol: {vol.historical_volatility:.1f}%")
    print(f"Regime: {vol.regime}")

    print("\n=== Gold Mean Reversion ===")
    mr = results['gold_mean_reversion']
    print(f"Z-Score: {mr['z_score']:.2f}")
    print(f"Half-Life: {mr['half_life']:.1f} days")
    print(f"Signal: {mr['signal']}")
