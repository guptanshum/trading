"""
Streamlit Dashboard for Gold & Silver ETF Trading
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    GOLD_TICKER, SILVER_TICKER,
    DASHBOARD_TITLE, DASHBOARD_REFRESH_SECONDS,
    CHART_HEIGHT, INITIAL_CAPITAL
)
from data_fetcher import DataFetcher, fetch_and_prepare_data
from indicators import TechnicalIndicators, add_indicators_to_data
from analysis import StatisticalAnalysis, analyze_market_data
from strategies import (
    create_default_strategies, Asset, SignalType,
    CompositeStrategy, generate_signal_context, Signal,
    get_mirrored_signal
)
from trading_engine import ShadowTradingEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_trading_engine():
    """Get or create trading engine (cached)"""
    os.makedirs("data", exist_ok=True)
    return ShadowTradingEngine()


@st.cache_resource
def get_strategy():
    """Get composite strategy (cached)"""
    return create_default_strategies()


@st.cache_data(ttl=300)  # Cache for 5 minutes to avoid rate limiting
def fetch_data(period: str = "6mo"):
    """Fetch and prepare market data with retry logic"""
    import time as retry_time

    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = fetch_and_prepare_data(period=period)
            if len(df) > 0:
                df = add_indicators_to_data(df)
                return df
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                wait_time = (attempt + 1) * 30  # 30s, 60s, 90s backoff
                logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                retry_time.sleep(wait_time)
            else:
                raise e

    # Return empty df if all retries failed
    return pd.DataFrame()


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_intraday_data():
    """Fetch intraday data for day trading"""
    fetcher = DataFetcher()
    try:
        data = fetcher.fetch_intraday(period="5d", interval="1m")
        combined = fetcher.get_combined_data(data)
        return add_indicators_to_data(combined) if len(combined) > 0 else combined
    except Exception as e:
        logger.error(f"Error fetching intraday data: {e}")
        return pd.DataFrame()


def create_price_chart(df: pd.DataFrame, asset: str, show_indicators: bool = True) -> go.Figure:
    """Create candlestick chart with indicators"""
    prefix = f"{asset}_"

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{asset.upper()} Price', 'RSI', 'MACD')
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df[f'{prefix}open'],
            high=df[f'{prefix}high'],
            low=df[f'{prefix}low'],
            close=df[f'{prefix}close'],
            name='Price'
        ),
        row=1, col=1
    )

    if show_indicators:
        # Moving averages
        if f'{prefix}sma_fast' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[f'{prefix}sma_fast'],
                          name='SMA Fast', line=dict(color='orange', width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df[f'{prefix}sma_slow'],
                          name='SMA Slow', line=dict(color='blue', width=1)),
                row=1, col=1
            )

        # Bollinger Bands
        if f'{prefix}bb_upper' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[f'{prefix}bb_upper'],
                          name='BB Upper', line=dict(color='gray', width=1, dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df[f'{prefix}bb_lower'],
                          name='BB Lower', line=dict(color='gray', width=1, dash='dash'),
                          fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
                row=1, col=1
            )

        # RSI
        if f'{prefix}rsi' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[f'{prefix}rsi'],
                          name='RSI', line=dict(color='purple', width=1)),
                row=2, col=1
            )
            # Overbought/Oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # MACD
        if f'{prefix}macd' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[f'{prefix}macd'],
                          name='MACD', line=dict(color='blue', width=1)),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df[f'{prefix}macd_signal'],
                          name='Signal', line=dict(color='orange', width=1)),
                row=3, col=1
            )
            fig.add_trace(
                go.Bar(x=df.index, y=df[f'{prefix}macd_hist'],
                      name='Histogram', marker_color='gray'),
                row=3, col=1
            )

    fig.update_layout(
        height=CHART_HEIGHT + 200,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )

    return fig


def create_ratio_chart(df: pd.DataFrame) -> go.Figure:
    """Create Gold/Silver ratio chart"""
    if 'gs_ratio' not in df.columns:
        return None

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['gs_ratio'],
            name='G/S Ratio',
            line=dict(color='gold', width=2)
        )
    )

    # Add mean line
    mean_ratio = df['gs_ratio'].mean()
    fig.add_hline(y=mean_ratio, line_dash="dash", line_color="gray",
                  annotation_text=f"Mean: {mean_ratio:.1f}")

    # Add +/- 2 std lines
    std_ratio = df['gs_ratio'].std()
    fig.add_hline(y=mean_ratio + 2*std_ratio, line_dash="dot", line_color="red",
                  annotation_text="Buy Silver Zone")
    fig.add_hline(y=mean_ratio - 2*std_ratio, line_dash="dot", line_color="green",
                  annotation_text="Buy Gold Zone")

    fig.update_layout(
        title="Gold/Silver Price Ratio",
        height=300,
        showlegend=True
    )

    return fig


def display_signal_card(
    asset: Asset,
    consensus: SignalType,
    confidence: float,
    signals: list,
    context: dict
):
    """Display a comprehensive signal card for an asset"""

    # Signal color and styling
    if consensus in [SignalType.BUY, SignalType.STRONG_BUY]:
        signal_color = "#28a745"
        bg_color = "rgba(40, 167, 69, 0.1)"
        signal_icon = "arrow_upward"
    elif consensus in [SignalType.SELL, SignalType.STRONG_SELL]:
        signal_color = "#dc3545"
        bg_color = "rgba(220, 53, 69, 0.1)"
        signal_icon = "arrow_downward"
    else:
        signal_color = "#6c757d"
        bg_color = "rgba(108, 117, 125, 0.1)"
        signal_icon = "remove"

    # Header with signal
    st.markdown(f"""
    <div style="background: {bg_color}; padding: 15px; border-radius: 10px; border-left: 4px solid {signal_color}; margin-bottom: 10px;">
        <h3 style="margin: 0; color: {signal_color};">{asset.value.upper()} - {consensus.name}</h3>
        <p style="margin: 5px 0 0 0; font-size: 1.1em;">Confidence: <strong>{confidence:.0%}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    # Confluence metrics
    if context and 'confluence' in context:
        conf = context['confluence']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Bullish Signals", conf['bullish_count'])
        with col2:
            st.metric("Bearish Signals", conf['bearish_count'])
        with col3:
            st.metric("Agreement", f"{conf['agreement_pct']:.0f}%")

    # Market Context
    if context and 'regime' in context:
        regime = context['regime']
        st.markdown("**Market Context:**")

        trend_color = "green" if regime['trend'] == 'UPTREND' else "red" if regime['trend'] == 'DOWNTREND' else "gray"
        vol_color = "red" if regime['volatility'] == 'HIGH' else "green" if regime['volatility'] == 'LOW' else "gray"

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"Trend: :{trend_color}[**{regime['trend']}**] ({regime['trend_strength']:.1f}% from MA)")
        with col2:
            st.markdown(f"Volatility: :{vol_color}[**{regime['volatility']}**] ({regime['volatility_pct']:.2f}% ATR)")

    # Key Levels
    if context and 'key_levels' in context:
        levels = context['key_levels']
        if levels['support'] or levels['resistance']:
            st.markdown("**Key Levels:**")
            col1, col2 = st.columns(2)
            with col1:
                if levels['support']:
                    support_str = ", ".join([f"${s:,.2f}" for s in levels['support']])
                    st.markdown(f":green[Support:] {support_str}")
                else:
                    st.markdown(":green[Support:] N/A")
            with col2:
                if levels['resistance']:
                    resistance_str = ", ".join([f"${r:,.2f}" for r in levels['resistance']])
                    st.markdown(f":red[Resistance:] {resistance_str}")
                else:
                    st.markdown(":red[Resistance:] N/A")

    # Risk Context
    if context and 'risk_context' in context:
        risk = context['risk_context']
        if risk['atr_value']:
            st.markdown("**Risk Management:**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"ATR: ${risk['atr_value']:.2f}")
                if risk['suggested_stop_long']:
                    st.markdown(f"Stop (Long): ${risk['suggested_stop_long']:,.2f}")
            with col2:
                if risk['suggested_stop_short']:
                    st.markdown(f"Stop (Short): ${risk['suggested_stop_short']:,.2f}")
                if risk['risk_reward_note']:
                    st.caption(risk['risk_reward_note'])

    # Individual Signals with Full Explanations
    st.markdown("---")
    st.markdown("**Individual Strategy Signals:**")

    if signals:
        for s in signals:
            signal_emoji = "+" if s.signal_type.value > 0 else "-" if s.signal_type.value < 0 else "~"
            s_color = "green" if s.signal_type.value > 0 else "red" if s.signal_type.value < 0 else "gray"

            with st.expander(f"[{signal_emoji}] {s.strategy}: {s.signal_type.name} ({s.confidence:.0%})", expanded=False):
                # Full reason - no truncation
                st.markdown(f"**Analysis:**")
                st.write(s.reason)

                # Show indicator values
                if s.indicators:
                    st.markdown("**Indicator Values:**")
                    indicator_cols = st.columns(min(len(s.indicators), 4))
                    for i, (key, value) in enumerate(s.indicators.items()):
                        with indicator_cols[i % len(indicator_cols)]:
                            if isinstance(value, float):
                                if abs(value) > 100:
                                    st.metric(key.replace('_', ' ').title(), f"${value:,.2f}")
                                else:
                                    st.metric(key.replace('_', ' ').title(), f"{value:.2f}")
                            else:
                                st.metric(key.replace('_', ' ').title(), str(value))
    else:
        st.info("No active signals from individual strategies")


def display_signals(df: pd.DataFrame, strategy: CompositeStrategy):
    """Display current trading signals with comprehensive analysis for all assets"""
    st.subheader("Trading Signals")

    # Last updated timestamp
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Bull ETFs Section
    st.markdown("### Bull ETFs (Long)")
    col1, col2 = st.columns(2)

    bull_signals = {}  # Store bull signals for mirror display

    for asset, col in [(Asset.GOLD, col1), (Asset.SILVER, col2)]:
        with col:
            consensus, confidence, signals = strategy.get_consensus_signal(df, asset)
            bull_signals[asset] = (consensus, confidence)

            # Generate comprehensive context
            context = generate_signal_context(df, asset, signals)

            # Display the rich signal card
            display_signal_card(asset, consensus, confidence, signals, context)

    # Bear ETFs Section
    st.markdown("---")
    st.markdown("### Bear ETFs (2x Inverse)")

    col3, col4 = st.columns(2)

    for asset, col, bull_asset in [
        (Asset.GOLD_BEAR, col3, Asset.GOLD),
        (Asset.SILVER_BEAR, col4, Asset.SILVER)
    ]:
        with col:
            # Get independent analysis for bear ETF
            consensus, confidence, signals = strategy.get_consensus_signal(df, asset)

            # Generate comprehensive context
            context = generate_signal_context(df, asset, signals)

            # Display the rich signal card
            display_signal_card(asset, consensus, confidence, signals, context)

            # Show mirrored signal from bull ETF
            bull_consensus, bull_confidence = bull_signals.get(bull_asset, (SignalType.HOLD, 0))
            mirrored = get_mirrored_signal(bull_consensus)

            mirror_color = "green" if mirrored.value > 0 else "red" if mirrored.value < 0 else "gray"
            st.markdown(f"""
            <div style="background: rgba(100,100,100,0.1); padding: 10px; border-radius: 5px; margin-top: 10px;">
                <strong>Mirrored Signal:</strong> {bull_asset.get_display_name()} is {bull_consensus.name}
                → Consider <span style="color: {mirror_color}; font-weight: bold;">{mirrored.name}</span> on {asset.get_display_name()}
            </div>
            """, unsafe_allow_html=True)


def display_portfolio(engine: ShadowTradingEngine):
    """Display portfolio status"""
    st.subheader("Shadow Portfolio")

    state = engine.get_portfolio_state()

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Value",
            f"${state.total_value:,.2f}",
            f"${state.total_pnl:,.2f}"
        )

    with col2:
        pnl_pct = state.total_pnl / INITIAL_CAPITAL * 100
        st.metric(
            "Total Return",
            f"{pnl_pct:.2f}%",
            f"${state.daily_pnl:,.2f} today"
        )

    with col3:
        st.metric(
            "Cash",
            f"${state.cash:,.2f}"
        )

    with col4:
        drawdown_pct = state.drawdown / state.peak_value * 100 if state.peak_value > 0 else 0
        st.metric(
            "Drawdown",
            f"{drawdown_pct:.2f}%",
            f"-${state.drawdown:,.2f}"
        )

    # Positions
    if state.positions:
        st.markdown("#### Open Positions")
        positions_data = []
        for asset, pos in state.positions.items():
            positions_data.append({
                'Asset': asset,
                'Quantity': f"{pos.quantity:.4f}",
                'Avg Price': f"${pos.avg_price:.2f}",
                'Current': f"${pos.current_price:.2f}",
                'P&L': f"${pos.unrealized_pnl:.2f}",
                'P&L %': f"{pos.unrealized_pnl_pct:.2f}%"
            })
        st.dataframe(pd.DataFrame(positions_data), hide_index=True)
    else:
        st.info("No open positions")


def display_trade_history(engine: ShadowTradingEngine):
    """Display recent trades"""
    st.subheader("Trade History")

    trades_df = engine.get_trade_history(limit=20)

    if trades_df.empty:
        st.info("No trades yet")
        return

    # Format the dataframe
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    trades_df['price'] = trades_df['price'].apply(lambda x: f"${x:.2f}")
    trades_df['value'] = trades_df['value'].apply(lambda x: f"${x:.2f}")
    trades_df['pnl'] = trades_df['pnl'].apply(lambda x: f"${x:.2f}")

    st.dataframe(
        trades_df[['timestamp', 'asset', 'side', 'quantity', 'price', 'value', 'pnl', 'strategy']],
        hide_index=True
    )


def display_performance(engine: ShadowTradingEngine):
    """Display performance metrics"""
    st.subheader("Performance Metrics")

    perf = engine.get_performance_summary()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", perf['total_trades'])
    with col2:
        st.metric("Win Rate", f"{perf['win_rate']:.1f}%")
    with col3:
        st.metric("Profit Factor", f"{perf['profit_factor']:.2f}")
    with col4:
        st.metric("Total P&L", f"${perf['total_pnl']:.2f}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg Win", f"${perf['avg_win']:.2f}")
    with col2:
        st.metric("Avg Loss", f"${perf['avg_loss']:.2f}")


def display_analysis(df: pd.DataFrame):
    """Display statistical analysis"""
    st.subheader("Market Analysis")

    analysis = analyze_market_data(df)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Gold/Silver Ratio")
        if 'ratio' in analysis:
            ratio = analysis['ratio']
            st.metric("Current Ratio", f"{ratio.current_ratio:.2f}")
            st.metric("Z-Score", f"{ratio.z_score:.2f}")
            st.metric("Signal", ratio.signal)

    with col2:
        st.markdown("#### Correlation")
        if 'correlation' in analysis:
            corr = analysis['correlation']
            st.metric("Correlation", f"{corr.correlation:.3f}")
            st.metric("Silver Beta", f"{corr.beta:.3f}")

    # Volatility
    st.markdown("#### Volatility Regime")
    col1, col2 = st.columns(2)

    with col1:
        if 'gold_volatility' in analysis:
            vol = analysis['gold_volatility']
            st.metric("Gold Volatility", f"{vol.current_volatility:.1f}%")
            st.caption(f"Regime: {vol.regime}")

    with col2:
        if 'silver_volatility' in analysis:
            vol = analysis['silver_volatility']
            st.metric("Silver Volatility", f"{vol.current_volatility:.1f}%")
            st.caption(f"Regime: {vol.regime}")


def main():
    """Main dashboard application"""
    st.title(f"{DASHBOARD_TITLE}")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Data period selection
        period = st.selectbox(
            "Historical Period",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=2
        )

        # View mode
        view_mode = st.radio(
            "View Mode",
            ["Daily", "Intraday"],
            index=0
        )

        # Show indicators
        show_indicators = st.checkbox("Show Indicators", value=True)

        # Auto refresh - enabled by default (5 min to avoid rate limits)
        auto_refresh = st.checkbox("Auto Refresh (5 min)", value=True)

        st.divider()

        # Manual actions
        st.header("Actions")

        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        if st.button("Save Portfolio Snapshot"):
            engine = get_trading_engine()
            engine.save_snapshot()
            st.success("Snapshot saved!")

    # Load data
    try:
        if view_mode == "Daily":
            df = fetch_data(period)
        else:
            df = fetch_intraday_data()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return

    if df.empty:
        st.warning("No data available. Market might be closed.")
        return

    # Data loaded timestamp - prominent display
    data_time = datetime.now()
    if auto_refresh:
        st.success(f"Data loaded at {data_time.strftime('%H:%M:%S')} | Auto-refresh enabled (5 min)")
    else:
        st.info(f"Data loaded at {data_time.strftime('%H:%M:%S')} | Auto-refresh disabled")

    # Get strategy and engine
    strategy = get_strategy()
    engine = get_trading_engine()

    # Update prices in engine
    if 'gold_close' in df.columns and 'silver_close' in df.columns:
        engine.update_prices(
            df['gold_close'].iloc[-1],
            df['silver_close'].iloc[-1]
        )

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Charts", "Portfolio", "Signals", "Analysis"
    ])

    with tab1:
        # Bull ETF Charts
        st.markdown("### Bull ETFs (Long)")
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                create_price_chart(df, "gold", show_indicators),
                use_container_width=True
            )

        with col2:
            st.plotly_chart(
                create_price_chart(df, "silver", show_indicators),
                use_container_width=True
            )

        # Bear ETF Charts (if data available)
        if 'gold_bear_close' in df.columns or 'silver_bear_close' in df.columns:
            st.markdown("---")
            st.markdown("### Bear ETFs (2x Inverse)")
            col3, col4 = st.columns(2)

            with col3:
                if 'gold_bear_close' in df.columns:
                    st.plotly_chart(
                        create_price_chart(df, "gold_bear", show_indicators),
                        use_container_width=True
                    )
                else:
                    st.info("Gold Bear (GLL) data not available")

            with col4:
                if 'silver_bear_close' in df.columns:
                    st.plotly_chart(
                        create_price_chart(df, "silver_bear", show_indicators),
                        use_container_width=True
                    )
                else:
                    st.info("Silver Bear (ZSL) data not available")

        # Ratio chart
        ratio_chart = create_ratio_chart(df)
        if ratio_chart:
            st.plotly_chart(ratio_chart, use_container_width=True)

    with tab2:
        display_portfolio(engine)
        st.divider()
        display_trade_history(engine)
        st.divider()
        display_performance(engine)

    with tab3:
        display_signals(df, strategy)

        st.divider()

        # Execute signals section
        st.subheader("Execute Signals")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Execute Gold Signal", type="primary"):
                consensus, confidence, _ = strategy.get_consensus_signal(df, Asset.GOLD)
                if consensus != SignalType.HOLD:
                    from strategies import Signal
                    signal = Signal(
                        timestamp=datetime.now(),
                        asset=Asset.GOLD,
                        signal_type=consensus,
                        strategy="Manual",
                        price=df['gold_close'].iloc[-1],
                        confidence=confidence,
                        reason="Manual execution from dashboard"
                    )
                    trade = engine.execute_signal(signal)
                    if trade:
                        st.success(f"Executed: {trade.side} {trade.quantity:.4f} @ ${trade.price:.2f}")
                    else:
                        st.warning("Trade rejected (check risk limits)")
                else:
                    st.info("No signal to execute (HOLD)")

        with col2:
            if st.button("Execute Silver Signal", type="primary"):
                consensus, confidence, _ = strategy.get_consensus_signal(df, Asset.SILVER)
                if consensus != SignalType.HOLD:
                    from strategies import Signal
                    signal = Signal(
                        timestamp=datetime.now(),
                        asset=Asset.SILVER,
                        signal_type=consensus,
                        strategy="Manual",
                        price=df['silver_close'].iloc[-1],
                        confidence=confidence,
                        reason="Manual execution from dashboard"
                    )
                    trade = engine.execute_signal(signal)
                    if trade:
                        st.success(f"Executed: {trade.side} {trade.quantity:.4f} @ ${trade.price:.2f}")
                    else:
                        st.warning("Trade rejected (check risk limits)")
                else:
                    st.info("No signal to execute (HOLD)")

    with tab4:
        display_analysis(df)

    # Footer
    st.divider()
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Auto refresh (5 min = 300s to avoid Yahoo Finance rate limits)
    if auto_refresh:
        time.sleep(300)
        st.rerun()


if __name__ == "__main__":
    main()
