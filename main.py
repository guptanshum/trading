#!/usr/bin/env python3
"""
Gold & Silver ETF Trading Application

Main entry point for running the trading system.

Usage:
    # Run the Streamlit dashboard
    python main.py dashboard

    # Run the analysis CLI
    python main.py analyze

    # Run automated shadow trading (background)
    python main.py trade

    # Show current portfolio status
    python main.py status
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config import (
    GOLD_TICKER, SILVER_TICKER,
    LIVE_REFRESH_SECONDS, INITIAL_CAPITAL
)
from data_fetcher import DataFetcher, fetch_and_prepare_data
from indicators import add_indicators_to_data
from analysis import analyze_market_data
from strategies import create_default_strategies, Asset, SignalType
from trading_engine import ShadowTradingEngine
from avanza_api import get_avanza_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_dashboard():
    """Launch the Streamlit dashboard"""
    import subprocess
    dashboard_path = os.path.join(os.path.dirname(__file__), 'src', 'dashboard.py')
    subprocess.run(['streamlit', 'run', dashboard_path])


def run_analysis():
    """Run analysis and print results to console"""
    print("\n" + "=" * 60)
    print("Gold & Silver ETF Analysis")
    print("=" * 60)

    # Fetch data
    print("\nFetching market data...")
    df = fetch_and_prepare_data(period="6mo")
    df = add_indicators_to_data(df)

    print(f"Data range: {df.index[0]} to {df.index[-1]}")
    print(f"Total bars: {len(df)}")

    # Current prices
    gold_price = df['gold_close'].iloc[-1]
    silver_price = df['silver_close'].iloc[-1]
    ratio = gold_price / silver_price

    print(f"\n--- Current Prices ---")
    print(f"Gold ({GOLD_TICKER}):   ${gold_price:.2f}")
    print(f"Silver ({SILVER_TICKER}): ${silver_price:.2f}")
    print(f"G/S Ratio:      {ratio:.2f}")

    # Run analysis
    print("\n--- Market Analysis ---")
    analysis = analyze_market_data(df)

    if 'ratio' in analysis:
        r = analysis['ratio']
        print(f"\nGold/Silver Ratio:")
        print(f"  Current: {r.current_ratio:.2f}")
        print(f"  Mean: {r.mean_ratio:.2f}")
        print(f"  Z-Score: {r.z_score:.2f}")
        print(f"  Signal: {r.signal}")

    if 'correlation' in analysis:
        c = analysis['correlation']
        print(f"\nCorrelation Analysis:")
        print(f"  Correlation: {c.correlation:.3f}")
        print(f"  Silver Beta: {c.beta:.3f}")

    # Generate signals
    print("\n--- Trading Signals ---")
    strategy = create_default_strategies()

    for asset in [Asset.GOLD, Asset.SILVER]:
        consensus, confidence, signals = strategy.get_consensus_signal(df, asset)
        print(f"\n{asset.value.upper()}:")
        print(f"  Signal: {consensus.name}")
        print(f"  Confidence: {confidence:.0%}")
        if signals:
            print(f"  Contributing strategies: {len(signals)}")

    # Technical indicators
    print("\n--- Technical Indicators ---")
    for asset in ['gold', 'silver']:
        print(f"\n{asset.upper()}:")
        last = df.iloc[-1]
        print(f"  RSI: {last[f'{asset}_rsi']:.1f}")
        print(f"  MACD: {last[f'{asset}_macd']:.3f}")
        print(f"  BB Width: {last[f'{asset}_bb_width']:.3f}")

    print("\n" + "=" * 60)


def run_shadow_trading():
    """Run automated shadow trading loop"""
    print("\n" + "=" * 60)
    print("Shadow Trading Mode")
    print("=" * 60)
    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Refresh Interval: {LIVE_REFRESH_SECONDS}s")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    # Initialize components
    os.makedirs("data", exist_ok=True)
    engine = ShadowTradingEngine()
    strategy = create_default_strategies()
    fetcher = DataFetcher()

    iteration = 0

    try:
        while True:
            iteration += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            print(f"\n[{timestamp}] Iteration {iteration}")
            print("-" * 40)

            try:
                # Fetch latest data
                df = fetch_and_prepare_data(period="1mo")
                df = add_indicators_to_data(df)

                if df.empty:
                    print("No data available (market closed?)")
                    time.sleep(LIVE_REFRESH_SECONDS)
                    continue

                # Get current prices
                gold_price = df['gold_close'].iloc[-1]
                silver_price = df['silver_close'].iloc[-1]

                print(f"Gold: ${gold_price:.2f} | Silver: ${silver_price:.2f}")

                # Update engine prices
                engine.update_prices(gold_price, silver_price)

                # Get portfolio state
                state = engine.get_portfolio_state()
                print(f"Portfolio: ${state.total_value:,.2f} (P&L: ${state.total_pnl:,.2f})")

                # Generate and execute signals
                for asset in [Asset.GOLD, Asset.SILVER]:
                    consensus, confidence, _ = strategy.get_consensus_signal(df, asset)

                    if consensus not in [SignalType.HOLD]:
                        print(f"  {asset.value}: {consensus.name} ({confidence:.0%})")

                        # Create signal
                        price = gold_price if asset == Asset.GOLD else silver_price
                        from strategies import Signal
                        signal = Signal(
                            timestamp=datetime.now(),
                            asset=asset,
                            signal_type=consensus,
                            strategy="AutoTrader",
                            price=price,
                            confidence=confidence,
                            reason=f"Automated signal from composite strategy"
                        )

                        # Execute if strong signal
                        if confidence >= 0.6:
                            trade = engine.execute_signal(signal)
                            if trade:
                                print(f"    -> Executed: {trade.side} {trade.quantity:.4f}")
                    else:
                        print(f"  {asset.value}: HOLD")

                # Save periodic snapshot
                if iteration % 10 == 0:
                    engine.save_snapshot()
                    print("  [Snapshot saved]")

            except Exception as e:
                logger.error(f"Error in trading loop: {e}")

            # Wait for next iteration
            time.sleep(LIVE_REFRESH_SECONDS)

    except KeyboardInterrupt:
        print("\n\nStopping shadow trading...")
        engine.save_snapshot()
        print("Final snapshot saved.")

        # Print summary
        state = engine.get_portfolio_state()
        print(f"\n--- Final Portfolio ---")
        print(f"Total Value: ${state.total_value:,.2f}")
        print(f"Total P&L: ${state.total_pnl:,.2f}")
        print(f"Return: {state.total_pnl / INITIAL_CAPITAL * 100:.2f}%")


def show_status():
    """Show current portfolio status"""
    print("\n" + "=" * 60)
    print("Portfolio Status")
    print("=" * 60)

    os.makedirs("data", exist_ok=True)
    engine = ShadowTradingEngine()

    state = engine.get_portfolio_state()

    print(f"\n--- Account Summary ---")
    print(f"Cash:           ${state.cash:,.2f}")
    print(f"Positions:      ${state.total_value - state.cash:,.2f}")
    print(f"Total Value:    ${state.total_value:,.2f}")
    print(f"Total P&L:      ${state.total_pnl:,.2f}")
    print(f"Return:         {state.total_pnl / INITIAL_CAPITAL * 100:.2f}%")
    print(f"Peak Value:     ${state.peak_value:,.2f}")
    print(f"Drawdown:       ${state.drawdown:,.2f}")

    if state.positions:
        print(f"\n--- Open Positions ---")
        for asset, pos in state.positions.items():
            print(f"\n{asset}:")
            print(f"  Quantity:     {pos.quantity:.4f}")
            print(f"  Avg Price:    ${pos.avg_price:.2f}")
            print(f"  Current:      ${pos.current_price:.2f}")
            print(f"  P&L:          ${pos.unrealized_pnl:.2f} ({pos.unrealized_pnl_pct:.2f}%)")
    else:
        print("\nNo open positions")

    # Performance
    print(f"\n--- Performance Metrics ---")
    perf = engine.get_performance_summary()
    print(f"Total Trades:   {perf['total_trades']}")
    print(f"Winning:        {perf['winning_trades']}")
    print(f"Losing:         {perf['losing_trades']}")
    print(f"Win Rate:       {perf['win_rate']:.1f}%")
    print(f"Profit Factor:  {perf['profit_factor']:.2f}")

    # Recent trades
    trades = engine.get_trade_history(limit=5)
    if not trades.empty:
        print(f"\n--- Recent Trades ---")
        for _, t in trades.iterrows():
            print(f"  {t['timestamp']}: {t['side']} {t['asset']} "
                  f"@ ${t['price']:.2f} (P&L: ${t['pnl']:.2f})")

    # Avanza status
    print(f"\n--- Avanza Integration ---")
    client = get_avanza_client()
    if client.is_configured():
        print("Status: Configured")
        print("Trading: " + ("Enabled" if os.getenv("AVANZA_ALLOW_TRADING") else "Disabled"))
    else:
        print("Status: Not configured")
        print("Set AVANZA_* environment variables to enable")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Gold & Silver ETF Trading Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  dashboard   Launch the Streamlit web dashboard
  analyze     Run market analysis and show results
  trade       Start automated shadow trading loop
  status      Show current portfolio status

Examples:
  python main.py dashboard
  python main.py analyze
  python main.py trade
  python main.py status
        """
    )

    parser.add_argument(
        'command',
        choices=['dashboard', 'analyze', 'trade', 'status'],
        help='Command to run'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    if args.command == 'dashboard':
        run_dashboard()
    elif args.command == 'analyze':
        run_analysis()
    elif args.command == 'trade':
        run_shadow_trading()
    elif args.command == 'status':
        show_status()


if __name__ == "__main__":
    main()
