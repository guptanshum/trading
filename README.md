# Gold & Silver ETF Trading App

A comprehensive trading analysis and shadow trading system for gold (GLD) and silver (SIVR) ETFs, with optional Avanza broker integration.

## Features

- **Historical Data Analysis**: Fetch and analyze historical price data
- **Technical Indicators**: MA, RSI, MACD, Bollinger Bands, ATR, Stochastic
- **Statistical Analysis**: Gold/Silver ratio, correlation, mean reversion, seasonality
- **Multiple Trading Strategies**: Trend following, mean reversion, ratio trading
- **Shadow Trading**: Virtual portfolio with $100K capital for strategy testing
- **Risk Management**: Stop loss, take profit, max drawdown limits
- **Web Dashboard**: Interactive Streamlit dashboard with charts
- **Avanza Integration**: Optional connection to Avanza broker (ISK account)

## Quick Start

### 1. Set Up Virtual Environment

```bash
cd /Users/user/workspace/trading

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
source venv/bin/activate  # Activate venv if not already
python main.py dashboard
```

This opens a web browser with the interactive trading dashboard.

### 3. Other Commands

```bash
# Run market analysis
python main.py analyze

# Start automated shadow trading
python main.py trade

# Check portfolio status
python main.py status
```

## Project Structure

```
trading/
├── main.py              # Main entry point
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── venv/               # Python virtual environment
├── data/               # SQLite database and data files
├── src/
│   ├── __init__.py     # Package exports
│   ├── config.py       # Configuration settings
│   ├── data_fetcher.py # Yahoo Finance data fetching
│   ├── indicators.py   # Technical indicators
│   ├── analysis.py     # Statistical analysis
│   ├── strategies.py   # Trading strategies
│   ├── trading_engine.py # Shadow trading engine
│   ├── avanza_api.py   # Avanza broker integration
│   └── dashboard.py    # Streamlit web dashboard
└── tests/              # Unit tests
```

## Configuration

### ETFs Tracked
- **GLD**: SPDR Gold Shares (Gold ETF)
- **SIVR**: abrdn Physical Silver Shares (Silver ETF)

Both are available on Avanza for Swedish investors.

### Shadow Trading Settings (config.py)
- Initial Capital: $100,000
- Max Position Size: 25% of portfolio
- Stop Loss: 2%
- Take Profit: 5%
- Max Daily Loss: 3%
- Max Drawdown: 10%

## Avanza Integration

To enable Avanza API integration:

1. Set environment variables:
```bash
export AVANZA_USERNAME="your_username"
export AVANZA_PASSWORD="your_password"
export AVANZA_TOTP_SECRET="your_totp_secret"
export AVANZA_ACCOUNT_ID="your_isk_account_id"
export AVANZA_ENABLED="true"
```

2. To enable real trading (USE WITH EXTREME CAUTION):
```bash
export AVANZA_ALLOW_TRADING="true"
```

### Finding Your TOTP Secret
1. Log into Avanza web
2. Go to Settings > Security > Two-factor authentication
3. When setting up a new authenticator, you'll see a QR code
4. The TOTP secret is encoded in the QR code URL

## Trading Strategies

### 1. MA Crossover
Generates buy/sell signals when fast MA crosses slow MA.

### 2. RSI Mean Reversion
Buys when RSI bounces from oversold (<30), sells when RSI falls from overbought (>70).

### 3. Bollinger Band
Buys at lower band bounces, sells at upper band rejections.

### 4. MACD
Generates signals on MACD/Signal line crossovers.

### 5. Gold/Silver Ratio
Mean reversion strategy based on the G/S ratio:
- Buy silver when ratio > 2 std devs above mean
- Buy gold when ratio < 2 std devs below mean

## Dashboard Features

### Charts Tab
- Candlestick charts for gold and silver
- Technical indicators overlay
- Gold/Silver ratio chart

### Portfolio Tab
- Current portfolio value and P&L
- Open positions
- Trade history
- Performance metrics

### Signals Tab
- Current trading signals
- Signal confidence and details
- Manual trade execution

### Analysis Tab
- G/S ratio analysis
- Correlation metrics
- Volatility regime

## API Reference

### DataFetcher
```python
from src.data_fetcher import DataFetcher

fetcher = DataFetcher()
data = fetcher.fetch_historical_daily("1y")
intraday = fetcher.fetch_intraday("5d", "1m")
```

### TechnicalIndicators
```python
from src.indicators import TechnicalIndicators

indicators = TechnicalIndicators()
rsi = indicators.rsi(prices, period=14)
macd, signal, hist = indicators.macd(prices)
```

### ShadowTradingEngine
```python
from src.trading_engine import ShadowTradingEngine

engine = ShadowTradingEngine()
engine.execute_signal(signal)
state = engine.get_portfolio_state()
```

## Disclaimer

This software is for educational and research purposes only. It is NOT financial advice. Trading ETFs involves risk and you may lose money. Always do your own research and consider consulting a financial advisor before making investment decisions.

The shadow trading feature simulates trades without using real money. If you enable Avanza integration with real trading, you are responsible for any financial losses.

## License

MIT License - See LICENSE file for details.
