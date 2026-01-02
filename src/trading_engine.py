"""
Shadow Trading Engine with Portfolio Management and SQLite Storage
"""
import logging
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass, field
from enum import Enum
import json

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.orm import Session
import pandas as pd

from config import (
    INITIAL_CAPITAL,
    MAX_POSITION_SIZE_PCT,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MAX_DAILY_LOSS_PCT,
    MAX_DRAWDOWN_PCT,
    DATABASE_PATH
)
from strategies import Signal, SignalType, Asset

logger = logging.getLogger(__name__)

Base = declarative_base()


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


# SQLAlchemy Models
class Trade(Base):
    """Database model for trades"""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    asset = Column(String(20))
    side = Column(String(10))
    quantity = Column(Float)
    price = Column(Float)
    value = Column(Float)
    commission = Column(Float, default=0)
    strategy = Column(String(50))
    signal_confidence = Column(Float)
    reason = Column(Text)
    status = Column(String(20), default=OrderStatus.FILLED.value)
    pnl = Column(Float, default=0)  # Realized P&L for closing trades


class Position(Base):
    """Database model for open positions"""
    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True)
    asset = Column(String(20), unique=True)
    quantity = Column(Float, default=0)
    avg_price = Column(Float, default=0)
    current_price = Column(Float, default=0)
    unrealized_pnl = Column(Float, default=0)
    unrealized_pnl_pct = Column(Float, default=0)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)


class PortfolioSnapshot(Base):
    """Database model for portfolio snapshots"""
    __tablename__ = 'portfolio_snapshots'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    cash = Column(Float)
    positions_value = Column(Float)
    total_value = Column(Float)
    daily_pnl = Column(Float)
    daily_pnl_pct = Column(Float)
    total_pnl = Column(Float)
    total_pnl_pct = Column(Float)
    drawdown = Column(Float)
    drawdown_pct = Column(Float)


class PerformanceMetrics(Base):
    """Database model for strategy performance metrics"""
    __tablename__ = 'performance_metrics'

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    strategy = Column(String(50))
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0)
    total_pnl = Column(Float, default=0)
    avg_win = Column(Float, default=0)
    avg_loss = Column(Float, default=0)
    profit_factor = Column(Float, default=0)
    sharpe_ratio = Column(Float, default=0)
    max_drawdown = Column(Float, default=0)


@dataclass
class PortfolioState:
    """Current portfolio state"""
    cash: float
    positions: dict  # asset -> Position
    total_value: float
    daily_pnl: float
    total_pnl: float
    drawdown: float
    peak_value: float


class ShadowTradingEngine:
    """
    Shadow trading engine that simulates trades without real money

    Features:
    - Virtual portfolio with $100K starting capital
    - Position tracking and P&L calculation
    - Risk management (stop loss, take profit, max drawdown)
    - Trade logging to SQLite database
    - Performance analytics
    """

    def __init__(
        self,
        db_path: str = DATABASE_PATH,
        initial_capital: float = INITIAL_CAPITAL,
        commission_rate: float = 0.001  # 0.1% commission
    ):
        self.db_path = db_path
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate

        # Initialize database
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

        # Initialize portfolio state
        self._init_portfolio()

    def _init_portfolio(self):
        """Initialize or load portfolio state"""
        with self.Session() as session:
            # Check for existing portfolio
            latest = session.query(PortfolioSnapshot).order_by(
                PortfolioSnapshot.timestamp.desc()
            ).first()

            if latest:
                self.cash = latest.cash
                self.peak_value = latest.total_value
            else:
                self.cash = self.initial_capital
                self.peak_value = self.initial_capital
                # Create initial snapshot
                self._save_snapshot(session)

    def get_position(self, asset: Asset) -> Optional[Position]:
        """Get current position for an asset"""
        with self.Session() as session:
            return session.query(Position).filter_by(asset=asset.value).first()

    def get_all_positions(self) -> List[Position]:
        """Get all open positions"""
        with self.Session() as session:
            return session.query(Position).filter(Position.quantity > 0).all()

    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state"""
        positions = {}
        positions_value = 0

        with self.Session() as session:
            for pos in session.query(Position).all():
                if pos.quantity > 0:
                    positions[pos.asset] = pos
                    positions_value += pos.quantity * pos.current_price

        total_value = self.cash + positions_value
        total_pnl = total_value - self.initial_capital
        daily_pnl = self._calculate_daily_pnl()

        # Update peak and drawdown
        if total_value > self.peak_value:
            self.peak_value = total_value

        drawdown = self.peak_value - total_value

        return PortfolioState(
            cash=self.cash,
            positions=positions,
            total_value=total_value,
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
            drawdown=drawdown,
            peak_value=self.peak_value
        )

    def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L"""
        today = datetime.now().date()
        with self.Session() as session:
            trades = session.query(Trade).filter(
                Trade.timestamp >= datetime(today.year, today.month, today.day)
            ).all()
            return sum(t.pnl for t in trades)

    def execute_signal(
        self,
        signal: Signal,
        current_price: Optional[float] = None
    ) -> Optional[Trade]:
        """
        Execute a trading signal

        Args:
            signal: Trading signal to execute
            current_price: Override price (uses signal.price if None)

        Returns:
            Trade object if executed, None if rejected
        """
        price = current_price or signal.price
        asset = signal.asset

        # Check risk limits
        if not self._check_risk_limits(signal):
            logger.warning(f"Signal rejected due to risk limits: {signal}")
            return None

        # Determine order side and quantity
        if signal.signal_type in [SignalType.BUY, SignalType.STRONG_BUY]:
            return self._execute_buy(asset, price, signal)
        elif signal.signal_type in [SignalType.SELL, SignalType.STRONG_SELL]:
            return self._execute_sell(asset, price, signal)

        return None

    def _execute_buy(
        self,
        asset: Asset,
        price: float,
        signal: Signal
    ) -> Optional[Trade]:
        """Execute a buy order"""
        # Calculate position size
        portfolio = self.get_portfolio_state()
        max_position_value = portfolio.total_value * MAX_POSITION_SIZE_PCT

        # Adjust based on signal confidence
        position_value = max_position_value * signal.confidence

        # Check available cash
        commission = position_value * self.commission_rate
        total_cost = position_value + commission

        if total_cost > self.cash:
            position_value = (self.cash - commission) / (1 + self.commission_rate)
            commission = position_value * self.commission_rate
            total_cost = position_value + commission

        if position_value <= 0:
            logger.warning(f"Insufficient cash for buy order: {asset.value}")
            return None

        quantity = position_value / price

        with self.Session() as session:
            # Update or create position
            position = session.query(Position).filter_by(asset=asset.value).first()
            if position:
                # Average in
                total_qty = position.quantity + quantity
                position.avg_price = (
                    (position.quantity * position.avg_price + quantity * price) / total_qty
                )
                position.quantity = total_qty
            else:
                position = Position(
                    asset=asset.value,
                    quantity=quantity,
                    avg_price=price,
                    current_price=price
                )
                session.add(position)

            # Set stop loss and take profit
            position.stop_loss = price * (1 - STOP_LOSS_PCT)
            position.take_profit = price * (1 + TAKE_PROFIT_PCT)
            position.updated_at = datetime.now()

            # Create trade record
            trade = Trade(
                asset=asset.value,
                side=OrderSide.BUY.value,
                quantity=quantity,
                price=price,
                value=position_value,
                commission=commission,
                strategy=signal.strategy,
                signal_confidence=signal.confidence,
                reason=signal.reason,
                status=OrderStatus.FILLED.value,
                pnl=0
            )
            session.add(trade)

            # Update cash
            self.cash -= total_cost

            session.commit()

            logger.info(
                f"BUY {quantity:.4f} {asset.value} @ ${price:.2f} "
                f"(value: ${position_value:.2f}, commission: ${commission:.2f})"
            )

            return trade

    def _execute_sell(
        self,
        asset: Asset,
        price: float,
        signal: Signal
    ) -> Optional[Trade]:
        """Execute a sell order"""
        with self.Session() as session:
            position = session.query(Position).filter_by(asset=asset.value).first()

            if not position or position.quantity <= 0:
                logger.warning(f"No position to sell: {asset.value}")
                return None

            # Sell entire position (or partial based on signal strength)
            if signal.signal_type == SignalType.STRONG_SELL:
                sell_qty = position.quantity
            else:
                sell_qty = position.quantity * 0.5  # Sell half on normal sell

            sell_value = sell_qty * price
            commission = sell_value * self.commission_rate
            net_proceeds = sell_value - commission

            # Calculate realized P&L
            cost_basis = sell_qty * position.avg_price
            realized_pnl = net_proceeds - cost_basis

            # Update position
            position.quantity -= sell_qty
            if position.quantity < 0.0001:  # Close position if nearly zero
                position.quantity = 0
                position.avg_price = 0
                position.stop_loss = None
                position.take_profit = None
            position.updated_at = datetime.now()

            # Create trade record
            trade = Trade(
                asset=asset.value,
                side=OrderSide.SELL.value,
                quantity=sell_qty,
                price=price,
                value=sell_value,
                commission=commission,
                strategy=signal.strategy,
                signal_confidence=signal.confidence,
                reason=signal.reason,
                status=OrderStatus.FILLED.value,
                pnl=realized_pnl
            )
            session.add(trade)

            # Update cash
            self.cash += net_proceeds

            session.commit()

            logger.info(
                f"SELL {sell_qty:.4f} {asset.value} @ ${price:.2f} "
                f"(value: ${sell_value:.2f}, P&L: ${realized_pnl:.2f})"
            )

            return trade

    def update_prices(self, gold_price: float, silver_price: float):
        """Update current prices and check stop loss/take profit"""
        with self.Session() as session:
            for asset, price in [(Asset.GOLD.value, gold_price), (Asset.SILVER.value, silver_price)]:
                position = session.query(Position).filter_by(asset=asset).first()
                if position and position.quantity > 0:
                    position.current_price = price
                    position.unrealized_pnl = (price - position.avg_price) * position.quantity
                    position.unrealized_pnl_pct = (price - position.avg_price) / position.avg_price * 100
                    position.updated_at = datetime.now()

                    # Check stop loss
                    if position.stop_loss and price <= position.stop_loss:
                        logger.warning(f"STOP LOSS triggered for {asset} @ ${price:.2f}")
                        self._trigger_stop_loss(session, position, price)

                    # Check take profit
                    if position.take_profit and price >= position.take_profit:
                        logger.info(f"TAKE PROFIT triggered for {asset} @ ${price:.2f}")
                        self._trigger_take_profit(session, position, price)

            session.commit()

    def _trigger_stop_loss(self, session: Session, position: Position, price: float):
        """Execute stop loss"""
        sell_signal = Signal(
            timestamp=datetime.now(),
            asset=Asset(position.asset),
            signal_type=SignalType.STRONG_SELL,
            strategy="StopLoss",
            price=price,
            confidence=1.0,
            reason=f"Stop loss triggered at ${price:.2f}"
        )
        # Close session temporarily to execute sell
        session.commit()
        self._execute_sell(Asset(position.asset), price, sell_signal)

    def _trigger_take_profit(self, session: Session, position: Position, price: float):
        """Execute take profit"""
        sell_signal = Signal(
            timestamp=datetime.now(),
            asset=Asset(position.asset),
            signal_type=SignalType.SELL,
            strategy="TakeProfit",
            price=price,
            confidence=1.0,
            reason=f"Take profit triggered at ${price:.2f}"
        )
        session.commit()
        self._execute_sell(Asset(position.asset), price, sell_signal)

    def _check_risk_limits(self, signal: Signal) -> bool:
        """Check if trade passes risk management rules"""
        portfolio = self.get_portfolio_state()

        # Check max drawdown
        drawdown_pct = portfolio.drawdown / self.peak_value if self.peak_value > 0 else 0
        if drawdown_pct > MAX_DRAWDOWN_PCT:
            logger.warning(f"Max drawdown exceeded: {drawdown_pct:.1%}")
            return False

        # Check daily loss limit
        daily_loss_pct = abs(portfolio.daily_pnl) / self.initial_capital if portfolio.daily_pnl < 0 else 0
        if daily_loss_pct > MAX_DAILY_LOSS_PCT:
            logger.warning(f"Max daily loss exceeded: {daily_loss_pct:.1%}")
            return False

        return True

    def _save_snapshot(self, session: Session):
        """Save current portfolio snapshot"""
        portfolio = self.get_portfolio_state()
        snapshot = PortfolioSnapshot(
            cash=portfolio.cash,
            positions_value=portfolio.total_value - portfolio.cash,
            total_value=portfolio.total_value,
            daily_pnl=portfolio.daily_pnl,
            daily_pnl_pct=portfolio.daily_pnl / self.initial_capital * 100,
            total_pnl=portfolio.total_pnl,
            total_pnl_pct=portfolio.total_pnl / self.initial_capital * 100,
            drawdown=portfolio.drawdown,
            drawdown_pct=portfolio.drawdown / self.peak_value * 100 if self.peak_value > 0 else 0
        )
        session.add(snapshot)
        session.commit()

    def save_snapshot(self):
        """Public method to save portfolio snapshot"""
        with self.Session() as session:
            self._save_snapshot(session)

    def get_trade_history(self, limit: int = 100) -> pd.DataFrame:
        """Get recent trade history as DataFrame"""
        with self.Session() as session:
            trades = session.query(Trade).order_by(
                Trade.timestamp.desc()
            ).limit(limit).all()

            return pd.DataFrame([
                {
                    'timestamp': t.timestamp,
                    'asset': t.asset,
                    'side': t.side,
                    'quantity': t.quantity,
                    'price': t.price,
                    'value': t.value,
                    'commission': t.commission,
                    'pnl': t.pnl,
                    'strategy': t.strategy,
                    'reason': t.reason
                }
                for t in trades
            ])

    def get_performance_summary(self) -> dict:
        """Get overall performance summary"""
        with self.Session() as session:
            trades = session.query(Trade).filter(
                Trade.status == OrderStatus.FILLED.value
            ).all()

            if not trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 0
                }

            winning = [t for t in trades if t.pnl > 0]
            losing = [t for t in trades if t.pnl < 0]

            total_wins = sum(t.pnl for t in winning)
            total_losses = abs(sum(t.pnl for t in losing))

            return {
                'total_trades': len(trades),
                'winning_trades': len(winning),
                'losing_trades': len(losing),
                'win_rate': len(winning) / len(trades) * 100 if trades else 0,
                'total_pnl': sum(t.pnl for t in trades),
                'avg_win': total_wins / len(winning) if winning else 0,
                'avg_loss': total_losses / len(losing) if losing else 0,
                'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf')
            }


if __name__ == "__main__":
    # Test the trading engine
    import os
    logging.basicConfig(level=logging.INFO)

    # Use test database
    test_db = "data/test_trading.db"
    os.makedirs("data", exist_ok=True)

    engine = ShadowTradingEngine(db_path=test_db)

    print("Initial portfolio state:")
    state = engine.get_portfolio_state()
    print(f"  Cash: ${state.cash:,.2f}")
    print(f"  Total Value: ${state.total_value:,.2f}")

    # Create test signals
    from strategies import Signal, SignalType, Asset

    buy_signal = Signal(
        timestamp=datetime.now(),
        asset=Asset.GOLD,
        signal_type=SignalType.BUY,
        strategy="Test",
        price=250.0,
        confidence=0.8,
        reason="Test buy signal"
    )

    print("\nExecuting buy signal...")
    trade = engine.execute_signal(buy_signal)
    if trade:
        print(f"  Trade executed: {trade.side} {trade.quantity:.4f} @ ${trade.price:.2f}")

    print("\nUpdated portfolio state:")
    state = engine.get_portfolio_state()
    print(f"  Cash: ${state.cash:,.2f}")
    print(f"  Positions: {len(state.positions)}")
    print(f"  Total Value: ${state.total_value:,.2f}")

    # Update prices
    engine.update_prices(255.0, 30.0)

    print("\nAfter price update:")
    state = engine.get_portfolio_state()
    print(f"  Total Value: ${state.total_value:,.2f}")
    print(f"  Total P&L: ${state.total_pnl:,.2f}")

    # Get performance
    print("\nPerformance summary:")
    perf = engine.get_performance_summary()
    for k, v in perf.items():
        print(f"  {k}: {v}")
