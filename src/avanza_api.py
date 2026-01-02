"""
Avanza API Integration Module

This module provides integration with Avanza's unofficial API for:
- Reading portfolio data
- Fetching real-time quotes
- Placing orders (when enabled)

To use this module, you need to set up your Avanza credentials:
1. Get your TOTP secret from Avanza's 2FA setup
2. Set environment variables or update config

Environment variables:
- AVANZA_USERNAME: Your Avanza username
- AVANZA_PASSWORD: Your Avanza password
- AVANZA_TOTP_SECRET: Your TOTP secret for 2FA
- AVANZA_ACCOUNT_ID: Your ISK account ID (found in Avanza web/app)
- AVANZA_ENABLED: Set to "true" to enable API integration
"""
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import os

from config import AvanzaSettings

logger = logging.getLogger(__name__)


class AvanzaOrderType(Enum):
    """Order types supported by Avanza"""
    BUY = "BUY"
    SELL = "SELL"


class AvanzaOrderStatus(Enum):
    """Order statuses from Avanza"""
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class AvanzaPosition:
    """Position data from Avanza"""
    instrument_id: str
    name: str
    ticker: str
    quantity: float
    avg_price: float
    current_price: float
    value: float
    profit_loss: float
    profit_loss_percent: float
    account_id: str


@dataclass
class AvanzaQuote:
    """Real-time quote from Avanza"""
    instrument_id: str
    ticker: str
    name: str
    last_price: float
    bid: float
    ask: float
    high: float
    low: float
    volume: int
    change: float
    change_percent: float
    updated: datetime


@dataclass
class AvanzaOrder:
    """Order data from Avanza"""
    order_id: str
    account_id: str
    instrument_id: str
    order_type: AvanzaOrderType
    quantity: float
    price: float
    status: AvanzaOrderStatus
    created: datetime
    message: Optional[str] = None


class AvanzaClient:
    """
    Client for interacting with Avanza API

    Usage:
        # Initialize with credentials
        client = AvanzaClient.from_env()

        # Or with explicit credentials
        client = AvanzaClient(
            username="your_username",
            password="your_password",
            totp_secret="your_totp_secret"
        )

        # Connect to Avanza
        if client.connect():
            # Get portfolio
            positions = client.get_positions()

            # Get quote
            quote = client.get_quote("GLD")

            # Place order (if enabled)
            order = client.place_order(
                instrument_id="12345",
                order_type=AvanzaOrderType.BUY,
                quantity=10,
                price=250.00
            )
    """

    # Known instrument IDs for ETFs on Avanza
    # These may need to be updated - check Avanza for current IDs
    INSTRUMENT_IDS = {
        "GLD": "34427",  # SPDR Gold Shares
        "SIVR": "186588",  # abrdn Physical Silver Shares
        # Add more as needed
    }

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        totp_secret: Optional[str] = None,
        account_id: Optional[str] = None,
        enabled: bool = False
    ):
        self.username = username
        self.password = password
        self.totp_secret = totp_secret
        self.account_id = account_id
        self.enabled = enabled
        self._client = None
        self._connected = False

    @classmethod
    def from_env(cls) -> 'AvanzaClient':
        """Create client from environment variables"""
        settings = AvanzaSettings.from_env()
        return cls(
            username=settings.username,
            password=settings.password,
            totp_secret=settings.totp_secret,
            account_id=settings.account_id,
            enabled=settings.enabled
        )

    @classmethod
    def from_settings(cls, settings: AvanzaSettings) -> 'AvanzaClient':
        """Create client from settings object"""
        return cls(
            username=settings.username,
            password=settings.password,
            totp_secret=settings.totp_secret,
            account_id=settings.account_id,
            enabled=settings.enabled
        )

    def is_configured(self) -> bool:
        """Check if credentials are configured"""
        return all([
            self.username,
            self.password,
            self.totp_secret,
            self.enabled
        ])

    def connect(self) -> bool:
        """
        Connect to Avanza API

        Returns:
            True if connected successfully, False otherwise
        """
        if not self.is_configured():
            logger.warning("Avanza API not configured. Set credentials to enable.")
            return False

        try:
            from avanza import Avanza

            self._client = Avanza({
                'username': self.username,
                'password': self.password,
                'totpSecret': self.totp_secret
            })
            self._connected = True
            logger.info("Connected to Avanza API")
            return True

        except ImportError:
            logger.error("avanza-api package not installed. Run: pip install avanza-api")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Avanza: {e}")
            return False

    def disconnect(self):
        """Disconnect from Avanza API"""
        self._client = None
        self._connected = False
        logger.info("Disconnected from Avanza API")

    def get_positions(self, account_id: Optional[str] = None) -> List[AvanzaPosition]:
        """
        Get current positions from Avanza

        Args:
            account_id: Specific account ID, or use default

        Returns:
            List of positions
        """
        if not self._connected:
            logger.warning("Not connected to Avanza")
            return []

        account_id = account_id or self.account_id

        try:
            overview = self._client.get_overview()
            positions = []

            for account in overview.get('accounts', []):
                if account_id and account.get('accountId') != account_id:
                    continue

                for pos in account.get('positions', []):
                    positions.append(AvanzaPosition(
                        instrument_id=pos.get('instrumentId', ''),
                        name=pos.get('name', ''),
                        ticker=pos.get('tickerSymbol', ''),
                        quantity=pos.get('volume', 0),
                        avg_price=pos.get('averageAcquiredPrice', 0),
                        current_price=pos.get('lastPrice', 0),
                        value=pos.get('value', 0),
                        profit_loss=pos.get('profit', 0),
                        profit_loss_percent=pos.get('profitPercent', 0),
                        account_id=account.get('accountId', '')
                    ))

            return positions

        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    def get_quote(self, ticker_or_id: str) -> Optional[AvanzaQuote]:
        """
        Get real-time quote for an instrument

        Args:
            ticker_or_id: Ticker symbol or instrument ID

        Returns:
            Quote data or None
        """
        if not self._connected:
            logger.warning("Not connected to Avanza")
            return None

        # Convert ticker to ID if needed
        instrument_id = self.INSTRUMENT_IDS.get(ticker_or_id, ticker_or_id)

        try:
            data = self._client.get_stock_info(instrument_id)

            return AvanzaQuote(
                instrument_id=instrument_id,
                ticker=data.get('tickerSymbol', ''),
                name=data.get('name', ''),
                last_price=data.get('lastPrice', 0),
                bid=data.get('buyPrice', 0),
                ask=data.get('sellPrice', 0),
                high=data.get('highestPrice', 0),
                low=data.get('lowestPrice', 0),
                volume=data.get('totalVolumeTraded', 0),
                change=data.get('change', 0),
                change_percent=data.get('changePercent', 0),
                updated=datetime.now()
            )

        except Exception as e:
            logger.error(f"Failed to get quote for {ticker_or_id}: {e}")
            return None

    def get_account_balance(self, account_id: Optional[str] = None) -> Dict[str, float]:
        """
        Get account balance and buying power

        Returns:
            Dict with 'total_value', 'buying_power', 'cash'
        """
        if not self._connected:
            return {}

        account_id = account_id or self.account_id

        try:
            overview = self._client.get_overview()

            for account in overview.get('accounts', []):
                if account.get('accountId') == account_id:
                    return {
                        'total_value': account.get('totalValue', 0),
                        'buying_power': account.get('buyingPower', 0),
                        'cash': account.get('totalBalance', 0)
                    }

            return {}

        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return {}

    def place_order(
        self,
        instrument_id: str,
        order_type: AvanzaOrderType,
        quantity: float,
        price: Optional[float] = None,
        account_id: Optional[str] = None,
        valid_until: Optional[str] = None
    ) -> Optional[AvanzaOrder]:
        """
        Place an order on Avanza

        WARNING: This executes real trades! Use with caution.

        Args:
            instrument_id: Instrument ID to trade
            order_type: BUY or SELL
            quantity: Number of shares
            price: Limit price (None for market order)
            account_id: Account to trade in
            valid_until: Order validity (e.g., "TODAY", "GOOD_TILL_CANCELLED")

        Returns:
            Order object or None if failed
        """
        if not self._connected:
            logger.error("Not connected to Avanza")
            return None

        account_id = account_id or self.account_id
        if not account_id:
            logger.error("No account ID specified")
            return None

        # Safety check - require explicit confirmation for real trading
        if not os.getenv("AVANZA_ALLOW_TRADING", "").lower() == "true":
            logger.warning(
                "Real trading disabled. Set AVANZA_ALLOW_TRADING=true to enable. "
                "Order NOT placed."
            )
            return AvanzaOrder(
                order_id="SIMULATED",
                account_id=account_id,
                instrument_id=instrument_id,
                order_type=order_type,
                quantity=quantity,
                price=price or 0,
                status=AvanzaOrderStatus.PENDING,
                created=datetime.now(),
                message="Simulated - real trading disabled"
            )

        try:
            order_params = {
                'accountId': account_id,
                'orderbookId': instrument_id,
                'orderType': order_type.value,
                'volume': int(quantity),
                'validUntil': valid_until or 'TODAY'
            }

            if price:
                order_params['price'] = price

            result = self._client.place_order(**order_params)

            return AvanzaOrder(
                order_id=result.get('orderId', ''),
                account_id=account_id,
                instrument_id=instrument_id,
                order_type=order_type,
                quantity=quantity,
                price=price or 0,
                status=AvanzaOrderStatus.PENDING,
                created=datetime.now(),
                message=result.get('message', '')
            )

        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return None

    def get_orders(self, account_id: Optional[str] = None) -> List[AvanzaOrder]:
        """Get active orders"""
        if not self._connected:
            return []

        try:
            orders_data = self._client.get_deals_and_orders()
            orders = []

            for order in orders_data.get('orders', []):
                if account_id and order.get('accountId') != account_id:
                    continue

                orders.append(AvanzaOrder(
                    order_id=order.get('orderId', ''),
                    account_id=order.get('accountId', ''),
                    instrument_id=order.get('orderbookId', ''),
                    order_type=AvanzaOrderType(order.get('orderType', 'BUY')),
                    quantity=order.get('volume', 0),
                    price=order.get('price', 0),
                    status=AvanzaOrderStatus.PENDING,
                    created=datetime.now(),
                    message=order.get('statusText', '')
                ))

            return orders

        except Exception as e:
            logger.error(f"Failed to get orders: {e}")
            return []

    def cancel_order(self, order_id: str, account_id: Optional[str] = None) -> bool:
        """Cancel an active order"""
        if not self._connected:
            return False

        account_id = account_id or self.account_id

        try:
            self._client.delete_order(account_id, order_id)
            logger.info(f"Cancelled order {order_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False


def get_avanza_client() -> AvanzaClient:
    """
    Get Avanza client instance

    Tries to create client from environment variables.
    Returns unconfigured client if credentials not set.
    """
    return AvanzaClient.from_env()


# Convenience functions for dashboard integration
def sync_avanza_positions(client: AvanzaClient, engine) -> bool:
    """
    Sync Avanza positions with shadow trading engine

    This allows comparing real portfolio with shadow portfolio.
    """
    if not client.is_configured():
        return False

    if not client.connect():
        return False

    positions = client.get_positions()

    for pos in positions:
        logger.info(
            f"Avanza Position: {pos.name} - "
            f"{pos.quantity} shares @ ${pos.current_price:.2f} "
            f"(P&L: ${pos.profit_loss:.2f})"
        )

    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Avanza API Integration Module")
    print("=" * 40)

    # Check configuration
    client = get_avanza_client()

    if client.is_configured():
        print("Avanza credentials configured!")

        if client.connect():
            print("Connected to Avanza")

            # Get positions
            positions = client.get_positions()
            print(f"\nPositions ({len(positions)}):")
            for pos in positions:
                print(f"  {pos.ticker}: {pos.quantity} @ ${pos.current_price:.2f}")

            # Get balance
            balance = client.get_account_balance()
            print(f"\nAccount Balance:")
            print(f"  Total Value: ${balance.get('total_value', 0):,.2f}")
            print(f"  Buying Power: ${balance.get('buying_power', 0):,.2f}")

            client.disconnect()
    else:
        print("Avanza not configured. Set environment variables:")
        print("  - AVANZA_USERNAME")
        print("  - AVANZA_PASSWORD")
        print("  - AVANZA_TOTP_SECRET")
        print("  - AVANZA_ACCOUNT_ID")
        print("  - AVANZA_ENABLED=true")
        print("\nTo enable real trading (USE WITH CAUTION):")
        print("  - AVANZA_ALLOW_TRADING=true")
