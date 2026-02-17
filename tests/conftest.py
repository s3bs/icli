"""Shared test fixtures for icli test suite.

FakeIB provides a test double for ib_async's IB class, allowing
headless testing without a live TWS/Gateway connection.
"""

import datetime
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock

import pytest


# ── Lightweight stubs for ib_async types used in tests ──
# We define minimal stubs here so tests don't require ib_async installed.
# For tests that DO need real ib_async types, use @pytest.mark.slow.

@dataclass
class FakeContract:
    """Minimal stub for ib_async Contract."""
    symbol: str = ""
    secType: str = "STK"
    exchange: str = "SMART"
    currency: str = "USD"
    conId: int = 0
    localSymbol: str = ""
    tradingClass: str = ""
    multiplier: str = ""
    lastTradeDateOrContractMonth: str = ""
    right: str = ""
    strike: float = 0.0


@dataclass
class FakePosition:
    """Stub for ib_async Position."""
    account: str = "DU1234567"
    contract: FakeContract = field(default_factory=FakeContract)
    position: float = 0.0
    avgCost: float = 0.0


@dataclass
class FakeTicker:
    """Stub for ib_async Ticker."""
    contract: FakeContract = field(default_factory=FakeContract)
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    close: float | None = None
    high: float | None = None
    low: float | None = None
    open: float | None = None
    volume: float = 0
    bidSize: float = 0
    askSize: float = 0
    lastSize: float = 0


@dataclass
class FakeAccountValue:
    """Stub for ib_async AccountValue."""
    account: str = "DU1234567"
    tag: str = ""
    value: str = "0"
    currency: str = "USD"
    modelCode: str = ""


@dataclass
class FakeOrder:
    """Stub for ib_async Order - minimal fields for testing."""
    orderId: int = 0
    action: str = "BUY"
    totalQuantity: float = 0
    orderType: str = "LMT"
    lmtPrice: float = 0.0
    auxPrice: float = 0.0
    tif: str = "GTC"
    outsideRth: bool = True
    whatIf: bool = False
    goodTillDate: str = ""
    goodAfterTime: str = ""
    algoStrategy: str = ""
    algoParams: list = field(default_factory=list)
    smartComboRoutingParams: list = field(default_factory=list)
    cashQty: float = 0
    trailStopPrice: float = 0.0
    trailingPercent: float = 0.0
    lmtPriceOffset: float = 0.0
    triggerPrice: float = 0.0
    notHeld: bool = False
    postToAts: int = 0
    sweepToFill: bool = False


@dataclass
class FakeTrade:
    """Stub for ib_async Trade."""
    contract: FakeContract = field(default_factory=FakeContract)
    order: FakeOrder = field(default_factory=FakeOrder)
    orderStatus: Any = None
    fills: list = field(default_factory=list)
    log: list = field(default_factory=list)


class FakeIB:
    """Test double for ib_async.IB().
    
    Provides canned data for positions, orders, account values,
    and ticker subscriptions without requiring a live TWS connection.
    """

    def __init__(self):
        self._positions: list[FakePosition] = []
        self._orders: list[FakeTrade] = []
        self._account_values: list[FakeAccountValue] = []
        self._tickers: dict[str, FakeTicker] = {}
        self._connected: bool = False
        self._client_id: int = 1
        
        # Event callbacks (matching ib_async's event system)
        self.connectedEvent = MagicMock()
        self.disconnectedEvent = MagicMock()
        self.pendingTickersEvent = MagicMock()
        self.orderStatusEvent = MagicMock()
        self.newOrderEvent = MagicMock()
        self.errorEvent = MagicMock()

    # ── Connection ──
    
    async def connectAsync(self, host="127.0.0.1", port=7497, clientId=1, **kwargs):
        self._connected = True
        self._client_id = clientId
        return self
    
    def isConnected(self) -> bool:
        return self._connected

    def disconnect(self):
        self._connected = False

    # ── Positions ──

    def positions(self) -> list[FakePosition]:
        return list(self._positions)

    def add_position(self, symbol: str, quantity: float, avg_cost: float, 
                     sec_type: str = "STK", exchange: str = "SMART"):
        """Test helper: add a canned position."""
        contract = FakeContract(symbol=symbol, secType=sec_type, exchange=exchange, localSymbol=symbol)
        self._positions.append(FakePosition(
            contract=contract, position=quantity, avgCost=avg_cost
        ))

    # ── Account Values ──

    def accountValues(self) -> list[FakeAccountValue]:
        return list(self._account_values)
    
    def add_account_value(self, tag: str, value: str, currency: str = "USD"):
        """Test helper: add a canned account value."""
        self._account_values.append(FakeAccountValue(tag=tag, value=value, currency=currency))

    # ── Orders ──
    
    def openTrades(self) -> list[FakeTrade]:
        return list(self._orders)

    def trades(self) -> list[FakeTrade]:
        return list(self._orders)

    async def placeOrder(self, contract, order) -> FakeTrade:
        trade = FakeTrade(contract=contract, order=order)
        self._orders.append(trade)
        return trade

    async def cancelOrder(self, order) -> None:
        pass

    # ── Market Data ──

    def reqMktData(self, contract, genericTickList="", snapshot=False, 
                   regulatorySnapshot=False, mktDataOptions=None) -> FakeTicker:
        symbol = contract.symbol or contract.localSymbol
        if symbol not in self._tickers:
            self._tickers[symbol] = FakeTicker(contract=contract)
        return self._tickers[symbol]

    def cancelMktData(self, contract):
        pass

    def add_ticker(self, symbol: str, bid: float, ask: float, last: float,
                   volume: float = 1000, close: float | None = None):
        """Test helper: add a canned ticker."""
        contract = FakeContract(symbol=symbol, localSymbol=symbol)
        self._tickers[symbol] = FakeTicker(
            contract=contract, bid=bid, ask=ask, last=last, 
            close=close or last, volume=volume,
            bidSize=100, askSize=100, lastSize=50
        )

    # ── Misc ──

    def reqCurrentTime(self) -> datetime.datetime:
        return datetime.datetime.now()

    def managedAccounts(self) -> list[str]:
        return ["DU1234567"]


# ── Fixtures ──

@pytest.fixture
def fake_ib() -> FakeIB:
    """Pre-populated FakeIB with sample data."""
    ib = FakeIB()
    ib._connected = True
    
    # Sample positions
    ib.add_position("AAPL", 100, 178.50)
    ib.add_position("SPY", -50, 582.30)
    ib.add_position("MSFT", 200, 415.20)
    
    # Sample tickers
    ib.add_ticker("AAPL", bid=179.50, ask=179.52, last=179.51)
    ib.add_ticker("SPY", bid=583.10, ask=583.12, last=583.11)
    ib.add_ticker("MSFT", bid=416.00, ask=416.05, last=416.02)
    
    # Sample account values
    ib.add_account_value("NetLiquidation", "250000.00")
    ib.add_account_value("BuyingPower4", "1000000.00")
    ib.add_account_value("TotalCashValue", "125000.00")
    ib.add_account_value("AvailableFunds", "200000.00")
    ib.add_account_value("UnrealizedPnL", "3500.00")
    ib.add_account_value("RealizedPnL", "1200.00")
    ib.add_account_value("DailyPnL", "450.00")
    ib.add_account_value("OptionMarketValue", "15000.00")
    ib.add_account_value("GrossPositionValue", "175000.00")
    ib.add_account_value("MaintMarginReq", "50000.00")
    ib.add_account_value("ExcessLiquidity", "200000.00")
    ib.add_account_value("SMA", "220000.00")
    ib.add_account_value("EquityWithLoanValue", "250000.00")
    
    return ib


@pytest.fixture
def empty_ib() -> FakeIB:
    """Empty FakeIB for testing from scratch."""
    ib = FakeIB()
    ib._connected = True
    return ib
