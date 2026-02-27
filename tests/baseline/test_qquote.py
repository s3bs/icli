"""Tests for qquote data readiness logic (icli/cmds/quotes/qquote.py).

Tests the standalone is_ticker_ready() function directly.
Does not require a live IB connection or full app construction.
"""

import math

import pytest
from ib_async import Ticker, Stock, Future, FuturesOption, Index, Forex, Option

from icli.cmds.quotes.qquote import is_ticker_ready


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_ticker(contract, **fields) -> Ticker:
    """Create a Ticker and set fields AFTER __post_init__ resets them to nan.

    ib_async's Ticker.__post_init__ overwrites all numeric fields to nan when
    created is falsy (the default). You cannot pass values via the constructor.

    For hasBidAsk() to return True, you need at minimum:
        bid (not nan, not -1), bidSize > 0,
        ask (not nan, not -1), askSize > 0
    """
    ticker = Ticker(contract=contract)
    for k, v in fields.items():
        setattr(ticker, k, v)
    return ticker


# ── NaN safety baseline ─────────────────────────────────────────────────────


class TestNaNSafety:
    """Verify that is_ticker_ready does NOT fall for the bool(nan)==True trap."""

    def test_fresh_ticker_all_nan_is_not_ready(self):
        contract = Future(symbol="ES", exchange="CME")
        ticker = Ticker(contract=contract)
        assert math.isnan(ticker.bid)
        assert math.isnan(ticker.ask)

        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is False

    def test_fresh_stock_ticker_all_nan_is_not_ready(self):
        contract = Stock("AAPL", "SMART", "USD")
        ticker = Ticker(contract=contract)

        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is False
        assert "missing" in status

    def test_nan_truthiness_trap_exists(self):
        """Document that bool(nan) is True — the bug this fix addresses."""
        assert bool(float("nan")) is True


# ── Stocks ───────────────────────────────────────────────────────────────────


class TestStockReadiness:
    """Stocks require bid/ask + impliedVolatility + histVolatility + shortable."""

    def test_fully_ready(self):
        contract = Stock("AAPL", "SMART", "USD")
        ticker = make_ticker(
            contract,
            bid=150.0, ask=150.50, bidSize=100.0, askSize=200.0,
            impliedVolatility=0.25, histVolatility=0.30, shortable=3.0,
        )
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is True
        assert status == "ready"

    def test_missing_implied_volatility(self):
        contract = Stock("AAPL", "SMART", "USD")
        ticker = make_ticker(
            contract,
            bid=150.0, ask=150.50, bidSize=100.0, askSize=200.0,
            histVolatility=0.30, shortable=3.0,
        )
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is False
        assert "impliedVolatility" in status

    def test_missing_hist_volatility(self):
        contract = Stock("MSFT", "SMART", "USD")
        ticker = make_ticker(
            contract,
            bid=400.0, ask=400.50, bidSize=50.0, askSize=50.0,
            impliedVolatility=0.20, shortable=3.0,
        )
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is False
        assert "histVolatility" in status

    def test_missing_shortable(self):
        contract = Stock("GME", "SMART", "USD")
        ticker = make_ticker(
            contract,
            bid=25.0, ask=25.10, bidSize=100.0, askSize=100.0,
            impliedVolatility=0.80, histVolatility=0.75,
        )
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is False
        assert "shortable" in status

    def test_missing_bid_ask(self):
        contract = Stock("AAPL", "SMART", "USD")
        ticker = make_ticker(
            contract,
            impliedVolatility=0.25, histVolatility=0.30, shortable=3.0,
        )
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is False
        assert "bid/ask" in status

    def test_reports_all_missing_fields(self):
        contract = Stock("AAPL", "SMART", "USD")
        ticker = make_ticker(
            contract,
            bid=150.0, ask=150.50, bidSize=100.0, askSize=200.0,
        )
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is False
        assert "impliedVolatility" in status
        assert "histVolatility" in status
        assert "shortable" in status


# ── Futures ──────────────────────────────────────────────────────────────────


class TestFuturesReadiness:
    """Futures require only a valid bid/ask pair."""

    def test_ready_with_bid_ask(self):
        contract = Future(symbol="ES", exchange="CME")
        ticker = make_ticker(
            contract, bid=6891.25, ask=6891.50, bidSize=5.0, askSize=6.0,
        )
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is True
        assert status == "ready"

    def test_not_ready_missing_ask(self):
        contract = Future(symbol="ES", exchange="CME")
        ticker = make_ticker(contract, bid=6891.25, bidSize=5.0)
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is False

    def test_not_ready_ibkr_empty_price(self):
        """IBKR sends bid=-1, bidSize=0 when no quote on that side."""
        contract = Future(symbol="ES", exchange="CME")
        ticker = make_ticker(
            contract, bid=-1, ask=6891.50, bidSize=0.0, askSize=6.0,
        )
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is False

    def test_ignores_volatility_fields(self):
        """Futures should be ready even when vol/shortable fields are nan."""
        contract = Future(symbol="ES", exchange="CME")
        ticker = make_ticker(
            contract, bid=6891.25, ask=6891.50, bidSize=5.0, askSize=6.0,
        )
        assert math.isnan(ticker.impliedVolatility)
        assert math.isnan(ticker.histVolatility)
        assert math.isnan(ticker.shortable)
        is_ready, _ = is_ticker_ready(ticker, contract)
        assert is_ready is True


# ── Options ──────────────────────────────────────────────────────────────────


class TestOptionReadiness:
    """Options require only bid/ask. Greeks arrive via modelGreeks, not direct ticks."""

    def test_equity_option_ready_with_bid_ask(self):
        contract = Option("AAPL", "20250321", 180.0, "C", "SMART")
        ticker = make_ticker(
            contract, bid=5.20, ask=5.40, bidSize=10.0, askSize=15.0,
        )
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is True
        assert status == "ready"

    def test_equity_option_not_ready(self):
        contract = Option("AAPL", "20250321", 180.0, "C", "SMART")
        ticker = Ticker(contract=contract)
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is False
        assert "bid/ask" in status

    def test_equity_option_ignores_vol_fields(self):
        """Option vol comes via modelGreeks, not direct tick fields."""
        contract = Option("SPY", "20250321", 550.0, "P", "SMART")
        ticker = make_ticker(
            contract, bid=3.50, ask=3.70, bidSize=20.0, askSize=25.0,
        )
        assert math.isnan(ticker.impliedVolatility)
        is_ready, _ = is_ticker_ready(ticker, contract)
        assert is_ready is True


class TestFuturesOptionReadiness:
    """FuturesOptions require only bid/ask, same as futures."""

    def test_fop_ready_with_bid_ask(self):
        contract = FuturesOption(
            symbol="ES", lastTradeDateOrContractMonth="20250321",
            strike=5500.0, right="C", exchange="CME",
        )
        ticker = make_ticker(
            contract, bid=45.0, ask=46.0, bidSize=3.0, askSize=4.0,
        )
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is True

    def test_fop_not_ready(self):
        contract = FuturesOption(
            symbol="ES", lastTradeDateOrContractMonth="20250321",
            strike=5500.0, right="C", exchange="CME",
        )
        ticker = Ticker(contract=contract)
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is False


# ── Indexes ──────────────────────────────────────────────────────────────────


class TestIndexReadiness:
    """Indexes prefer bid/ask, with fallback to last/close for calculation indexes."""

    def test_ready_with_bid_ask(self):
        contract = Index("SPX", "CBOE", "USD")
        ticker = make_ticker(
            contract, bid=5500.0, ask=5501.0, bidSize=1.0, askSize=1.0,
        )
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is True
        assert status == "ready"

    def test_not_ready_no_data(self):
        contract = Index("VIX", "CBOE", "USD")
        ticker = Ticker(contract=contract)
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is False

    def test_ready_with_last_only(self):
        """Calculation indexes like TICK-NYSE may have last but no bid/ask."""
        contract = Index("TICK-NYSE", "NYSE", "USD")
        ticker = make_ticker(contract, last=352.0)
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is True
        assert "last/close" in status

    def test_ready_with_close_only(self):
        contract = Index("VIX", "CBOE", "USD")
        ticker = make_ticker(contract, close=15.25)
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is True
        assert "last/close" in status

    def test_vix_family_ready_with_bid_ask(self):
        for sym in ["VIX", "VIN", "VIF"]:
            contract = Index(sym, "CBOE", "USD")
            ticker = make_ticker(
                contract, bid=15.0, ask=15.10, bidSize=1.0, askSize=1.0,
            )
            is_ready, status = is_ticker_ready(ticker, contract)
            assert is_ready is True, f"{sym} should be ready with bid/ask"


# ── Forex ────────────────────────────────────────────────────────────────────


class TestForexReadiness:
    def test_ready_with_bid_ask(self):
        contract = Forex("EURUSD")
        ticker = make_ticker(
            contract, bid=1.0850, ask=1.0851, bidSize=1000000.0, askSize=1000000.0,
        )
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is True

    def test_not_ready(self):
        contract = Forex("GBPUSD")
        ticker = Ticker(contract=contract)
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is False


# ── Progressive data arrival ─────────────────────────────────────────────────


class TestProgressiveDataArrival:
    """Simulate tickers starting empty and fields populating over time."""

    def test_future_becomes_ready_after_bid_ask(self):
        contract = Future(symbol="ES", exchange="CME")
        ticker = Ticker(contract=contract)

        assert is_ticker_ready(ticker, contract)[0] is False

        ticker.bid = 6891.25
        ticker.ask = 6891.50
        ticker.bidSize = 5.0
        ticker.askSize = 6.0

        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is True

    def test_stock_partial_then_full(self):
        contract = Stock("AAPL", "SMART", "USD")
        ticker = Ticker(contract=contract)

        # Phase 1: bid/ask arrive
        ticker.bid = 150.0
        ticker.ask = 150.50
        ticker.bidSize = 100.0
        ticker.askSize = 200.0
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is False
        assert "impliedVolatility" in status

        # Phase 2: volatility arrives
        ticker.impliedVolatility = 0.25
        ticker.histVolatility = 0.30
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is False
        assert "shortable" in status

        # Phase 3: shortable arrives — now ready
        ticker.shortable = 3.0
        is_ready, status = is_ticker_ready(ticker, contract)
        assert is_ready is True

    def test_mixed_contract_types(self):
        """qquote with both a stock and a future — future resolves first."""
        stock_c = Stock("AAPL", "SMART", "USD")
        future_c = Future(symbol="ES", exchange="CME")
        stock_t = Ticker(contract=stock_c)
        future_t = Ticker(contract=future_c)

        tickers = [stock_t, future_t]
        contracts = [stock_c, future_c]

        # Both start not ready
        assert not all(is_ticker_ready(t, c)[0] for t, c in zip(tickers, contracts))

        # Future gets data first
        future_t.bid = 6891.25
        future_t.ask = 6891.50
        future_t.bidSize = 5.0
        future_t.askSize = 6.0

        statuses = [is_ticker_ready(t, c) for t, c in zip(tickers, contracts)]
        assert statuses[1][0] is True   # future ready
        assert statuses[0][0] is False  # stock still waiting

        # Stock gets all data
        stock_t.bid = 150.0
        stock_t.ask = 150.50
        stock_t.bidSize = 100.0
        stock_t.askSize = 200.0
        stock_t.impliedVolatility = 0.25
        stock_t.histVolatility = 0.30
        stock_t.shortable = 3.0

        assert all(is_ticker_ready(t, c)[0] for t, c in zip(tickers, contracts))
