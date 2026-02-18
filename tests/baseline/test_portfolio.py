"""Tests for icli.engine.portfolio — portfolio position queries."""

import math
import pytest
from decimal import Decimal
from unittest.mock import MagicMock

from icli.engine.portfolio import PortfolioQueries
from tests.conftest import FakeContract, FakePosition, FakeIB


class MockIDB:
    """Minimal mock for instrumentdb."""
    def decimals(self, contract):
        if contract.secType == "FUT":
            return 2
        return 2

    def exchanges(self, contract):
        return {"SMART", "NYSE", "ARCA"}

    async def round(self, contract, price, direction):
        return Decimal(str(price))


class MockApp:
    """Minimal mock for IBKRCmdlineApp providing accountId."""
    accountId = "DU1234567"


@pytest.fixture
def portfolio():
    ib = FakeIB()
    ib._connected = True
    ib.add_position("AAPL", 100, 178.50)
    ib.add_position("SPY", -50, 582.30)
    ib.add_position("MSFT", 200, 415.20)

    conIdCache = {}  # Simple dict for tests
    idb = MockIDB()

    return PortfolioQueries(ib, MockApp(), conIdCache, idb)


class TestPositionsDB:
    def test_positionsdb_uses_account_id(self, portfolio):
        class FakeWrapper:
            positions = {
                "DU1234567": {1: FakePosition(account="DU1234567")},
                "OTHER": {2: FakePosition(account="OTHER")},
            }

        portfolio.ib.wrapper = FakeWrapper()
        result = portfolio.positionsDB
        assert 1 in result
        assert 2 not in result


class TestQuantityForContract:
    def test_no_position_returns_zero(self, portfolio):
        class FakeWrapper:
            positions = {"DU1234567": {}}

        portfolio.ib.wrapper = FakeWrapper()
        c = FakeContract(symbol="AAPL", conId=99999)
        # The bug: pos.position is not returned, so always returns 0
        result = portfolio.quantityForContract(c)
        assert result == 0


class TestAveragePriceForContract:
    def test_no_position_returns_zero(self, portfolio):
        class FakeWrapper:
            positions = {"DU1234567": {}}

        portfolio.ib.wrapper = FakeWrapper()
        c = FakeContract(symbol="AAPL", conId=99999, secType="STK")
        c.comboLegs = None
        result = portfolio.averagePriceForContract(c)
        assert result == 0

    def test_existing_position_returns_avg_cost(self, portfolio):
        pos = FakePosition(account="DU1234567", avgCost=178.50)

        class FakeWrapper:
            positions = {"DU1234567": {42: pos}}

        portfolio.ib.wrapper = FakeWrapper()
        c = FakeContract(symbol="AAPL", conId=42, secType="STK")
        c.comboLegs = None
        result = portfolio.averagePriceForContract(c)
        assert result == 178.50


class TestMultiplier:
    def test_stock_multiplier_is_1(self, portfolio):
        c = FakeContract(symbol="AAPL", secType="STK")
        assert portfolio.multiplier(c) == 1

    def test_option_multiplier_default_100(self, portfolio):
        # FakeContract won't match isinstance(contract, (Option, FuturesOption))
        # so it falls to the else branch using contract.multiplier attribute.
        c = FakeContract(symbol="AAPL", secType="OPT", multiplier="100")
        assert portfolio.multiplier(c) == 100

    def test_empty_multiplier_string_returns_1(self, portfolio):
        c = FakeContract(symbol="AAPL", secType="STK", multiplier="")
        assert portfolio.multiplier(c) == 1

    @pytest.mark.parametrize("multiplier_str,expected_value,expected_type", [
        ("5", 5, int),
        ("2.5", 2.5, float),
    ])
    def test_multiplier_type_matches_value(self, portfolio, multiplier_str, expected_value, expected_type):
        c = FakeContract(symbol="AAPL", secType="STK", multiplier=multiplier_str)
        result = portfolio.multiplier(c)
        assert result == expected_value
        assert isinstance(result, expected_type)


class TestQuantityForAmount:
    def test_dollar_amount_to_shares(self, portfolio):
        c = FakeContract(symbol="AAPL", secType="STK")
        qty = portfolio.quantityForAmount(c, 10000, Decimal("200"))
        assert qty == 50  # 10000 / (200 * 1) = 50

    def test_rounds_down_to_whole_shares(self, portfolio):
        c = FakeContract(symbol="AAPL", secType="STK")
        qty = portfolio.quantityForAmount(c, 10000, Decimal("333"))
        assert qty == 30  # floor(10000 / 333) = 30

    def test_zero_amount_returns_zero(self, portfolio):
        c = FakeContract(symbol="AAPL", secType="STK")
        qty = portfolio.quantityForAmount(c, 0, Decimal("200"))
        assert qty == 0

    def test_with_multiplier_100(self, portfolio):
        c = FakeContract(symbol="AAPL", secType="STK", multiplier="100")
        # 10000 / (5 * 100) = 20
        qty = portfolio.quantityForAmount(c, 10000, Decimal("5"))
        assert qty == 20


class TestDecimals:
    def test_stock_returns_2(self, portfolio):
        c = FakeContract(symbol="AAPL", secType="STK")
        assert portfolio.decimals(c) == 2

    def test_none_decimals_returns_2(self, portfolio):
        class MockIDBNone:
            def decimals(self, contract):
                return None

        portfolio.idb = MockIDBNone()
        c = FakeContract(symbol="AAPL", secType="STK")
        assert portfolio.decimals(c) == 2

    def test_1_decimal_rounds_up_to_2(self, portfolio):
        class MockIDB1:
            def decimals(self, contract):
                return 1

        portfolio.idb = MockIDB1()
        c = FakeContract(symbol="AAPL", secType="STK")
        assert portfolio.decimals(c) == 2

    def test_4_decimals_returns_4(self, portfolio):
        class MockIDB4:
            def decimals(self, contract):
                return 4

        portfolio.idb = MockIDB4()
        c = FakeContract(symbol="AAPL", secType="STK")
        assert portfolio.decimals(c) == 4
