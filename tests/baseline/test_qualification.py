"""Tests for icli.engine.qualification — contract qualification."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from ib_async import Stock, Future, Option, Bag, ComboLeg, Contract
from icli.engine.qualification import ContractQualifier
from tests.conftest import FakeContract, FakeIB


class FakeConIdCache(dict):
    """Dict-like cache with .set() and .get() matching diskcache API."""

    def set(self, key, value, expire=None):
        self[key] = value


class FakePortfolioItem:
    """Minimal stub for ib_async PortfolioItem."""

    def __init__(self, contract, position, marketPrice=0.0):
        self.contract = contract
        self.position = position
        self.marketPrice = marketPrice


@pytest.fixture
def qualifier():
    ib = FakeIB()
    ib._connected = True
    cache = FakeConIdCache()
    quote_state = {}
    return ContractQualifier(ib, cache, quote_state)


# ── TestQualify ─────────────────────────────────────────────────────────────

class TestQualify:
    @pytest.mark.asyncio
    async def test_bag_contract_skips_ibkr_lookup(self, qualifier):
        """Bag contracts are returned as-is without calling qualifyContractsAsync."""
        bag = Bag(symbol="AAPL", comboLegs=[], currency="USD")
        bag.comboLegs = []

        # qualifyContractsAsync should never be called for bags
        qualifier.ib.qualifyContractsAsync = AsyncMock()

        result = await qualifier.qualify(bag)

        assert result == [bag]
        qualifier.ib.qualifyContractsAsync.assert_not_called()

    @pytest.mark.asyncio
    async def test_cached_contract_returned_without_network_call(self, qualifier):
        """If a contract is already in conIdCache by conId, it is returned from cache."""
        cached = Stock("AAPL", "SMART", "USD")
        cached.conId = 12345
        cached.localSymbol = "AAPL"

        qualifier.conIdCache[12345] = cached
        qualifier.ib.qualifyContractsAsync = AsyncMock()

        lookup = Contract(conId=12345)
        result = await qualifier.qualify(lookup)

        assert result == [cached]
        qualifier.ib.qualifyContractsAsync.assert_not_called()

    @pytest.mark.asyncio
    async def test_overwrite_forces_network_lookup(self, qualifier):
        """When overwrite=True, cached contracts are ignored and looked up again."""
        cached = Stock("AAPL", "SMART", "USD")
        cached.conId = 12345
        cached.localSymbol = "AAPL"
        qualifier.conIdCache[12345] = cached

        fresh = Stock("AAPL", "SMART", "USD")
        fresh.conId = 12345
        fresh.localSymbol = "AAPL"

        qualifier.ib.qualifyContractsAsync = AsyncMock(return_value=[fresh])

        lookup = Contract(conId=12345)
        result = await qualifier.qualify(lookup, overwrite=True)

        qualifier.ib.qualifyContractsAsync.assert_called_once()
        assert result[0].conId == 12345

    @pytest.mark.asyncio
    async def test_result_length_matches_input_length(self, qualifier):
        """Result list must have same length as input contracts."""
        bag1 = Bag(symbol="A", comboLegs=[], currency="USD")
        bag2 = Bag(symbol="B", comboLegs=[], currency="USD")

        result = await qualifier.qualify(bag1, bag2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_assert_raises_for_non_contract_input(self, qualifier):
        """Passing a non-Contract object should raise AssertionError."""
        with pytest.raises(AssertionError):
            await qualifier.qualify("AAPL")


# ── TestContractsForPosition ─────────────────────────────────────────────────

class TestContractsForPosition:
    def _add_portfolio_item(self, qualifier, localSymbol, position, marketPrice=100.0):
        contract = FakeContract(symbol=localSymbol, localSymbol=localSymbol)
        item = FakePortfolioItem(contract, position, marketPrice)
        qualifier.ib._portfolio_items = getattr(qualifier.ib, "_portfolio_items", []) + [item]
        qualifier.ib.portfolio = lambda: qualifier.ib._portfolio_items

    def test_returns_empty_for_no_match(self, qualifier):
        qualifier.ib.portfolio = lambda: []
        result = qualifier.contractsForPosition("ZZZZ")
        assert result == []

    def test_returns_matching_position(self, qualifier):
        contract = FakeContract(symbol="AAPL", localSymbol="AAPL")
        item = FakePortfolioItem(contract, 100.0, 179.50)
        qualifier.ib.portfolio = lambda: [item]

        result = qualifier.contractsForPosition("AAPL")
        assert len(result) == 1
        assert result[0][0] == contract
        assert result[0][1] == 100.0
        assert result[0][2] == 179.50

    def test_glob_pattern_matches_multiple(self, qualifier):
        c1 = FakeContract(symbol="AAPL", localSymbol="AAPL")
        c2 = FakeContract(symbol="AMZN", localSymbol="AMZN")
        c3 = FakeContract(symbol="MSFT", localSymbol="MSFT")
        items = [
            FakePortfolioItem(c1, 100.0),
            FakePortfolioItem(c2, 50.0),
            FakePortfolioItem(c3, 200.0),
        ]
        qualifier.ib.portfolio = lambda: items

        result = qualifier.contractsForPosition("A*")
        # AAPL and AMZN both start with A
        symbols = [r[0].localSymbol for r in result]
        assert "AAPL" in symbols
        assert "AMZN" in symbols
        assert "MSFT" not in symbols

    def test_qty_none_uses_full_position(self, qualifier):
        c = FakeContract(symbol="AAPL", localSymbol="AAPL")
        item = FakePortfolioItem(c, 150.0)
        qualifier.ib.portfolio = lambda: [item]

        result = qualifier.contractsForPosition("AAPL", qty=None)
        assert result[0][1] == 150.0

    def test_qty_larger_than_position_is_capped(self, qualifier):
        c = FakeContract(symbol="AAPL", localSymbol="AAPL")
        item = FakePortfolioItem(c, 50.0)
        qualifier.ib.portfolio = lambda: [item]

        result = qualifier.contractsForPosition("AAPL", qty=200.0)
        # qty > position, so return full position
        assert result[0][1] == 50.0

    def test_qty_smaller_than_position_uses_requested_qty(self, qualifier):
        import math
        c = FakeContract(symbol="AAPL", localSymbol="AAPL")
        item = FakePortfolioItem(c, 100.0)
        qualifier.ib.portfolio = lambda: [item]

        result = qualifier.contractsForPosition("AAPL", qty=30.0)
        # qty < position, so use qty with sign of position (positive)
        assert result[0][1] == math.copysign(30.0, 100.0)

    def test_short_position_preserves_sign(self, qualifier):
        import math
        c = FakeContract(symbol="SPY", localSymbol="SPY")
        item = FakePortfolioItem(c, -50.0)
        qualifier.ib.portfolio = lambda: [item]

        result = qualifier.contractsForPosition("SPY", qty=20.0)
        # qty < abs(position), use requested qty with sign of position (negative)
        assert result[0][1] == math.copysign(20.0, -50.0)

    def test_slash_prefix_stripped_for_futures(self, qualifier):
        c = FakeContract(symbol="NQ", localSymbol="NQ")
        item = FakePortfolioItem(c, 1.0)
        qualifier.ib.portfolio = lambda: [item]

        # Input with slash should still match
        result = qualifier.contractsForPosition("/NQ")
        assert len(result) == 1

    def test_space_stripped_from_local_symbol(self, qualifier):
        # Options have spaces in localSymbol (OCC format)
        c = FakeContract(symbol="AAPL", localSymbol="AAPL  230120C00150000")
        item = FakePortfolioItem(c, 1.0)
        qualifier.ib.portfolio = lambda: [item]

        result = qualifier.contractsForPosition("AAPL230120C00150000")
        assert len(result) == 1


# ── TestIsGuaranteedSpread ───────────────────────────────────────────────────

class TestIsGuaranteedSpread:
    @pytest.mark.asyncio
    async def test_empty_legs_returns_true(self, qualifier):
        """A bag with no combo legs has no security type conflicts — trivially guaranteed."""
        bag = Bag(symbol="TEST", comboLegs=[], currency="USD")

        # qualify will be called with zero contracts, returning []
        qualifier.qualify = AsyncMock(return_value=[])

        result = await qualifier.isGuaranteedSpread(bag)
        assert result is True

    @pytest.mark.asyncio
    async def test_single_sectype_returns_true(self, qualifier):
        """All legs having the same secType (e.g. all OPT) is guaranteed."""
        leg1 = ComboLeg(conId=1, ratio=1, action="BUY", exchange="SMART")
        leg2 = ComboLeg(conId=2, ratio=1, action="SELL", exchange="SMART")
        bag = Bag(symbol="AAPL", comboLegs=[leg1, leg2], currency="USD")

        c1 = Stock("AAPL", "SMART", "USD")
        c1.conId = 1
        c1.secType = "STK"
        c2 = Stock("AAPL", "SMART", "USD")
        c2.conId = 2
        c2.secType = "STK"

        qualifier.qualify = AsyncMock(return_value=[c1, c2])

        result = await qualifier.isGuaranteedSpread(bag)
        assert result is True

    @pytest.mark.asyncio
    async def test_stk_opt_combination_returns_true(self, qualifier):
        """STK + OPT combination is guaranteed."""
        leg1 = ComboLeg(conId=1, ratio=1, action="BUY", exchange="SMART")
        leg2 = ComboLeg(conId=2, ratio=1, action="SELL", exchange="SMART")
        bag = Bag(symbol="AAPL", comboLegs=[leg1, leg2], currency="USD")

        stk = Stock("AAPL", "SMART", "USD")
        stk.conId = 1
        stk.secType = "STK"
        opt = Option("AAPL", "20251220", 150.0, "C", "SMART")
        opt.conId = 2
        opt.secType = "OPT"

        qualifier.qualify = AsyncMock(return_value=[stk, opt])

        result = await qualifier.isGuaranteedSpread(bag)
        assert result is True

    @pytest.mark.asyncio
    async def test_fut_opt_combination_returns_false(self, qualifier):
        """FUT + OPT combination is NOT guaranteed (FUT is not in {'STK','OPT'})."""
        leg1 = ComboLeg(conId=1, ratio=1, action="BUY", exchange="CME")
        leg2 = ComboLeg(conId=2, ratio=1, action="SELL", exchange="CME")
        bag = Bag(symbol="ES", comboLegs=[leg1, leg2], currency="USD")

        fut = Future("ES", "20251219", "CME")
        fut.conId = 1
        fut.secType = "FUT"
        fop = Contract()
        fop.conId = 2
        fop.secType = "FOP"

        qualifier.qualify = AsyncMock(return_value=[fut, fop])

        result = await qualifier.isGuaranteedSpread(bag)
        assert result is False


# ── TestAddNonGuaranteeTagsIfRequired ────────────────────────────────────────

class TestAddNonGuaranteeTagsIfRequired:
    @pytest.mark.asyncio
    async def test_non_bag_does_nothing(self, qualifier):
        """Non-Bag contracts skip all logic — method should return without error."""
        stock = Stock("AAPL", "SMART", "USD")
        stock.conId = 123
        mock_order = MagicMock()

        # Should not raise or modify anything
        await qualifier.addNonGuaranteeTagsIfRequired(stock, mock_order)

        # markOrderNotGuaranteed should never be called for non-bags
        mock_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_guaranteed_bag_does_not_mark_orders(self, qualifier):
        """If the bag is guaranteed, orders should not be marked non-guaranteed."""
        bag = Bag(symbol="AAPL", comboLegs=[], currency="USD")
        mock_order = MagicMock()

        qualifier.isGuaranteedSpread = AsyncMock(return_value=True)

        await qualifier.addNonGuaranteeTagsIfRequired(bag, mock_order)

        # orders module should not be touched
        # (we verify by checking no attributes were set that markOrderNotGuaranteed would set)
        mock_order.smartComboRoutingParams.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_guaranteed_bag_marks_orders(self, qualifier):
        """If the bag is not guaranteed, orders.markOrderNotGuaranteed is called."""
        bag = Bag(symbol="ES", comboLegs=[], currency="USD")

        qualifier.isGuaranteedSpread = AsyncMock(return_value=False)

        with patch("icli.engine.qualification.orders.markOrderNotGuaranteed") as mock_mark:
            mock_order = MagicMock()
            await qualifier.addNonGuaranteeTagsIfRequired(bag, mock_order)
            mock_mark.assert_called_once_with(mock_order)

    @pytest.mark.asyncio
    async def test_none_orders_are_skipped(self, qualifier):
        """None orders in reqorders are skipped gracefully."""
        bag = Bag(symbol="ES", comboLegs=[], currency="USD")

        qualifier.isGuaranteedSpread = AsyncMock(return_value=False)

        with patch("icli.engine.qualification.orders.markOrderNotGuaranteed") as mock_mark:
            await qualifier.addNonGuaranteeTagsIfRequired(bag, None, None)
            mock_mark.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_orders_all_marked(self, qualifier):
        """All non-None orders should be marked when bag is not guaranteed."""
        bag = Bag(symbol="ES", comboLegs=[], currency="USD")

        qualifier.isGuaranteedSpread = AsyncMock(return_value=False)

        with patch("icli.engine.qualification.orders.markOrderNotGuaranteed") as mock_mark:
            o1 = MagicMock()
            o2 = MagicMock()
            await qualifier.addNonGuaranteeTagsIfRequired(bag, o1, o2)
            assert mock_mark.call_count == 2
