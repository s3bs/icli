"""Tests for icli.engine.quotemanager — quote management."""
import pytest
from unittest.mock import MagicMock
from icli.engine.quotemanager import QuoteManager


def make_quotemanager(quotes=None):
    """Create a QuoteManager with mock/fake dependencies.

    Parameters
    ----------
    quotes:
        Optional list of (symkey, iticker) tuples to pre-populate
        quotesPositional.  Defaults to an empty list.
    """
    if quotes is None:
        quotes = []

    ib = MagicMock()
    quoteState = {}
    quotesPositional = list(quotes)
    contractIdsToQuoteKeysMappings = {}
    conIdCache = {}
    idb = MagicMock()
    ol = MagicMock()

    return QuoteManager(
        ib=ib,
        quoteState=quoteState,
        quotesPositional=quotesPositional,
        contractIdsToQuoteKeysMappings=contractIdsToQuoteKeysMappings,
        conIdCache=conIdCache,
        idb=idb,
        ol=ol,
        qualifier=MagicMock(),
        portfolio=MagicMock(),
        clock=MagicMock(),
        iticker_state=MagicMock(),
    )


class TestQuoteResolve:
    @pytest.mark.parametrize("lookup", [":", ":abc", ":99"])
    def test_invalid_lookup_returns_none_pair(self, lookup):
        qm = make_quotemanager()
        result = qm.quoteResolve(lookup)
        assert result == (None, None)

    def test_valid_index_returns_name_and_contract(self):
        from tests.conftest import FakeContract

        fake_contract = FakeContract(symbol="AAPL", localSymbol="AAPL", conId=1)
        fake_ticker = MagicMock()
        fake_ticker.contract = fake_contract

        qm = make_quotemanager(quotes=[("AAPL", fake_ticker)])
        name, contract = qm.quoteResolve(":0")
        assert name == "AAPL"
        assert contract is fake_contract

    def test_valid_index_strips_spaces_from_local_symbol(self):
        from tests.conftest import FakeContract

        fake_contract = FakeContract(symbol="AAPL", localSymbol="AAPL  240119C00150000", conId=2)
        fake_ticker = MagicMock()
        fake_ticker.contract = fake_contract

        qm = make_quotemanager(quotes=[("AAPL240119C00150000", fake_ticker)])
        name, contract = qm.quoteResolve(":0")
        # spaces are removed
        assert " " not in name


class TestQuoteExists:
    def test_missing_returns_false(self):
        # Use a real ib_async Contract so lookupKey (which uses a cachetools cache
        # requiring hashability) can process it correctly.
        from ib_async import Contract
        qm = make_quotemanager()
        c = Contract(symbol="ZZZZ", secType="STK", conId=999)
        assert not qm.quoteExists(c)

    def test_present_returns_true(self):
        from ib_async import Contract
        from icli.engine.contracts import lookupKey
        c = Contract(symbol="AAPL", secType="STK", conId=1)
        qm = make_quotemanager()
        key = lookupKey(c)
        qm.quoteState[key] = MagicMock()
        assert qm.quoteExists(c)


class TestScanStringReplacePositionsWithSymbols:
    def test_no_colons_unchanged(self):
        qm = make_quotemanager()
        assert qm.scanStringReplacePositionsWithSymbols("info AAPL") == "info AAPL"

    def test_colon_number_replaced_with_symbol(self):
        from tests.conftest import FakeContract

        fake_contract = FakeContract(symbol="AAPL", localSymbol="AAPL", conId=1)
        fake_ticker = MagicMock()
        fake_ticker.contract = fake_contract

        qm = make_quotemanager(quotes=[("AAPL", fake_ticker)])
        result = qm.scanStringReplacePositionsWithSymbols("info :0 today")
        assert result == "info AAPL today"

    def test_missing_position_replaced_with_not_found(self):
        qm = make_quotemanager(quotes=[])
        result = qm.scanStringReplacePositionsWithSymbols("info :99 today")
        assert result == "info NOT_FOUND today"

    def test_multiple_positions_replaced(self):
        from tests.conftest import FakeContract

        c0 = FakeContract(symbol="AAPL", localSymbol="AAPL", conId=1)
        t0 = MagicMock()
        t0.contract = c0

        c1 = FakeContract(symbol="SPY", localSymbol="SPY", conId=2)
        t1 = MagicMock()
        t1.contract = c1

        qm = make_quotemanager(quotes=[("AAPL", t0), ("SPY", t1)])
        result = qm.scanStringReplacePositionsWithSymbols(":0 and :1")
        assert result == "AAPL and SPY"
