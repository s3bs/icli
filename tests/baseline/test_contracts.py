"""Characterisation tests for contracts.py — contract creation, parsing, and utilities.

Focus: business logic, edge cases, dispatch completeness, IBKR quirks.
Not testing: internal ib_async object internals beyond what the module promises.
"""

from __future__ import annotations

import pytest
from decimal import Decimal

ib_async = pytest.importorskip("ib_async")

from ib_async import (
    Bag,
    Bond,
    CFD,
    ComboLeg,
    Contract,
    ContFuture,
    Crypto,
    Forex,
    Future,
    FuturesOption,
    Index,
    MutualFund,
    Option,
    Order,
    Stock,
    Trade,
    Warrant,
)

import icli.engine.contracts as contracts
from icli.engine.contracts import (
    FullOrderPlacementRecord,
    TradeOrder,
    contractForName,
    contractFromSymbolDescriptor,
    contractFromTypeId,
    contractToSymbolDescriptor,
    isset,
    lookupKey,
    nameForContract,
    tickFieldsForContract,
)

# Must be set before any contractForName() futures tests run
contracts.FUT_EXP = "202503"


# ---------------------------------------------------------------------------
# isset()
# ---------------------------------------------------------------------------

class TestIsset:
    """isset() translates IBKR's sentinel FLOAT_MAX encoding to a real bool."""

    def test_none_is_not_set(self):
        assert isset(None) is False

    def test_unset_double_is_not_set(self):
        assert isset(ib_async.util.UNSET_DOUBLE) is False

    def test_rounded_unset_double_is_not_set(self):
        """The round() hack: sometimes UNSET_DOUBLE gets rounded to 2 decimals."""
        rounded = round(ib_async.util.UNSET_DOUBLE, 2)
        assert isset(rounded) is False

    def test_normal_float_is_set(self):
        assert isset(1.5) is True

    def test_zero_is_set(self):
        """Zero is a legitimate value (e.g. strike price of 0 is unusual but valid)."""
        assert isset(0.0) is True

    def test_zero_int_is_set(self):
        assert isset(0) is True

    def test_negative_is_set(self):
        assert isset(-3.14) is True

    def test_decimal_is_set(self):
        assert isset(Decimal("123.45")) is True

    def test_large_normal_float_is_set(self):
        """Make sure we don't accidentally flag large but valid floats as unset."""
        assert isset(1_000_000.0) is True


# ---------------------------------------------------------------------------
# contractForName()
# ---------------------------------------------------------------------------

class TestContractForNameSimpleStock:
    """Plain ticker symbols → Stock contracts."""

    def test_simple_stock(self):
        c = contractForName("AAPL")
        assert isinstance(c, Stock)
        assert c.symbol == "AAPL"
        assert c.exchange == "SMART"
        assert c.currency == "USD"

    def test_lowercase_uppercased(self):
        """Symbols are uppercased before parsing."""
        c = contractForName("aapl")
        assert isinstance(c, Stock)
        assert c.symbol == "AAPL"

    def test_mixed_case_uppercased(self):
        c = contractForName("mSfT")
        assert isinstance(c, Stock)
        assert c.symbol == "MSFT"

    def test_custom_exchange(self):
        c = contractForName("AAPL", exchange="NASDAQ")
        assert c.exchange == "NASDAQ"

    def test_custom_currency(self):
        c = contractForName("AAPL", currency="CAD")
        assert c.currency == "CAD"

    def test_three_letter_symbol(self):
        c = contractForName("IBM")
        assert isinstance(c, Stock)
        assert c.symbol == "IBM"

    def test_short_numeric_not_con_id(self):
        """Three-digit strings are NOT treated as conId — they become Stock."""
        c = contractForName("123")
        assert isinstance(c, Stock)
        assert c.symbol == "123"

    def test_four_digit_numeric_becomes_con_id(self):
        """Four+ digit pure-numeric strings become a bare Contract by conId."""
        c = contractForName("1234")
        assert isinstance(c, Contract)
        assert c.conId == 1234
        # It should NOT be a Stock subclass
        assert type(c) is Contract

    def test_eight_digit_numeric_becomes_con_id(self):
        c = contractForName("12345678")
        assert isinstance(c, Contract)
        assert c.conId == 12345678


class TestContractForNameNamespaced:
    """Colon-namespaced symbols (I:, C:, F:, B:, W:, S:, K:, CFD:)."""

    def test_index(self):
        c = contractForName("I:SPX")
        assert isinstance(c, Index)
        assert c.symbol == "SPX"

    def test_index_lowercase(self):
        c = contractForName("i:spx")
        assert isinstance(c, Index)
        assert c.symbol == "SPX"

    def test_crypto(self):
        c = contractForName("C:BTC")
        assert isinstance(c, Crypto)
        assert c.symbol == "BTC"
        assert c.exchange == "PAXOS"

    def test_crypto_currency_passed_through(self):
        c = contractForName("C:ETH", currency="USD")
        assert c.currency == "USD"

    def test_forex(self):
        c = contractForName("F:EURUSD")
        assert isinstance(c, Forex)
        # ib_async Forex("EURUSD") stores base currency in .symbol ("EUR")
        # and routes to exchange IDEALPRO; the pair is EURUSD
        assert c.symbol == "EUR"

    def test_forex_another_pair(self):
        c = contractForName("F:GBPUSD")
        assert isinstance(c, Forex)
        assert c.symbol == "GBP"

    def test_cfd(self):
        c = contractForName("CFD:XAUUSD")
        assert isinstance(c, CFD)
        assert c.symbol == "XAUUSD"

    def test_bond(self):
        c = contractForName("B:AAPL")
        assert isinstance(c, Bond)
        assert c.symbol == "AAPL"

    def test_warrant(self):
        c = contractForName("W:BGRY")
        assert isinstance(c, Warrant)
        assert c.symbol == "BGRY"
        assert c.right == "C"

    def test_stock_namespace(self):
        """Explicit S: namespace also creates a Stock."""
        c = contractForName("S:TSLA")
        assert isinstance(c, Stock)
        assert c.symbol == "TSLA"

    def test_contract_by_id_k_namespace(self):
        """K: namespace creates a bare Contract by conId."""
        c = contractForName("K:12345")
        assert isinstance(c, Contract)
        assert c.conId == 12345

    def test_invalid_namespace_raises(self):
        with pytest.raises(ValueError, match="Invalid contract type"):
            contractForName("Z:AAPL")

    def test_invalid_namespace_raises_for_x(self):
        with pytest.raises(ValueError):
            contractForName("X:AAPL")


class TestContractForNameFutures:
    """Slash-prefixed symbols → Future contracts."""

    def test_future_slash_prefix(self):
        c = contractForName("/ES")
        assert isinstance(c, Future)
        assert c.symbol == "ES"
        assert c.exchange == "CME"

    def test_future_comma_prefix_alias(self):
        """,ES is accepted as an alternative to /ES (for file-safe serialization)."""
        c = contractForName(",ES")
        assert isinstance(c, Future)
        assert c.symbol == "ES"

    def test_future_uses_fut_exp(self):
        """With no month code, the Future should use the module-level FUT_EXP."""
        contracts.FUT_EXP = "202503"
        c = contractForName("/ES")
        assert c.lastTradeDateOrContractMonth == "202503"

    def test_future_with_month_code(self):
        """ESZ5 → Z=December, 5=2025 → lastTrade=202512."""
        c = contractForName("/ESZ5")
        assert isinstance(c, Future)
        assert c.symbol == "ES"
        assert c.lastTradeDateOrContractMonth == "202512"

    def test_future_with_march_code(self):
        """ESH5 → H=March, 5=2025 → lastTrade=202503."""
        c = contractForName("/ESH5")
        assert isinstance(c, Future)
        assert c.lastTradeDateOrContractMonth == "202503"

    def test_future_nq(self):
        c = contractForName("/NQ")
        assert isinstance(c, Future)
        assert c.symbol == "NQ"
        assert c.exchange == "CME"

    def test_future_cl_oil(self):
        c = contractForName("/CL")
        assert isinstance(c, Future)
        assert c.symbol == "CL"

    def test_unknown_future_raises(self):
        # Use a short symbol (< 9 chars) so it hits the futures-lookup branch,
        # not the FOP parse branch which would fire first for 9-13 char symbols.
        with pytest.raises(ValueError, match="Unknown future mapping"):
            contractForName("/ZZZZ")

    def test_unknown_future_with_code_raises(self):
        # Month-coded symbol: ZZZZ5 → sym becomes "ZZZ" after stripping last 2 chars
        # (last char is digit, second-to-last must be in FUTS_MONTH_MAPPING)
        # Use /QQQZ5 where QQQ is unlikely to be a real future
        with pytest.raises(ValueError, match="Unknown future mapping"):
            contractForName("/QQQZ5")


class TestContractForNameEquityOptions:
    """OCC-format equity options (>15 chars without prefix)."""

    def test_call_option(self):
        """AAPL250321C00180000 → Option, strike=180, right='C', date='20250321'."""
        c = contractForName("AAPL250321C00180000")
        assert isinstance(c, Option)
        assert c.symbol == "AAPL"
        assert c.right == "C"
        assert c.strike == 180.0
        assert c.lastTradeDateOrContractMonth == "20250321"

    def test_put_option(self):
        """AAPL250321P00180000 → Option, right='P'."""
        c = contractForName("AAPL250321P00180000")
        assert isinstance(c, Option)
        assert c.right == "P"
        assert c.strike == 180.0

    def test_option_trading_class_matches_symbol(self):
        """tradingClass should equal the underlying symbol for standard options."""
        c = contractForName("MSFT250117C00300000")
        assert c.tradingClass == "MSFT"

    def test_option_strike_decimal_precision(self):
        """TSLA250117C00320000 → strike 320.0 (not 320000)."""
        c = contractForName("TSLA250117C00320000")
        assert c.strike == 320.0

    def test_option_large_strike(self):
        """SPX options can have 4-digit strikes: SPXW250117P04500000 → 4500.0."""
        c = contractForName("SPXW250117P04500000")
        assert isinstance(c, Option)
        assert c.strike == 4500.0
        assert c.symbol == "SPXW"

    def test_invalid_option_right_raises(self):
        """A right that is neither C nor P must raise."""
        with pytest.raises(Exception, match="Invalid option format right"):
            contractForName("AAPL250321X00180000")

    def test_option_exchange_passthrough(self):
        c = contractForName("AAPL250321C00180000", exchange="CBOE")
        assert c.exchange == "CBOE"

    def test_option_currency_passthrough(self):
        c = contractForName("AAPL250321C00180000", currency="USD")
        assert c.currency == "USD"


# ---------------------------------------------------------------------------
# nameForContract()
# ---------------------------------------------------------------------------

class TestNameForContractStock:
    def test_stock_uses_local_symbol(self):
        s = Stock(symbol="AAPL", exchange="SMART", currency="USD")
        s.localSymbol = "AAPL"
        assert nameForContract(s) == "AAPL"

    def test_stock_with_different_local_symbol(self):
        s = Stock(symbol="BRK B", exchange="SMART", currency="USD")
        s.localSymbol = "BRK B"
        assert nameForContract(s) == "BRK B"


class TestNameForContractOption:
    def test_option_strips_spaces(self):
        o = Option(symbol="AAPL")
        o.localSymbol = "AAPL  250321C00180000"
        assert nameForContract(o) == "AAPL250321C00180000"

    def test_option_no_spaces(self):
        o = Option(symbol="MSFT")
        o.localSymbol = "MSFT250117C00300000"
        assert nameForContract(o) == "MSFT250117C00300000"

    def test_option_via_secType(self):
        """secType='OPT' also triggers the option branch."""
        c = Contract()
        c.secType = "OPT"
        c.localSymbol = "IBM 250321C00120000"
        assert nameForContract(c) == "IBM250321C00120000"


class TestNameForContractFuture:
    def test_future_prefixed_with_slash(self):
        f = Future(symbol="ES", exchange="CME")
        f.localSymbol = "ESH5"
        assert nameForContract(f) == "/ESH5"

    def test_future_with_spaces_in_local_symbol(self):
        """localSymbol is used as-is (slash added, no space strip for futures)."""
        f = Future(symbol="NQ", exchange="CME")
        f.localSymbol = "NQH5"
        assert nameForContract(f) == "/NQH5"


class TestNameForContractIndex:
    def test_index_prefixed_with_i(self):
        i = Index(symbol="SPX")
        i.localSymbol = "SPX"
        assert nameForContract(i) == "I:SPX"

    def test_index_vix(self):
        i = Index(symbol="VIX")
        i.localSymbol = "VIX"
        assert nameForContract(i) == "I:VIX"


class TestNameForContractForex:
    def test_forex_prefixed_with_f(self):
        f = Forex(symbol="EURUSD")
        f.localSymbol = "EUR.USD"
        assert nameForContract(f) == "F:EUR.USD"

    def test_forex_gbp(self):
        f = Forex(symbol="GBPUSD")
        f.localSymbol = "GBP.USD"
        assert nameForContract(f) == "F:GBP.USD"


class TestNameForContractCrypto:
    def test_crypto_prefixed_with_c(self):
        c = Crypto(symbol="BTC", exchange="PAXOS")
        c.localSymbol = "BTC"
        assert nameForContract(c) == "C:BTC"


class TestNameForContractBag:
    def _make_leg(self, conId, action, ratio):
        leg = ComboLeg()
        leg.conId = conId
        leg.action = action
        leg.ratio = ratio
        return leg

    def test_bag_without_cdb(self):
        """Without a contract database, legs are described by raw conId."""
        bag = Bag(symbol="AAPL", exchange="SMART")
        leg1 = self._make_leg(conId=100, action="BUY", ratio=1)
        leg2 = self._make_leg(conId=200, action="SELL", ratio=1)
        bag.comboLegs = [leg1, leg2]
        result = nameForContract(bag)
        assert "BUY" in result
        assert "SELL" in result
        assert "100" in result
        assert "200" in result

    def test_bag_with_cdb(self):
        """With a contract db, leg names use the sub-contract's nameForContract."""
        bag = Bag(symbol="AAPL", exchange="SMART")
        leg1 = self._make_leg(conId=100, action="BUY", ratio=2)
        bag.comboLegs = [leg1]

        sub = Stock(symbol="AAPL", exchange="SMART")
        sub.localSymbol = "AAPL"
        cdb = {100: sub}

        result = nameForContract(bag, cdb=cdb)
        assert "BUY" in result
        assert "2" in result
        assert "AAPL" in result

    def test_bag_without_cdb_uses_con_id_strings(self):
        bag = Bag(symbol="SPY", exchange="SMART")
        leg = self._make_leg(conId=999, action="BUY", ratio=1)
        bag.comboLegs = [leg]
        result = nameForContract(bag)
        assert "999" in result

    def test_bag_with_cdb_missing_leg_uses_con_id(self):
        """If cdb doesn't have the conId, fall back to str(conId)."""
        bag = Bag(symbol="SPY", exchange="SMART")
        leg = self._make_leg(conId=777, action="SELL", ratio=1)
        bag.comboLegs = [leg]
        cdb = {}  # empty — leg won't be found
        result = nameForContract(bag, cdb=cdb)
        assert "777" in result


# ---------------------------------------------------------------------------
# contractFromTypeId()
# ---------------------------------------------------------------------------

class TestContractFromTypeId:
    """contractFromTypeId creates a typed contract from a class name + conId."""

    def _check(self, type_name, expected_class, con_id=42):
        c = contractFromTypeId(type_name, con_id)
        assert isinstance(c, expected_class)
        assert c.conId == con_id

    def test_stock(self):
        self._check("Stock", Stock)

    def test_future(self):
        self._check("Future", Future)

    def test_option(self):
        self._check("Option", Option)

    def test_index(self):
        self._check("Index", Index)

    def test_forex(self):
        self._check("Forex", Forex)

    def test_crypto(self):
        self._check("Crypto", Crypto)

    def test_bag(self):
        self._check("Bag", Bag)

    def test_bond(self):
        self._check("Bond", Bond)

    def test_warrant(self):
        self._check("Warrant", Warrant)

    def test_futures_option(self):
        self._check("FuturesOption", FuturesOption)

    def test_cont_future(self):
        self._check("ContFuture", ContFuture)

    def test_mutual_fund(self):
        self._check("MutualFund", MutualFund)

    def test_cfd(self):
        from ib_async import CFD
        self._check("CFD", CFD)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported contract type"):
            contractFromTypeId("WeirdThing", 1)

    def test_unknown_type_raises_arbitrary(self):
        with pytest.raises(ValueError):
            contractFromTypeId("NotAContract", 99)


# ---------------------------------------------------------------------------
# contractFromSymbolDescriptor()
# ---------------------------------------------------------------------------

class TestContractFromSymbolDescriptor:
    """contractFromSymbolDescriptor creates a typed contract from class name + symbol string."""

    def _check(self, type_name, expected_class, symbol="AAPL"):
        c = contractFromSymbolDescriptor(type_name, symbol)
        assert isinstance(c, expected_class)
        assert c.symbol == symbol

    def test_stock(self):
        self._check("Stock", Stock)

    def test_future(self):
        self._check("Future", Future, symbol="ES")

    def test_option(self):
        self._check("Option", Option)

    def test_index(self):
        self._check("Index", Index, symbol="SPX")

    def test_forex(self):
        self._check("Forex", Forex, symbol="EURUSD")

    def test_crypto(self):
        self._check("Crypto", Crypto, symbol="BTC")

    def test_bag(self):
        self._check("Bag", Bag)

    def test_bond(self):
        self._check("Bond", Bond)

    def test_warrant(self):
        self._check("Warrant", Warrant)

    def test_futures_option(self):
        self._check("FuturesOption", FuturesOption, symbol="ES")

    def test_cont_future(self):
        self._check("ContFuture", ContFuture, symbol="ES")

    def test_mutual_fund(self):
        self._check("MutualFund", MutualFund, symbol="VTSAX")

    def test_cfd(self):
        from ib_async import CFD
        self._check("CFD", CFD, symbol="XAUUSD")

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported contract type"):
            contractFromSymbolDescriptor("SomeFakeType", "AAPL")


# ---------------------------------------------------------------------------
# contractToSymbolDescriptor()
# ---------------------------------------------------------------------------

class TestContractToSymbolDescriptor:
    """contractToSymbolDescriptor returns a '-' joined string that uniquely identifies a contract."""

    def test_stock_includes_sec_type_and_symbol(self):
        s = Stock(symbol="AAPL", exchange="SMART", currency="USD")
        desc = contractToSymbolDescriptor(s)
        assert "STK" in desc
        assert "AAPL" in desc

    def test_option_includes_date_strike_right(self):
        o = Option(
            symbol="AAPL",
            lastTradeDateOrContractMonth="20250321",
            strike=180.0,
            right="C",
        )
        desc = contractToSymbolDescriptor(o)
        assert "20250321" in desc
        assert "180.0" in desc
        assert "C" in desc

    def test_option_put_right(self):
        o = Option(symbol="SPY", lastTradeDateOrContractMonth="20250117", strike=450.0, right="P")
        desc = contractToSymbolDescriptor(o)
        assert "P" in desc
        assert "450.0" in desc

    def test_no_date_uses_nodate_placeholder(self):
        """If lastTradeDateOrContractMonth is empty, 'NoDate' appears in descriptor."""
        s = Stock(symbol="AAPL")
        desc = contractToSymbolDescriptor(s)
        assert "NoDate" in desc

    def test_no_right_uses_noright_placeholder(self):
        s = Stock(symbol="AAPL")
        desc = contractToSymbolDescriptor(s)
        assert "NoRight" in desc

    def test_no_strike_uses_nostrike_placeholder(self):
        """Zero strike → '0.0' not 'NoStrike' due to logic in the function."""
        s = Stock(symbol="AAPL")
        desc = contractToSymbolDescriptor(s)
        # Strike is 0 (falsy) → the expression evaluates 'NoStrike' as the or-fallback string,
        # but float(0) is falsy so str(0.0 or "NoStrike") == "NoStrike"
        assert "NoStrike" in desc

    def test_future_sec_type(self):
        f = Future(symbol="ES", exchange="CME", lastTradeDateOrContractMonth="202503")
        desc = contractToSymbolDescriptor(f)
        assert "FUT" in desc
        assert "ES" in desc
        assert "202503" in desc

    def test_descriptor_is_string(self):
        s = Stock(symbol="IBM")
        assert isinstance(contractToSymbolDescriptor(s), str)

    def test_descriptor_separator_is_dash(self):
        s = Stock(symbol="IBM")
        parts = contractToSymbolDescriptor(s).split("-")
        # 7 fields joined by '-'
        assert len(parts) == 7


# ---------------------------------------------------------------------------
# tickFieldsForContract()
# ---------------------------------------------------------------------------

class TestTickFieldsForContract:
    """Verify the correct tick field sets are returned for each contract type."""

    BASE_FIELDS = {233, 295, 294}
    STOCK_EXTRA = {104, 106, 236, 595}

    def _field_set(self, contract):
        return set(int(f) for f in tickFieldsForContract(contract).split(","))

    def test_stock_includes_base_fields(self):
        s = Stock(symbol="AAPL")
        fields = self._field_set(s)
        assert self.BASE_FIELDS.issubset(fields)

    def test_stock_includes_extra_stock_fields(self):
        s = Stock(symbol="AAPL")
        fields = self._field_set(s)
        assert self.STOCK_EXTRA.issubset(fields)

    def test_future_does_not_include_stock_extra_fields(self):
        f = Future(symbol="ES", exchange="CME")
        fields = self._field_set(f)
        assert self.STOCK_EXTRA.isdisjoint(fields)

    def test_future_includes_base_fields(self):
        f = Future(symbol="ES", exchange="CME")
        fields = self._field_set(f)
        assert self.BASE_FIELDS.issubset(fields)

    def test_option_does_not_include_stock_extra_fields(self):
        o = Option(symbol="AAPL")
        fields = self._field_set(o)
        assert self.STOCK_EXTRA.isdisjoint(fields)

    def test_option_includes_base_fields(self):
        o = Option(symbol="AAPL")
        fields = self._field_set(o)
        assert self.BASE_FIELDS.issubset(fields)

    def test_index_does_not_include_stock_extra(self):
        i = Index(symbol="SPX")
        fields = self._field_set(i)
        assert self.STOCK_EXTRA.isdisjoint(fields)

    def test_index_includes_base_fields(self):
        i = Index(symbol="SPX")
        fields = self._field_set(i)
        assert self.BASE_FIELDS.issubset(fields)

    def test_forex_does_not_include_stock_extra(self):
        f = Forex(symbol="EURUSD")
        fields = self._field_set(f)
        assert self.STOCK_EXTRA.isdisjoint(fields)

    def test_return_type_is_string(self):
        s = Stock(symbol="AAPL")
        assert isinstance(tickFieldsForContract(s), str)

    def test_csv_format(self):
        """Fields must be comma-separated integers (IBKR API format)."""
        s = Stock(symbol="AAPL")
        raw = tickFieldsForContract(s)
        parts = raw.split(",")
        for p in parts:
            assert p.isdigit(), f"Non-integer tick field: {p!r}"


# ---------------------------------------------------------------------------
# lookupKey()
# ---------------------------------------------------------------------------

class TestLookupKey:
    """lookupKey() is @cached, and ib_async hashes contracts by conId.

    All non-Bag contracts must have a conId set so the cachetools @cached
    decorator can hash them. Bag contracts are special-cased: the cache key
    function uses 'x' (the contract itself), so Bags also need a conId.
    """

    def test_stock_with_local_symbol(self):
        s = Stock(conId=100, symbol="AAPL", exchange="SMART")
        s.localSymbol = "AAPL"
        key = lookupKey(s)
        assert key == "AAPL"

    def test_stock_local_symbol_spaces_stripped(self):
        """Spaces in localSymbol are removed from the key."""
        s = Stock(conId=101, symbol="AAPL")
        s.localSymbol = "AAPL  "
        key = lookupKey(s)
        assert " " not in key

    def test_contract_falls_back_to_symbol_when_no_local_symbol(self):
        """When localSymbol is empty, the .symbol field is the key."""
        s = Stock(conId=102, symbol="AAPL")
        # localSymbol is empty string by default
        s.localSymbol = ""
        key = lookupKey(s)
        assert key == "AAPL"

    def test_bag_returns_tuple(self):
        bag = Bag(conId=200, symbol="SPY", exchange="SMART")
        leg1 = ComboLeg()
        leg1.conId = 10
        leg1.ratio = 1
        leg1.action = "BUY"
        leg2 = ComboLeg()
        leg2.conId = 20
        leg2.ratio = 1
        leg2.action = "SELL"
        bag.comboLegs = [leg1, leg2]
        key = lookupKey(bag)
        assert isinstance(key, tuple)
        assert len(key) == 2

    def test_bag_key_is_sorted_by_ratio_action_conid(self):
        """The tuple is sorted so key is stable regardless of comboLegs order."""
        bag = Bag(conId=201, symbol="SPY", exchange="SMART")
        leg1 = ComboLeg()
        leg1.conId = 10
        leg1.ratio = 1
        leg1.action = "BUY"
        leg2 = ComboLeg()
        leg2.conId = 20
        leg2.ratio = 1
        leg2.action = "SELL"
        bag.comboLegs = [leg1, leg2]
        bag2 = Bag(conId=202, symbol="SPY", exchange="SMART")
        bag2.comboLegs = [leg2, leg1]  # reversed order
        # Both bags produce the same sorted tuple despite different leg ordering
        assert lookupKey(bag) == lookupKey(bag2)

    def test_bag_key_contains_leg_fields(self):
        bag = Bag(conId=203, symbol="SPY", exchange="SMART")
        leg = ComboLeg()
        leg.conId = 555
        leg.ratio = 2
        leg.action = "BUY"
        bag.comboLegs = [leg]
        key = lookupKey(bag)
        assert (2, "BUY", 555) in key

    def test_future_key_uses_local_symbol(self):
        f = Future(conId=300, symbol="ES", exchange="CME")
        f.localSymbol = "ESH5"
        key = lookupKey(f)
        assert key == "ESH5"


# ---------------------------------------------------------------------------
# TradeOrder / FullOrderPlacementRecord
# ---------------------------------------------------------------------------

class TestTradeOrder:
    def _make_trade_order(self):
        trade = Trade()
        order = Order()
        return TradeOrder(trade=trade, order=order)

    def test_basic_construction(self):
        to = self._make_trade_order()
        assert isinstance(to.trade, Trade)
        assert isinstance(to.order, Order)

    def test_is_frozen(self):
        """TradeOrder is a frozen dataclass — attributes must not be reassignable."""
        to = self._make_trade_order()
        with pytest.raises((AttributeError, TypeError)):
            to.trade = Trade()  # type: ignore[misc]

    def test_has_slots(self):
        """Slots dataclass should not have a __dict__ attribute."""
        to = self._make_trade_order()
        assert not hasattr(to, "__dict__")


class TestFullOrderPlacementRecord:
    def _make_trade_order(self):
        return TradeOrder(trade=Trade(), order=Order())

    def test_basic_construction_limit_only(self):
        to = self._make_trade_order()
        record = FullOrderPlacementRecord(limit=to)
        assert record.limit is to
        assert record.profit is None
        assert record.loss is None

    def test_full_construction(self):
        limit = self._make_trade_order()
        profit = self._make_trade_order()
        loss = self._make_trade_order()
        record = FullOrderPlacementRecord(limit=limit, profit=profit, loss=loss)
        assert record.profit is profit
        assert record.loss is loss

    def test_is_frozen(self):
        to = self._make_trade_order()
        record = FullOrderPlacementRecord(limit=to)
        with pytest.raises((AttributeError, TypeError)):
            record.limit = self._make_trade_order()  # type: ignore[misc]

    def test_has_slots(self):
        to = self._make_trade_order()
        record = FullOrderPlacementRecord(limit=to)
        assert not hasattr(record, "__dict__")
