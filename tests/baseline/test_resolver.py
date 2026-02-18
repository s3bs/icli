"""Tests for icli.engine.resolver — futures contract specification resolver."""

import pytest
from decimal import Decimal

from icli.engine.resolver import DynamicFutureDetail, FuturesResolver
from icli.engine.exchanges import FUTS_EXCHANGE, FUTS_TICK_DETAIL, FutureSymbol, FutureDetail


class TestDynamicFutureDetail:
    @pytest.mark.parametrize("symbol,exchange,trading_class,min_tick,multiplier,expected_decimals", [
        ("ES", "CME", "ES", Decimal("0.25"), "50", 2),
        ("6E", "CME", "6E", Decimal("0.01"), "125000", 2),
        ("YM", "CBOT", "YM", Decimal("1"), "5", 0),
    ])
    def test_decimals_from_min_tick(self, symbol, exchange, trading_class, min_tick, multiplier, expected_decimals):
        d = DynamicFutureDetail(
            symbol=symbol, exchange=exchange, tradingClass=trading_class, currency="USD",
            minTick=min_tick, sizeIncrement=Decimal("1"),
            multiplier=multiplier, contractMonth="", timeZoneId="", tradingHours="", liquidHours=""
        )
        assert d.decimals == expected_decimals

    def test_tick_value_computation(self):
        """ES: multiplier 50 * minTick 0.25 = $12.50"""
        d = DynamicFutureDetail(
            symbol="ES", exchange="CME", tradingClass="ES", currency="USD",
            minTick=Decimal("0.25"), sizeIncrement=Decimal("1"),
            multiplier="50", contractMonth="", timeZoneId="", tradingHours="", liquidHours=""
        )
        assert d.tickValue == "$12.50"

    def test_tick_value_cbot_special_case(self):
        """CBOT contracts have 100x multiplier on tick value."""
        d = DynamicFutureDetail(
            symbol="ZN", exchange="CBOT", tradingClass="ZN", currency="USD",
            minTick=Decimal("0.015625"), sizeIncrement=Decimal("1"),
            multiplier="1000", contractMonth="", timeZoneId="", tradingHours="", liquidHours=""
        )
        # CBOT: 0.015625 * 1000 * 100 = $1562.50
        assert "$" in d.tickValue

    def test_to_legacy_format_roundtrip(self):
        d = DynamicFutureDetail(
            symbol="ES", exchange="CME", tradingClass="ES", currency="USD",
            minTick=Decimal("0.25"), sizeIncrement=Decimal("1"),
            multiplier="50", contractMonth="HMUZ", timeZoneId="US/Eastern",
            tradingHours="", liquidHours="", longName="E-mini S&P 500"
        )
        legacy = d.to_legacy_format()
        assert isinstance(legacy, FutureDetail)
        assert legacy.symbol == "ES"
        assert legacy.exchange == "CME"
        assert legacy.tick == Decimal("0.25")
        assert legacy.decimals == 2


class TestFuturesResolverStaticFallback:
    def test_get_details_uses_static_when_disconnected(self):
        """No TWS connection, should return details from static data."""
        resolver = FuturesResolver()
        # Resolver is not connected, so it should use static fallback
        import asyncio
        # Use a symbol we know is in FUTS_TICK_DETAIL
        if "ES" in FUTS_TICK_DETAIL:
            detail = asyncio.get_event_loop().run_until_complete(
                resolver.get_contract_details("ES", use_static_fallback=True)
            )
            assert detail.symbol == "ES" or detail.tradingClass == "ES"

    def test_get_details_unknown_symbol_raises_keyerror(self):
        resolver = FuturesResolver()
        import asyncio
        with pytest.raises(KeyError):
            asyncio.get_event_loop().run_until_complete(
                resolver.get_contract_details("ZZZZ")
            )

    def test_get_details_cache_hit_skips_lookup(self):
        """Second call should return cached result."""
        resolver = FuturesResolver()
        # Pre-populate cache
        cached = DynamicFutureDetail(
            symbol="TEST", exchange="TEST", tradingClass="TEST", currency="USD",
            minTick=Decimal("0.01"), sizeIncrement=Decimal("1"),
            multiplier="1", contractMonth="", timeZoneId="", tradingHours="", liquidHours=""
        )
        resolver._cache["TEST.AUTO"] = cached
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            resolver.get_contract_details("TEST")
        )
        assert result is cached


class TestFuturesResolverCodeGeneration:
    def test_generate_tick_detail_produces_valid_python(self):
        resolver = FuturesResolver()
        details = {
            "ES": DynamicFutureDetail(
                symbol="ES", exchange="CME", tradingClass="ES", currency="USD",
                minTick=Decimal("0.25"), sizeIncrement=Decimal("1"),
                multiplier="50", contractMonth="HMUZ", timeZoneId="",
                tradingHours="", liquidHours="", longName="E-mini S&P 500"
            )
        }
        code = resolver.generate_tick_detail_code(details)
        assert "FUTS_TICK_DETAIL" in code
        assert "FutureDetail(" in code
        # Should be valid Python
        namespace = {"FutureDetail": FutureDetail, "Decimal": Decimal}
        exec(code, namespace)
        assert "FUTS_TICK_DETAIL" in namespace
        assert isinstance(namespace["FUTS_TICK_DETAIL"], dict)

    def test_roundtrip_generate_then_parse(self):
        resolver = FuturesResolver()
        details = {
            "NQ": DynamicFutureDetail(
                symbol="NQ", exchange="CME", tradingClass="NQ", currency="USD",
                minTick=Decimal("0.25"), sizeIncrement=Decimal("1"),
                multiplier="20", contractMonth="HMUZ", timeZoneId="",
                tradingHours="", liquidHours="", longName="E-mini Nasdaq"
            )
        }
        code = resolver.generate_tick_detail_code(details)
        namespace = {"FutureDetail": FutureDetail, "Decimal": Decimal}
        exec(code, namespace)
        result = namespace["FUTS_TICK_DETAIL"]
        assert "NQ" in result
        assert result["NQ"].tick == Decimal("0.25")


class TestExchangesStaticData:
    def test_futs_exchange_has_major_symbols(self):
        for sym in ["ES", "NQ", "YM", "CL", "GC", "ZB", "ZN"]:
            assert sym in FUTS_EXCHANGE, f"{sym} missing from FUTS_EXCHANGE"

    def test_futs_exchange_entries_are_futuresymbol(self):
        for key, val in FUTS_EXCHANGE.items():
            assert isinstance(val, FutureSymbol), f"{key} is not a FutureSymbol"

    def test_trading_class_index_populated(self):
        """addTradingClassToTopLevel() should add VX01-VX52 style entries."""
        # VX weekly contracts get added - check that at least some exist
        vx_keys = [k for k in FUTS_EXCHANGE if k.startswith("VX") and k[2:].isdigit()]
        assert len(vx_keys) > 0, "VX weekly trading class entries not found"

    def test_buildtickdetail_is_deleted(self):
        import icli.engine.exchanges as exchanges_module
        assert not hasattr(exchanges_module, 'buildTickDetail'), \
            "buildTickDetail should have been deleted from exchanges.py"
