"""Characterisation tests for helpers.py — pure functions only.

Tests functions that have actual logic worth preserving.
Skips: anything requiring IB connection, CLI state, or questionary prompts.
"""

import datetime
import math
import sys
import pytest
from decimal import Decimal

# helpers.py uses PEP 695 type aliases (Python 3.12+)
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="helpers.py requires Python 3.12+ (PEP 695 type syntax)"
)

# But we also need to handle the import itself failing
from icli.engine.primitives import (
    FillReport,
    ALGOMAP,
    FUTS_MONTH_MAPPING,
    convert_futures_code,
    boundsByPercentDifference,
    convert_time,
    as_duration,
    split_commands,
    sortLocalSymbol,
)
from icli.engine.technicals import TWEMA

# isset depends on ib_async — still lives in helpers.py
try:
    from icli.helpers import isset
except SyntaxError:
    pytest.skip("helpers.py requires Python 3.12+", allow_module_level=True)

import ib_async


class TestFillReportSignConvention:
    """FillReport.qty returns +qty for buys, -qty for sells.
    Getting this wrong flips your P&L sign."""

    def test_bot_returns_positive(self):
        fr = FillReport(orderId=1, conId=100, sym="AAPL", side="BOT",
                        shares=50.0, price=178.0, pnl=0.0, commission=1.0,
                        when=datetime.datetime.now())
        assert fr.qty == 50.0

    def test_sld_returns_negative(self):
        fr = FillReport(orderId=1, conId=100, sym="AAPL", side="SLD",
                        shares=50.0, price=178.0, pnl=0.0, commission=1.0,
                        when=datetime.datetime.now())
        assert fr.qty == -50.0

    def test_unknown_side_raises(self):
        """Anything other than BOT/SLD should blow up, not silently return wrong sign."""
        fr = FillReport(orderId=1, conId=100, sym="AAPL", side="UNKNOWN",
                        shares=50.0, price=178.0, pnl=0.0, commission=1.0,
                        when=datetime.datetime.now())
        with pytest.raises(AssertionError):
            _ = fr.qty


class TestFuturesCodeConversion:
    """convert_futures_code turns 'Z3' into '202312' style strings."""

    def test_z3_is_december_2023(self):
        result = convert_futures_code("Z3")
        assert result.endswith("12")  # December
        assert "202" in result        # 2020s decade

    def test_f4_is_january_2024(self):
        result = convert_futures_code("F4")
        assert result.endswith("01")

    def test_h5_is_march_2025(self):
        result = convert_futures_code("H5")
        assert result.endswith("03")

    def test_all_month_codes_valid(self):
        """Every code in FUTS_MONTH_MAPPING must produce a result."""
        for code in FUTS_MONTH_MAPPING:
            result = convert_futures_code(f"{code}5")
            assert len(result) == 6, f"Code {code}5 produced {result}"

    def test_invalid_length_raises(self):
        with pytest.raises(AssertionError):
            convert_futures_code("ZZZ")

    def test_invalid_month_code_raises(self):
        with pytest.raises((KeyError, ValueError)):
            convert_futures_code("A3")  # A is not a valid month code

    def test_lowercase_works(self):
        """Should be case-insensitive on the month letter."""
        result = convert_futures_code("z3")
        assert result.endswith("12")


class TestBoundsByPercentDifference:
    """Pure math: given a midpoint and percent, return (lower, upper) bounds."""

    def test_symmetric_around_mid(self):
        lower, upper = boundsByPercentDifference(100.0, 0.10)
        # Bounds should straddle the midpoint
        assert lower < 100.0 < upper

    def test_zero_percent_returns_mid(self):
        lower, upper = boundsByPercentDifference(100.0, 0.0)
        assert abs(lower - 100.0) < 0.001
        assert abs(upper - 100.0) < 0.001

    def test_wider_percent_gives_wider_bounds(self):
        narrow_lo, narrow_hi = boundsByPercentDifference(100.0, 0.01)
        wide_lo, wide_hi = boundsByPercentDifference(100.0, 0.10)
        assert wide_hi - wide_lo > narrow_hi - narrow_lo

    def test_known_value(self):
        """0.25% band around mid=100 → (99.7503, 100.2503).

        Formula: lower = -(mid*(p-2))/(p+2), upper = -(mid*(p+2))/(p-2)
        This is the percentage-difference midpoint formula — given a midpoint,
        it finds bounds whose percentage difference equals 2*p.
        With p=0.0025: the bounds are 0.5% apart in pct-diff terms,
        i.e. 0.25% each side of mid.

        Verified: lower=99.750312, upper=100.250313
        """
        lower, upper = boundsByPercentDifference(100.0, 0.0025)
        assert abs(lower - 99.7503) < 0.001
        assert abs(upper - 100.2503) < 0.001
        # Sanity: bounds are symmetric in percentage-difference terms
        pct_diff = abs(upper - lower) / ((upper + lower) / 2)
        assert abs(pct_diff - 0.005) < 1e-4, f"Expected 0.5% spread, got {pct_diff}"


class TestSplitCommands:
    """split_commands parses semicolon-delimited input respecting quotes."""

    def test_simple_split(self):
        assert split_commands("buy AAPL; sell SPY") == ["buy AAPL", "sell SPY"]

    def test_quoted_semicolons_preserved(self):
        result = split_commands('note "hello; world"; buy AAPL')
        assert len(result) == 2
        assert "hello; world" in result[0]

    def test_comment_stripping(self):
        result = split_commands("buy AAPL  # this is a comment")
        assert len(result) == 1
        assert "#" not in result[0]

    def test_empty_string(self):
        assert split_commands("") == []

    def test_single_command_no_semicolon(self):
        assert split_commands("buy AAPL 100") == ["buy AAPL 100"]

    def test_trailing_semicolon(self):
        result = split_commands("buy AAPL;")
        assert result == ["buy AAPL"]

    def test_escaped_characters(self):
        result = split_commands('say "hello\\nworld"')
        assert len(result) == 1


class TestConvertTime:
    """convert_time turns seconds into human-readable strings."""

    def test_seconds_only(self):
        result = convert_time(45)
        assert "45" in result
        assert "second" in result

    def test_minutes_and_seconds(self):
        result = convert_time(125)
        assert "2" in result and "minute" in result

    def test_hours(self):
        result = convert_time(3661)
        assert "1 hour" in result
        assert "1 minute" in result

    def test_zero_seconds(self):
        result = convert_time(0)
        assert "second" in result

    def test_one_week(self):
        result = convert_time(604800)
        assert "1 week" in result


class TestAsDuration:
    """as_duration is the compressed version of convert_time."""

    def test_seconds(self):
        result = as_duration(30)
        assert "s" in result

    def test_days(self):
        result = as_duration(90000)  # > 1 day
        assert "d" in result


class TestIsset:
    """isset checks if an IBKR float is a real value vs UNSET_DOUBLE sentinel."""

    def test_none_is_not_set(self):
        assert isset(None) is False

    def test_unset_double_is_not_set(self):
        assert isset(ib_async.util.UNSET_DOUBLE) is False

    def test_normal_float_is_set(self):
        assert isset(178.50) is True

    def test_zero_is_set(self):
        assert isset(0.0) is True

    def test_negative_is_set(self):
        assert isset(-5.0) is True


class TestALGOMAPCompleteness:
    """ALGOMAP maps user shortcuts like 'AF' to IBKR algo names like 'LMT + ADAPTIVE + FAST'.
    If a mapping is wrong, users get the wrong order type."""

    def test_known_aliases(self):
        assert ALGOMAP["AF"] == "LMT + ADAPTIVE + FAST"
        assert ALGOMAP["AS"] == "LMT + ADAPTIVE + SLOW"
        assert ALGOMAP["MID"] == "MIDPRICE"
        assert ALGOMAP["MKT"] == "MKT"
        assert ALGOMAP["LMT"] == "LMT"
        assert ALGOMAP["STOP"] == "STP"
        assert ALGOMAP["TSL"] == "TRAIL LIMIT"
        assert ALGOMAP["MOO"] == "MOO"
        assert ALGOMAP["MOC"] == "MOC"

    def test_no_duplicate_aliases(self):
        """Duplicate aliases would silently shadow each other."""
        # ALGOMAP is a dict so dupes are impossible at runtime,
        # but we can check that all values are valid IBKR names
        from icli.engine.orders import IOrder
        io = IOrder(action="BUY", qty=100, lmt=Decimal("150"), aux=Decimal("145"),
                    trailStopPrice=Decimal("148"), trailingPercent=Decimal("0"))
        valid_types = set()
        # Build the set of valid dispatch keys from IOrder.order()
        for ibkr_name in ALGOMAP.values():
            order = io.order(ibkr_name)
            if order is not None:
                valid_types.add(ibkr_name)
        # Every ALGOMAP value should be dispatchable
        broken = {k: v for k, v in ALGOMAP.items() if v not in valid_types}
        assert not broken, f"ALGOMAP entries with no IOrder dispatch: {broken}"


class TestTWEMA:
    """Time-Weighted EMA — the core of icli's quote display.
    This is actual math that matters for price rendering."""

    def test_first_update_sets_all_emas_to_value(self):
        tw = TWEMA()
        tw.update(100.0, timestamp=1000.0)
        for duration in tw.durations:
            assert tw.emas[duration] == 100.0

    def test_none_value_is_noop(self):
        tw = TWEMA()
        tw.update(100.0, timestamp=1000.0)
        tw.update(None, timestamp=1001.0)
        assert tw.emas[0] == 100.0  # unchanged

    def test_emas_converge_to_new_value(self):
        tw = TWEMA()
        tw.update(100.0, timestamp=1.0)  # t>0 to avoid 0-sentinel
        # Push many updates at the new price
        for i in range(1, 200):
            tw.update(110.0, timestamp=1.0 + float(i * 10))
        # Short EMAs should be very close to 110
        assert abs(tw.emas[15] - 110.0) < 0.1
        # Longer EMAs might still be catching up
        assert tw.emas[3900] > 100.0

    def test_zero_duration_is_always_last_value(self):
        tw = TWEMA()
        tw.update(100.0, timestamp=1.0)
        tw.update(200.0, timestamp=2.0)
        assert tw.emas[0] == 200.0

    def test_short_ema_reacts_faster_than_long(self):
        """After a price jump, 15s EMA converges faster than 3900s EMA.

        Bug/quirk: TWEMA uses `last_update == 0` as a sentinel for "first call",
        so timestamp=0.0 triggers re-initialisation on every call. We must use
        timestamps > 0 to get the actual EMA math path.

        With dt=1s: alpha_15 ≈ 0.0645, alpha_3900 ≈ 0.000256
        After one tick: ema_15 moves ~1.29, ema_3900 moves ~0.005
        """
        tw = TWEMA()
        tw.update(100.0, timestamp=1.0)   # init at t=1 (avoids 0-sentinel)
        tw.update(120.0, timestamp=2.0)   # first real EMA step at t=2
        short_move = abs(tw.emas[15] - 100.0)
        long_move = abs(tw.emas[3900] - 100.0)
        assert short_move > long_move, (
            f"15s EMA moved {short_move:.4f} but 3900s moved {long_move:.4f}"
        )

    def test_getitem_returns_ema(self):
        tw = TWEMA()
        tw.update(42.0, timestamp=1.0)
        assert tw[0] == 42.0
        assert tw[15] == 42.0

    def test_getitem_missing_returns_zero(self):
        tw = TWEMA()
        assert tw[99999] == 0

    def test_durations_are_sorted(self):
        tw = TWEMA(durations=(300, 15, 60, 0))
        assert tw.durations == (0, 15, 60, 300)
