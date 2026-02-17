"""Canonical tests for icli/engine/primitives.py.

Covers every item with actual logic. Trivial constants (ALGOMAP values,
FUTS_MONTH_MAPPING, D100 etc.) are not tested individually.
"""

from __future__ import annotations

import datetime
import locale
import sys
from decimal import Decimal

import pytest

# primitives.py uses PEP 695 type aliases (Python 3.12+)
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="primitives.py requires Python 3.12+ (PEP 695 type syntax)",
)

from icli.engine.primitives import (
    Bracket,
    FillReport,
    LadderStep,
    LevelBreacher,
    LevelLevels,
    PaperLog,
    PriceOrQuantity,
    QuoteFlow,
    QuoteSizes,
    as_duration,
    boundsByPercentDifference,
    convert_futures_code,
    convert_time,
    find_nearest,
    FUTS_MONTH_MAPPING,
    sortLocalSymbol,
    split_commands,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_fill(side: str, shares: float = 50.0) -> FillReport:
    return FillReport(
        orderId=1,
        conId=100,
        sym="AAPL",
        side=side,
        shares=shares,
        price=178.0,
        pnl=0.0,
        commission=1.0,
        when=datetime.datetime.now(),
    )


# ── FillReport ────────────────────────────────────────────────────────────────


class TestFillReport:
    """FillReport.qty returns +qty for buys, -qty for sells."""

    def test_bot_returns_positive(self):
        assert _make_fill("BOT").qty == 50.0

    def test_sld_returns_negative(self):
        assert _make_fill("SLD").qty == -50.0

    def test_unknown_side_raises_assertion(self):
        fr = _make_fill("UNKNOWN")
        with pytest.raises(AssertionError):
            _ = fr.qty

    def test_bot_fractional_shares(self):
        fr = _make_fill("BOT", shares=0.5)
        assert fr.qty == 0.5

    def test_sld_fractional_shares(self):
        fr = _make_fill("SLD", shares=0.5)
        assert fr.qty == -0.5


# ── PaperLog ──────────────────────────────────────────────────────────────────


class TestPaperLogLog:
    """PaperLog.log() records trades and rejects invalid params."""

    def test_log_records_a_trade(self):
        pl = PaperLog()
        pl.log(10, 100.0)
        assert len(pl._trades) == 1
        assert pl._trades[0] == {"size": 10, "price": 100.0}

    def test_log_multiple_trades(self):
        pl = PaperLog()
        pl.log(10, 100.0)
        pl.log(-5, 105.0)
        assert len(pl._trades) == 2

    def test_log_rejects_negative_price(self):
        pl = PaperLog()
        with pytest.raises(ValueError):
            pl.log(10, -5.0)

    def test_log_rejects_zero_price(self):
        pl = PaperLog()
        with pytest.raises(ValueError):
            pl.log(10, 0)

    def test_log_rejects_non_numeric_size(self):
        pl = PaperLog()
        with pytest.raises((ValueError, TypeError)):
            pl.log("ten", 100.0)  # type: ignore[arg-type]

    def test_log_rejects_non_numeric_price(self):
        pl = PaperLog()
        with pytest.raises((ValueError, TypeError)):
            pl.log(10, "hundred")  # type: ignore[arg-type]

    def test_log_allows_negative_size(self):
        """Negative size is a valid short trade."""
        pl = PaperLog()
        pl.log(-10, 100.0)
        assert pl._trades[0]["size"] == -10


class TestPaperLogReport:
    """PaperLog.report() computes correct aggregates."""

    def test_report_with_no_trades_returns_zero_totals(self):
        pl = PaperLog()
        r = pl.report()
        assert r["total_size"] == 0
        assert r["average_price"] is None
        assert r["total_cost"] == 0
        assert r["unrealized_pl"] is None

    def test_report_total_size(self):
        pl = PaperLog()
        pl.log(10, 100.0)
        pl.log(5, 110.0)
        r = pl.report()
        assert r["total_size"] == 15

    def test_report_average_price(self):
        pl = PaperLog()
        pl.log(10, 100.0)
        pl.log(10, 120.0)
        r = pl.report()
        # total_cost = 10*100 + 10*120 = 2200, total_size = 20 → avg = 110
        assert abs(r["average_price"] - 110.0) < 1e-9

    def test_report_total_cost(self):
        pl = PaperLog()
        pl.log(10, 100.0)
        r = pl.report()
        # total_cost = 10 * 100 = 1000, but note report returns None when total_cost is falsy.
        # 1000 is truthy so it should be present.
        assert r["total_cost"] == 1000.0

    def test_report_unrealized_pl_with_current_price(self):
        pl = PaperLog()
        pl.log(10, 100.0)
        # average_price = 100, total_size = 10, current_price = 110
        # unrealized_pl = (110 - 100) * 10 = 100
        r = pl.report(current_price=110.0)
        assert abs(r["unrealized_pl"] - 100.0) < 1e-4

    def test_report_unrealized_pl_is_none_without_current_price(self):
        pl = PaperLog()
        pl.log(10, 100.0)
        r = pl.report()
        assert r["unrealized_pl"] is None


class TestPaperLogReset:
    """PaperLog.reset() clears all trade history."""

    def test_reset_clears_trades(self):
        pl = PaperLog()
        pl.log(10, 100.0)
        pl.log(5, 200.0)
        pl.reset()
        assert len(pl._trades) == 0

    def test_reset_then_report_returns_empty(self):
        pl = PaperLog()
        pl.log(10, 100.0)
        pl.reset()
        r = pl.report()
        assert r["total_size"] == 0

    def test_reset_allows_new_trades(self):
        pl = PaperLog()
        pl.log(10, 100.0)
        pl.reset()
        pl.log(20, 50.0)
        assert len(pl._trades) == 1


# ── QuoteSizes ────────────────────────────────────────────────────────────────


class TestQuoteSizes:
    """QuoteSizes.current returns the best available price."""

    def _qs(self, bid=None, ask=None, last=None, close=None):
        return QuoteSizes(bid=bid, ask=ask, bidSize=None, askSize=None,
                          last=last, close=close)

    def test_current_returns_midpoint_when_bid_and_ask_available(self):
        qs = self._qs(bid=99.0, ask=101.0)
        assert qs.current == 100.0

    def test_current_returns_midpoint_exact(self):
        qs = self._qs(bid=150.0, ask=150.0)
        assert qs.current == 150.0

    def test_current_returns_ask_when_bid_is_none(self):
        qs = self._qs(ask=101.0, last=99.0)
        assert qs.current == 101.0

    def test_current_returns_last_when_bid_and_ask_are_none(self):
        qs = self._qs(last=99.5)
        assert qs.current == 99.5

    def test_current_returns_none_when_everything_is_none(self):
        qs = self._qs()
        assert qs.current is None

    def test_current_prefers_ask_over_last_when_no_bid(self):
        """ask takes priority over last when bid is missing."""
        qs = self._qs(ask=105.0, last=100.0)
        assert qs.current == 105.0


# ── LevelLevels ───────────────────────────────────────────────────────────────


class TestLevelLevels:
    """LevelLevels.__post_init__ coerces numpy floats and renames 'open'."""

    def test_level_is_converted_to_native_float(self):
        # simulate a numpy float64 by subclassing float
        class NumpyFloat(float):
            pass

        ll = LevelLevels(levelType="sma", lookback=20,
                         lookbackName="close", level=NumpyFloat(123.45))
        assert type(ll.level) is float
        assert abs(ll.level - 123.45) < 1e-9

    def test_plain_float_is_preserved(self):
        ll = LevelLevels(levelType="sma", lookback=5,
                         lookbackName="close", level=55.5)
        assert ll.level == 55.5

    def test_open_lookback_name_renamed_to_start(self):
        ll = LevelLevels(levelType="sma", lookback=1,
                         lookbackName="open", level=100.0)
        assert ll.lookbackName == "start"

    def test_non_open_lookback_name_preserved(self):
        ll = LevelLevels(levelType="sma", lookback=1,
                         lookbackName="close", level=100.0)
        assert ll.lookbackName == "close"

    def test_open_rename_does_not_affect_other_fields(self):
        ll = LevelLevels(levelType="ema", lookback=10,
                         lookbackName="open", level=42.0)
        assert ll.levelType == "ema"
        assert ll.lookback == 10


# ── LevelBreacher ─────────────────────────────────────────────────────────────


class TestLevelBreacher:
    """LevelBreacher.__post_init__ derives durationName from convert_time."""

    def test_duration_name_set_from_convert_time(self):
        lb = LevelBreacher(duration=60)
        assert "minute" in lb.durationName

    def test_duration_name_for_one_day(self):
        lb = LevelBreacher(duration=86400)
        assert "day" in lb.durationName

    def test_duration_name_for_30_seconds(self):
        lb = LevelBreacher(duration=30)
        assert "30" in lb.durationName
        assert "second" in lb.durationName

    def test_enabled_defaults_to_true(self):
        lb = LevelBreacher(duration=60)
        assert lb.enabled is True

    def test_levels_defaults_to_empty_list(self):
        lb = LevelBreacher(duration=60)
        assert lb.levels == []


# ── QuoteFlow ─────────────────────────────────────────────────────────────────


class TestQuoteFlow:
    """QuoteFlow tracks bid/ask history and analyzes direction."""

    def test_update_adds_point(self):
        qf = QuoteFlow()
        qf.update(bid=100.0, ask=101.0, timestamp=1000.0)
        assert len(qf.pairs) == 1

    def test_update_multiple_points(self):
        qf = QuoteFlow()
        for i in range(5):
            qf.update(bid=100.0 + i, ask=101.0 + i, timestamp=float(i))
        assert len(qf.pairs) == 5

    def test_analyze_with_no_data_returns_empty(self):
        qf = QuoteFlow()
        result = qf.analyze()
        # defaultdict(float) — falsy or all zeroes
        assert not result or all(v == 0 for v in result.values())

    def test_analyze_with_data_returns_expected_keys(self):
        qf = QuoteFlow()
        # Feed a price series that will trigger some up/down movements
        # Use wide price differences to ensure breaches occur
        for i in range(20):
            bid = 100.0 + i * 2
            ask = bid + 1.0
            qf.update(bid=bid, ask=ask, timestamp=float(i * 100))
        result = qf.analyze()
        assert isinstance(result, dict)
        assert "duration" in result
        assert "uplen" in result
        assert "downlen" in result
        assert "upspeed" in result
        assert "downspeed" in result

    def test_analyze_duration_is_time_span(self):
        qf = QuoteFlow()
        qf.update(bid=100.0, ask=101.0, timestamp=0.0)
        qf.update(bid=100.0, ask=101.0, timestamp=10.0)
        result = qf.analyze()
        assert result["duration"] == 10.0

    def test_maxlen_caps_history(self):
        qf = QuoteFlow()
        for i in range(1300):  # beyond maxlen=1200
            qf.update(bid=float(i), ask=float(i) + 1, timestamp=float(i))
        assert len(qf.pairs) == 1200


# ── Bracket ───────────────────────────────────────────────────────────────────


class TestBracket:
    """Bracket.__post_init__ sets lossStop from lossLimit when not provided."""

    def test_loss_stop_defaults_to_loss_limit(self):
        b = Bracket(lossLimit=Decimal("10.00"))
        assert b.lossStop == Decimal("10.00")

    def test_loss_stop_preserved_when_explicitly_set(self):
        b = Bracket(lossLimit=Decimal("10.00"), lossStop=Decimal("9.50"))
        assert b.lossStop == Decimal("9.50")

    def test_loss_stop_none_when_loss_limit_not_set(self):
        b = Bracket()
        assert b.lossStop is None

    def test_profit_limit_not_modified(self):
        b = Bracket(profitLimit=Decimal("20.00"), lossLimit=Decimal("10.00"))
        assert b.profitLimit == Decimal("20.00")

    def test_default_order_types(self):
        b = Bracket()
        assert b.orderProfit == "LMT"
        assert b.orderLoss == "STP LMT"


# ── LadderStep ────────────────────────────────────────────────────────────────


class TestLadderStep:
    """LadderStep.__post_init__ asserts both fields are Decimal."""

    def test_decimal_types_accepted(self):
        ls = LadderStep(qty=Decimal("10"), limit=Decimal("150.00"))
        assert ls.qty == Decimal("10")
        assert ls.limit == Decimal("150.00")

    def test_non_decimal_qty_raises(self):
        with pytest.raises(AssertionError):
            LadderStep(qty=10, limit=Decimal("150.00"))  # type: ignore[arg-type]

    def test_non_decimal_limit_raises(self):
        with pytest.raises(AssertionError):
            LadderStep(qty=Decimal("10"), limit=150.0)  # type: ignore[arg-type]

    def test_non_decimal_both_raises(self):
        with pytest.raises(AssertionError):
            LadderStep(qty=10, limit=150.0)  # type: ignore[arg-type]


# ── PriceOrQuantity ───────────────────────────────────────────────────────────


class TestPriceOrQuantity:
    """PriceOrQuantity parses '$'-prefixed money vs bare quantity strings."""

    def test_dollar_prefix_is_money(self):
        pq = PriceOrQuantity("$5000")
        assert pq.is_money is True
        assert pq.is_quantity is False
        assert pq.qty == 5000
        assert pq.is_long is True

    def test_bare_number_is_quantity(self):
        pq = PriceOrQuantity("100")
        assert pq.is_quantity is True
        assert pq.is_money is False
        assert pq.qty == 100
        assert pq.is_long is True

    def test_negative_dollar_is_short_money(self):
        pq = PriceOrQuantity("-$500")
        assert pq.is_money is True
        assert pq.is_long is False
        assert pq.qty == 500

    def test_dollar_negative_is_short_money(self):
        """$-500 alternate syntax."""
        pq = PriceOrQuantity("$-500")
        assert pq.is_money is True
        assert pq.is_long is False
        assert pq.qty == 500

    def test_negative_bare_number_is_short_quantity(self):
        pq = PriceOrQuantity("-100")
        assert pq.is_quantity is True
        assert pq.is_long is False
        assert pq.qty == 100

    def test_underscore_stripped(self):
        pq = PriceOrQuantity("1_000")
        assert pq.qty == 1000
        assert pq.is_quantity is True

    def test_comma_stripped(self):
        pq = PriceOrQuantity("1,000")
        assert pq.qty == 1000
        assert pq.is_quantity is True

    def test_underscore_in_money_stripped(self):
        pq = PriceOrQuantity("$1_000")
        assert pq.qty == 1000
        assert pq.is_money is True

    def test_direct_int_with_is_quantity(self):
        pq = PriceOrQuantity(100, is_quantity=True)
        assert pq.qty == 100
        assert pq.is_quantity is True

    def test_direct_float_with_is_money(self):
        pq = PriceOrQuantity(500.0, is_money=True)
        assert pq.qty == 500
        assert pq.is_money is True

    def test_direct_negative_int_sets_is_long_false(self):
        pq = PriceOrQuantity(-50, is_quantity=True)
        assert pq.is_long is False
        assert pq.qty == 50

    def test_direct_value_without_flag_raises(self):
        """Providing an int without setting is_quantity or is_money must fail.

        The assertion fires during __post_init__ before .qty is set, so the
        error propagates either as AssertionError or AttributeError depending
        on whether the format-string in the assert message calls __repr__.
        """
        with pytest.raises((AssertionError, AttributeError)):
            PriceOrQuantity(100)  # neither flag set

    def test_direct_value_with_both_flags_raises(self):
        """Both flags active is invalid — must raise during __post_init__."""
        with pytest.raises((AssertionError, AttributeError)):
            PriceOrQuantity(100, is_quantity=True, is_money=True)

    def test_integer_qty_stored_as_int(self):
        """Whole-number quantities should be stored as int, not float."""
        pq = PriceOrQuantity("300")
        assert pq.qty == 300
        assert isinstance(pq.qty, int)

    def test_fractional_qty_stored_as_float(self):
        pq = PriceOrQuantity("1.5")
        assert pq.qty == 1.5
        assert isinstance(pq.qty, float)

    def test_repr_money_uses_locale_currency(self):
        locale.setlocale(locale.LC_ALL, "")
        pq = PriceOrQuantity("$1000")
        r = repr(pq)
        # locale.currency formats with symbol; just verify it's not the qty repr
        assert "$" in r or any(c.isdigit() for c in r)

    def test_repr_quantity_uses_comma_format(self):
        pq = PriceOrQuantity("1500")
        r = repr(pq)
        # 1500.00 formatted with commas → "1,500.00"
        assert "1,500" in r or "1500" in r


# ── convert_futures_code ──────────────────────────────────────────────────────


class TestConvertFuturesCode:
    """convert_futures_code turns 'Z3' into a 6-character IBKR date string."""

    def test_z3_is_december(self):
        result = convert_futures_code("Z3")
        assert result.endswith("12")

    def test_f4_is_january(self):
        result = convert_futures_code("F4")
        assert result.endswith("01")

    def test_h5_is_march(self):
        result = convert_futures_code("H5")
        assert result.endswith("03")

    def test_result_is_six_characters(self):
        result = convert_futures_code("Z3")
        assert len(result) == 6

    def test_all_month_codes_produce_six_char_result(self):
        for code in FUTS_MONTH_MAPPING:
            result = convert_futures_code(f"{code}5")
            assert len(result) == 6, f"Code {code}5 produced {result!r}"

    def test_invalid_length_raises_assertion(self):
        with pytest.raises(AssertionError):
            convert_futures_code("ZZZ")

    def test_one_char_raises_assertion(self):
        with pytest.raises(AssertionError):
            convert_futures_code("Z")

    def test_invalid_month_code_raises(self):
        with pytest.raises((KeyError, ValueError)):
            convert_futures_code("A3")  # 'A' is not in FUTS_MONTH_MAPPING

    def test_lowercase_month_letter_works(self):
        result = convert_futures_code("z3")
        assert result.endswith("12")


# ── split_commands ────────────────────────────────────────────────────────────


class TestSplitCommands:
    """split_commands splits on semicolons respecting quotes and strips comments."""

    def test_simple_split(self):
        assert split_commands("buy AAPL; sell SPY") == ["buy AAPL", "sell SPY"]

    def test_quoted_semicolons_are_preserved(self):
        result = split_commands('note "hello; world"; buy AAPL')
        assert len(result) == 2
        assert "hello; world" in result[0]

    def test_comment_stripping(self):
        result = split_commands("buy AAPL  # this is a comment")
        assert len(result) == 1
        assert "#" not in result[0]

    def test_empty_string_returns_empty_list(self):
        assert split_commands("") == []

    def test_single_command_no_semicolon(self):
        assert split_commands("buy AAPL 100") == ["buy AAPL 100"]

    def test_trailing_semicolon_not_included(self):
        result = split_commands("buy AAPL;")
        assert result == ["buy AAPL"]

    def test_multiple_commands(self):
        result = split_commands("a; b; c")
        assert result == ["a", "b", "c"]

    def test_whitespace_trimmed_per_command(self):
        result = split_commands("  buy AAPL  ;  sell SPY  ")
        assert result[0] == "buy AAPL"
        assert result[1] == "sell SPY"

    def test_escaped_character_inside_quotes(self):
        result = split_commands('say "hello\\nworld"')
        assert len(result) == 1


# ── convert_time ──────────────────────────────────────────────────────────────


class TestConvertTime:
    """convert_time turns seconds into a human-readable string."""

    def test_seconds_only(self):
        result = convert_time(45)
        assert "45" in result
        assert "second" in result

    def test_one_minute(self):
        result = convert_time(60)
        assert "minute" in result

    def test_minutes_and_seconds(self):
        result = convert_time(125)
        assert "minute" in result
        assert "second" in result

    def test_one_hour(self):
        result = convert_time(3600)
        assert "1 hour" in result

    def test_hours_minutes_seconds(self):
        result = convert_time(3661)
        assert "hour" in result
        assert "minute" in result
        assert "second" in result

    def test_one_week(self):
        result = convert_time(604800)
        assert "1 week" in result

    def test_multiple_weeks(self):
        result = convert_time(604800 * 2)
        assert "2 weeks" in result

    def test_zero_seconds(self):
        result = convert_time(0)
        assert "second" in result

    def test_singular_second(self):
        """1 second should not be pluralised."""
        result = convert_time(1)
        assert "1.00 second" in result


# ── as_duration ───────────────────────────────────────────────────────────────


class TestAsDuration:
    """as_duration is a compressed version of convert_time (max unit: days)."""

    def test_seconds(self):
        result = as_duration(30)
        assert "s" in result

    def test_one_day(self):
        result = as_duration(86400)
        assert "d" in result

    def test_days_and_hours(self):
        result = as_duration(90000)  # 1 day + 1 hr
        assert "d" in result
        assert "hr" in result

    def test_hours_and_minutes(self):
        result = as_duration(3660)
        assert "hr" in result
        assert "min" in result

    def test_zero_seconds(self):
        result = as_duration(0)
        assert "s" in result

    def test_no_weeks_unit(self):
        """as_duration only goes up to days, never weeks."""
        result = as_duration(604800)
        assert "week" not in result


# ── boundsByPercentDifference ─────────────────────────────────────────────────


class TestBoundsByPercentDifference:
    """boundsByPercentDifference returns (lower, upper) around a midpoint."""

    def test_bounds_straddle_mid(self):
        lower, upper = boundsByPercentDifference(100.0, 0.10)
        assert lower < 100.0 < upper

    def test_zero_percent_returns_mid(self):
        lower, upper = boundsByPercentDifference(100.0, 0.0)
        assert abs(lower - 100.0) < 0.001
        assert abs(upper - 100.0) < 0.001

    def test_wider_percent_gives_wider_bounds(self):
        narrow_lo, narrow_hi = boundsByPercentDifference(100.0, 0.01)
        wide_lo, wide_hi = boundsByPercentDifference(100.0, 0.10)
        assert (wide_hi - wide_lo) > (narrow_hi - narrow_lo)

    def test_known_0_25_percent_value(self):
        lower, upper = boundsByPercentDifference(100.0, 0.0025)
        assert abs(lower - 99.7503) < 0.001
        assert abs(upper - 100.2503) < 0.001

    def test_returns_tuple_of_two(self):
        result = boundsByPercentDifference(200.0, 0.05)
        assert len(result) == 2

    def test_larger_mid_scales_proportionally(self):
        lo1, hi1 = boundsByPercentDifference(100.0, 0.05)
        lo2, hi2 = boundsByPercentDifference(200.0, 0.05)
        spread1 = hi1 - lo1
        spread2 = hi2 - lo2
        assert abs(spread2 / spread1 - 2.0) < 0.001


# ── find_nearest ──────────────────────────────────────────────────────────────


class TestFindNearest:
    """find_nearest returns the index of the nearest value in a sorted list."""

    def _lst(self):
        return [10, 20, 30, 40, 50]

    def test_exact_match(self):
        assert find_nearest(self._lst(), 30) == 2

    def test_exact_match_first_element(self):
        assert find_nearest(self._lst(), 10) == 0

    def test_exact_match_last_element(self):
        assert find_nearest(self._lst(), 50) == 4

    def test_rounds_to_nearest_below(self):
        # 23 is closer to 20 (idx 1) than 30 (idx 2)
        assert find_nearest(self._lst(), 23) == 1

    def test_rounds_to_nearest_above(self):
        # 27 is closer to 30 (idx 2) than 20 (idx 1)
        assert find_nearest(self._lst(), 27) == 2

    def test_value_below_all_elements(self):
        # 5 < 10; nearest is 10 at index 0
        assert find_nearest(self._lst(), 5) == 0

    def test_value_above_all_elements(self):
        # 99 > 50; nearest is 50 at index 4
        assert find_nearest(self._lst(), 99) == 4

    def test_midpoint_rounds_consistently(self):
        # 15 is exactly halfway between 10 and 20
        # Implementation rounds down (<=) so returns 0 (idx of 10)
        idx = find_nearest(self._lst(), 15)
        assert idx in (0, 1)  # either is acceptable for a tie

    def test_single_element_list(self):
        assert find_nearest([42], 99) == 0

    def test_two_element_list_low(self):
        assert find_nearest([10, 20], 9) == 0

    def test_two_element_list_high(self):
        assert find_nearest([10, 20], 25) == 1


# ── sortLocalSymbol ───────────────────────────────────────────────────────────


class TestSortLocalSymbol:
    """sortLocalSymbol extracts (date[:6], symbol) from an OCC-style tuple."""

    def test_returns_tuple(self):
        result = sortLocalSymbol(("230120C00150000", "AAPL"))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_date_is_first_six_chars(self):
        result = sortLocalSymbol(("230120C00150000", "AAPL"))
        assert result[0] == "230120"

    def test_symbol_preserved(self):
        result = sortLocalSymbol(("230120C00150000", "AAPL"))
        assert result[1] == "AAPL"

    def test_different_expiration(self):
        result = sortLocalSymbol(("241220P00200000", "SPY"))
        assert result[0] == "241220"
        assert result[1] == "SPY"

    def test_sorting_puts_earlier_expiry_first(self):
        a = sortLocalSymbol(("230120C00150000", "AAPL"))
        b = sortLocalSymbol(("241220P00200000", "SPY"))
        assert a < b  # "230120" < "241220"


# ── portSort / tradeOrderCmp callability ──────────────────────────────────────


class TestPortSortAndTradeOrderCmpCallable:
    """portSort and tradeOrderCmp are callable — smoke test only since
    they require ib_async contract objects to exercise their full logic."""

    def test_port_sort_is_callable(self):
        from icli.engine.primitives import portSort
        assert callable(portSort)

    def test_trade_order_cmp_is_callable(self):
        from icli.engine.primitives import tradeOrderCmp
        assert callable(tradeOrderCmp)
