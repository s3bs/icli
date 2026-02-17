"""Characterisation tests for calc.py — the Lisp-style calculator.

Tests the pure math operations of the calculator. State-dependent
lookups (positionlookup, portfoliovaluelookup, stringlookup) are
tested separately since they need the app state mock.
"""

import types
import pytest
from decimal import Decimal
from unittest.mock import MagicMock
from hypothesis import given, settings, strategies as st

from icli.calc import Calculator, CalculatorTransformer


class FakeState:
    """Minimal state mock for calculator tests."""
    def __init__(self):
        self.quotesPositional = {}
        self.quoteState = {}
        self.accountStatus = {}
        self.dim = 22
        self.diy = 252


@pytest.fixture
def calculator():
    state = FakeState()
    return Calculator(state=state)


class TestPureCalculations:
    """Test calculator operations that don't need state lookups."""

    def test_simple_number(self, calculator):
        result = calculator.calc("42")
        assert result == Decimal("42")

    def test_negative_number(self, calculator):
        result = calculator.calc("-5")
        assert result == Decimal("-5")

    def test_decimal_number(self, calculator):
        result = calculator.calc("3.14")
        assert result == Decimal("3.14")

    def test_underscore_separator(self, calculator):
        result = calculator.calc("1_000_000")
        assert result == Decimal("1000000")

    def test_addition(self, calculator):
        result = calculator.calc("(+ 3 4)")
        assert result == Decimal("7")

    def test_addition_multiple(self, calculator):
        result = calculator.calc("(+ 1 2 3 4)")
        assert result == Decimal("10")

    def test_subtraction(self, calculator):
        result = calculator.calc("(- 10 3)")
        assert result == Decimal("7")

    def test_subtraction_multiple(self, calculator):
        # (- 10 3 2) = 10 - 3 - 2 = 5
        result = calculator.calc("(- 10 3 2)")
        assert result == Decimal("5")

    def test_negation(self, calculator):
        # (- 5) = -5
        result = calculator.calc("(- 5)")
        assert result == Decimal("-5")

    def test_multiplication(self, calculator):
        result = calculator.calc("(* 3 4)")
        assert result == Decimal("12")

    def test_multiplication_multiple(self, calculator):
        result = calculator.calc("(* 2 3 4)")
        assert result == Decimal("24")

    def test_division(self, calculator):
        result = calculator.calc("(/ 10 2)")
        assert result == Decimal("5")

    def test_nested_operations(self, calculator):
        # (+ (* 3 4) (- 10 5)) = 12 + 5 = 17
        result = calculator.calc("(+ (* 3 4) (- 10 5))")
        assert result == Decimal("17")

    def test_gains_positive(self, calculator):
        # (gains 3 6) => 100 (100% gain)
        result = calculator.calc("(gains 3 6)")
        assert result == Decimal("100")

    def test_gains_negative(self, calculator):
        # (gains 6 3) => -50 (50% loss)
        result = calculator.calc("(gains 6 3)")
        assert result == Decimal("-50")

    def test_grow_simple(self, calculator):
        # (grow 100 10) => 100 * 1.10 = 110
        result = calculator.calc("(grow 100 10)")
        assert abs(float(result) - 110.0) < 0.01

    def test_grow_with_duration(self, calculator):
        # (grow 100 10 2) => 100 * 1.10^2 = 121
        result = calculator.calc("(grow 100 10 2)")
        assert abs(float(result) - 121.0) < 0.01

    def test_optgains(self, calculator):
        # (o 10 2.5) => 10 * 100 * 2.5 = 2500
        result = calculator.calc("(o 10 2.5)")
        assert result == Decimal("2500")

    def test_optgains_custom_multiplier(self, calculator):
        # (o 10 2.5 50) => 10 * 50 * 2.5 = 1250
        result = calculator.calc("(o 10 2.5 50)")
        assert result == Decimal("1250")

    def test_optgainsdiff(self, calculator):
        # (og 10 1.0 2.0) => 10 * 100 * (2.0 - 1.0) = 1000
        result = calculator.calc("(og 10 1.0 2.0)")
        assert result == Decimal("1000")

    def test_round_no_decimals(self, calculator):
        # (r 3.7) => 4
        result = calculator.calc("(r 3.7)")
        assert result == 4

    def test_round_with_decimals(self, calculator):
        # (r 3.456 2) => 3.46
        result = calculator.calc("(r 3.456 2)")
        assert result == Decimal("3.46")

    def test_abs_positive(self, calculator):
        """abs() returns absolute value of a number."""
        result = calculator.calc("(abs 5)")
        assert result == Decimal("5")

    def test_abs_negative(self, calculator):
        """abs() of negative returns positive."""
        result = calculator.calc("(abs -5)")
        assert result == Decimal("5")

    def test_average_cost(self, calculator):
        # (ac 10 3.33 4 2.22) => (10*3.33 + 4*2.22) / (10+4) = 42.18/14 ≈ 3.0129
        result = calculator.calc("(ac 10 3.33 4 2.22)")
        assert abs(float(result) - 3.0129) < 0.01


class TestCalculatorTransformerDirect:
    """Test transformer methods directly for edge cases."""

    def test_add_empty_list(self):
        t = CalculatorTransformer(state=FakeState())
        assert t.add([]) == 0

    def test_mul_empty_list(self):
        t = CalculatorTransformer(state=FakeState())
        assert t.mul([]) == 1

    def test_gains_100_percent(self):
        t = CalculatorTransformer(state=FakeState())
        result = t.gains([Decimal("50"), Decimal("100")])
        assert result == Decimal("100")


class TestCalculatorEdgeCases:
    """Edge cases that would cause silent wrong results or crashes."""

    def test_division_by_zero_returns_nan(self, calculator):
        """Division by zero returns NaN (not an exception)."""
        result = calculator.calc("(/ 10 0)")
        assert result.is_nan()

    def test_deeply_nested(self, calculator):
        """Verify parser handles reasonable nesting depth."""
        result = calculator.calc("(+ (* 2 (- 10 (/ 20 4))) 1)")
        # 20/4=5, 10-5=5, 2*5=10, 10+1=11
        assert result == Decimal("11")

    def test_large_number(self, calculator):
        result = calculator.calc("1_000_000_000")
        assert result == Decimal("1000000000")

    def test_grow_zero_percent(self, calculator):
        """(grow 100 0) => 100 * 1.0 = 100"""
        result = calculator.calc("(grow 100 0)")
        assert result == Decimal("100")

    def test_gains_from_zero_raises(self, calculator):
        """(gains 0 100) — division by zero in gains."""
        with pytest.raises(Exception):
            calculator.calc("(gains 0 100)")

    def test_optgainsdiff_negative_result(self, calculator):
        """(og 10 3.0 1.0) => 10 * 100 * (1.0 - 3.0) = -2000"""
        result = calculator.calc("(og 10 3.0 1.0)")
        assert result == Decimal("-2000")

    def test_average_cost_single_lot(self, calculator):
        """(ac 100 5.50) => 5.50 (just one lot, cost is the price)"""
        result = calculator.calc("(ac 100 5.50)")
        assert result == Decimal("5.50")


def _make_quote(bid=None, ask=None, bidSize=None, askSize=None, last=None, close=None):
    """Helper to build a simple quote namespace object."""
    return types.SimpleNamespace(
        bid=bid,
        ask=ask,
        bidSize=bidSize,
        askSize=askSize,
        last=last,
        close=close,
    )


class FakeStateWithQuotes(FakeState):
    """FakeState pre-populated with quote data for stringlookup tests."""
    def __init__(self, quote_map):
        super().__init__()
        self.quoteState = quote_map


class TestStringLookup:
    """Tests for CalculatorTransformer.stringlookup()."""

    def _make_calc(self, quote_map):
        state = FakeStateWithQuotes(quote_map)
        return Calculator(state=state)

    def test_mid_price_when_bid_ask_present(self):
        """When bidSize and askSize are truthy, use mid = (bid + ask) / 2."""
        q = _make_quote(bid=10.0, ask=12.0, bidSize=100, askSize=200)
        calc = self._make_calc({"AAPL": q})
        result = calc.calc("AAPL")
        assert result == Decimal("11.0")

    def test_falls_back_to_close_when_last_is_none(self):
        """When last is None, fall back to close price."""
        q = _make_quote(bid=None, ask=None, bidSize=0, askSize=0, last=None, close=50.0)
        calc = self._make_calc({"MSFT": q})
        result = calc.calc("MSFT")
        assert result == Decimal("50.0")

    def test_uses_last_when_last_is_valid(self):
        """When last is a valid (non-None) value, use it."""
        q = _make_quote(bid=None, ask=None, bidSize=0, askSize=0, last=42.5, close=40.0)
        calc = self._make_calc({"GOOG": q})
        result = calc.calc("GOOG")
        assert result == Decimal("42.5")

    def test_missing_symbol_returns_none(self):
        """A symbol not in quoteState causes stringlookup to return None."""
        calc = self._make_calc({})
        # The transformer returns None for unknown symbols; the parser wraps it
        # in a Tree so calc() returns None from the start rule's single child.
        result = calc.calc("UNKN")
        assert result is None


class TestCalculatorProperties:
    """Hypothesis property-based tests for calculator invariants."""

    @given(
        st.decimals(
            min_value=Decimal("-1000"),
            max_value=Decimal("1000"),
            allow_nan=False,
            allow_infinity=False,
        )
    )
    @settings(max_examples=100)
    def test_abs_always_non_negative(self, x):
        """For any finite Decimal x, (abs x) >= 0."""
        state = FakeState()
        calc = Calculator(state=state)
        result = calc.calc(f"(abs {x})")
        assert result >= Decimal("0")

    @given(
        st.decimals(
            min_value=Decimal("1"),
            max_value=Decimal("1000"),
            allow_nan=False,
            allow_infinity=False,
        ),
        st.decimals(
            min_value=Decimal("1"),
            max_value=Decimal("1000"),
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=100)
    def test_div_by_nonzero_is_inverse_of_mul(self, a, b):
        """For nonzero a, b: (/ (* a b) b) ≈ a."""
        state = FakeState()
        calc = Calculator(state=state)
        product = calc.calc(f"(* {a} {b})")
        result = calc.calc(f"(/ {product} {b})")
        # Allow small floating-point tolerance
        assert abs(float(result) - float(a)) < 1e-4

    @given(
        st.decimals(
            min_value=Decimal("1"),
            max_value=Decimal("10000"),
            allow_nan=False,
            allow_infinity=False,
        ),
        st.decimals(
            min_value=Decimal("1"),
            max_value=Decimal("10000"),
            allow_nan=False,
            allow_infinity=False,
        ),
    )
    @settings(max_examples=100)
    def test_gains_symmetry(self, a, b):
        """gains(a, b) and gains(b, a) have opposite signs when a != b and both > 0."""
        from hypothesis import assume
        assume(a != b)
        state = FakeState()
        calc = Calculator(state=state)
        forward = float(calc.calc(f"(gains {a} {b})"))
        backward = float(calc.calc(f"(gains {b} {a})"))
        # One should be positive and the other negative
        assert (forward > 0) != (backward > 0)
