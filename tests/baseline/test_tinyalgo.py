"""Characterisation tests for tinyalgo.py — ATR calculations.

These tests lock down the existing ATR behaviour before any refactoring.
The ATR class uses a rolling moving average (RMA) of True Range.
"""

import pytest
from icli.engine.technicals import ATR, ATRLive


class TestATR:
    """Test the bar-based ATR calculator."""

    def test_initial_atr_is_zero(self):
        atr = ATR(length=20)
        assert atr.current == 0
        assert atr.updated == 0

    def test_first_update_uses_high_minus_low(self):
        atr = ATR(length=14)
        result = atr.update(high=105.0, low=100.0, close=103.0)
        assert result == 5.0
        assert atr.current == 5.0
        assert atr.updated == 1

    def test_first_update_equal_high_low_gives_zero(self):
        atr = ATR(length=14)
        result = atr.update(high=100.0, low=100.0, close=100.0)
        assert result == 0.0

    def test_second_update_uses_rma(self):
        atr = ATR(length=14)
        atr.update(high=105.0, low=100.0, close=103.0)
        # Second update: true range components:
        # high-low = 4, abs(high-prevClose) = abs(107-103) = 4, abs(low-prevClose) = abs(103-103) = 0
        # max(4, 4, 0) = 4
        # useUpdateLen = min(2, 14) = 2
        # new_atr = (4 + (2-1)*5) / 2 = 9/2 = 4.5
        result = atr.update(high=107.0, low=103.0, close=105.0)
        assert result == 4.5

    def test_atr_converges_with_constant_range(self):
        """With constant high-low range, ATR should converge to that range."""
        atr = ATR(length=10)
        for i in range(100):
            base = 100.0
            result = atr.update(high=base + 2.0, low=base, close=base + 1.0)
        # After many updates with constant range of 2, ATR should be ~2
        assert abs(result - 2.0) < 0.01

    def test_atr_increases_with_gap(self):
        """Gap up/down should increase ATR due to true range calculation."""
        atr = ATR(length=5)
        atr.update(high=100.0, low=98.0, close=99.0)
        atr.update(high=101.0, low=99.0, close=100.0)
        atr_before = atr.current
        # Gap up: low is above previous close
        atr.update(high=110.0, low=105.0, close=108.0)
        assert atr.current > atr_before

    def test_length_1_valid(self):
        atr = ATR(length=1)
        atr.update(high=105.0, low=100.0, close=103.0)
        result = atr.update(high=107.0, low=103.0, close=105.0)
        # With length=1, useUpdateLen=min(2,1)=1, so new_atr = currentTR/1 = currentTR
        # currentTR = max(4, 4, 0) = 4
        assert result == 4.0

    def test_length_0_raises(self):
        with pytest.raises(AssertionError):
            ATR(length=0)

    def test_negative_length_raises(self):
        with pytest.raises(AssertionError):
            ATR(length=-5)

    def test_prevclose_tracks(self):
        atr = ATR(length=14)
        atr.update(high=105.0, low=100.0, close=103.0)
        assert atr.prevClose == 103.0
        atr.update(high=107.0, low=103.0, close=106.0)
        assert atr.prevClose == 106.0


class TestATRLive:
    """Test the live-trade ATR that accumulates prices into a buffer."""

    def test_initial_state(self):
        atr = ATRLive(length=20, bufferLength=55)
        assert atr.current == 0

    def test_single_price_gives_zero_range(self):
        atr = ATRLive(length=20)
        result = atr.update(100.0)
        # buffer=[100], high=100, low=100 => range=0
        assert result == 0.0

    def test_two_prices_gives_range(self):
        atr = ATRLive(length=20)
        atr.update(100.0)
        result = atr.update(102.0)
        # buffer=[100, 102], high=102, low=100 => range=2
        assert result > 0

    def test_buffer_rolls_over(self):
        atr = ATRLive(length=5, bufferLength=3)
        atr.update(100.0)
        atr.update(110.0)
        atr.update(105.0)
        # Buffer full: [100, 110, 105], range = 10
        atr.update(106.0)
        # Buffer rolled: [110, 105, 106], range = 5
        # The ATR should adapt to smaller range
        assert atr.current > 0

    def test_current_property_passthrough(self):
        atr = ATRLive(length=14)
        atr.update(100.0)
        atr.update(105.0)
        assert atr.current == atr.atr.current

    def test_monotone_prices_converge(self):
        """Steadily rising prices should produce a stable ATR."""
        atr = ATRLive(length=10, bufferLength=20)
        for i in range(50):
            atr.update(100.0 + i * 0.1)
        # With small steady increments, ATR should be small
        assert atr.current < 3.0
