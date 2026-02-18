"""Tests for icli.engine.toolbar — bottom toolbar renderer.

Focus: the module-level formatting helpers and ToolbarRenderer construction /
render() with no real market data (exercises the except path).
"""
from __future__ import annotations

import math
import pytest
from unittest.mock import MagicMock, patch

from icli.engine.toolbar import ToolbarRenderer, fmtPrice2, fmtEquitySpread, fmtPriceOpt


# ---------------------------------------------------------------------------
# fmtPrice2
# ---------------------------------------------------------------------------


class TestFmtPrice2:
    @pytest.mark.parametrize("input_val", [0, 0.0])
    def test_fmtPrice2_falsy_inputs(self, input_val):
        result = fmtPrice2(input_val)
        assert "0.00" in result

    def test_small_positive_value(self):
        result = fmtPrice2(1234.56)
        assert "1,234.56" in result

    @pytest.mark.parametrize("input_val,has_decimal", [
        (1_000_000.00, False),
        (999_999.99, True),
    ])
    def test_fmtPrice2_million_boundary(self, input_val, has_decimal):
        result = fmtPrice2(input_val)
        if has_decimal:
            assert ".99" in result
        else:
            assert "." not in result.strip() or result.strip().endswith(",000")

    def test_negative_value(self):
        result = fmtPrice2(-500.25)
        assert "-" in result
        assert "500.25" in result

    @pytest.mark.parametrize("input_val", [1.00, 5_000_000])
    def test_fmtPrice2_alignment(self, input_val):
        result = fmtPrice2(input_val)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# fmtEquitySpread
# ---------------------------------------------------------------------------


class TestFmtEquitySpread:
    def test_small_float_uses_decimal_format(self):
        result = fmtEquitySpread(0.05)
        assert "0.05" in result

    def test_large_value_uses_comma_no_decimal(self):
        result = fmtEquitySpread(1500.0)
        # >= 1000 → comma format, no decimals
        assert "." not in result.strip()
        assert "1,500" in result

    def test_non_numeric_uses_right_pad(self):
        result = fmtEquitySpread("N/A")
        assert "N/A" in result
        assert len(result) == 5

    def test_integer_input_small(self):
        result = fmtEquitySpread(50, digits=2)
        assert "50.00" in result

    def test_custom_digits(self):
        result = fmtEquitySpread(0.123456, digits=4)
        assert "0.1235" in result


# ---------------------------------------------------------------------------
# fmtPriceOpt
# ---------------------------------------------------------------------------


class TestFmtPriceOpt:
    def test_non_zero_value_is_formatted(self):
        result = fmtPriceOpt(1.25)
        assert "1.25" in result

    @pytest.mark.parametrize("input_val", [0, None])
    def test_fmtPriceOpt_falsy_shows_nan(self, input_val):
        result = fmtPriceOpt(input_val)
        assert "nan" in result.lower()

    def test_custom_digits(self):
        result = fmtPriceOpt(1.2345, digits=4)
        assert "1.2345" in result

    def test_right_aligned_width_5(self):
        result = fmtPriceOpt(1.0)
        # format spec is ">5", so len >= 5
        assert len(result) >= 5


# ---------------------------------------------------------------------------
# ToolbarRenderer.render() — no-data / exception path
# ---------------------------------------------------------------------------


class TestToolbarRendererRender:
    def _make_renderer(self):
        """Build a ToolbarRenderer backed by a minimal mock app."""
        mock_app = MagicMock()
        # Attributes that render() increments / assigns
        mock_app.updates = 0
        mock_app.updatesReconnect = 0
        mock_app.localvars = {}

        # accountStatus must have required keys — raise KeyError to trigger except
        # (simulating "no data yet" by making accountStatus access fail)
        mock_app.accountStatus = {}  # "SMA" key missing → KeyError

        return ToolbarRenderer(app=mock_app)

    def test_render_returns_html_object(self):
        from prompt_toolkit.formatted_text import HTML
        renderer = self._make_renderer()
        result = renderer.render()
        assert isinstance(result, HTML)

    def test_render_no_data_returns_no_data_string(self):
        renderer = self._make_renderer()
        result = renderer.render()
        # When app has no account data, the except branch returns "No data yet..."
        assert "No data" in str(result)

    def test_render_with_full_account_status_succeeds(self):
        """render() completes the try block when all required account keys exist."""
        from prompt_toolkit.formatted_text import HTML

        mock_app = MagicMock()
        mock_app.updates = 0
        mock_app.updatesReconnect = 0
        mock_app.localvars = {}
        mock_app.accountStatus = {
            "SMA": 10000.0,
            "MaintMarginReq": 5000.0,
            "AvailableFunds": 50000.0,
        }
        # quoteStateSorted returns an empty iterable (no quotes)
        mock_app.quoteStateSorted = []
        mock_app.quoteState = {}
        mock_app.ib.openTrades.return_value = []
        mock_app.ib.portfolio.return_value = []
        mock_app.ib.fills.return_value = []
        mock_app.clientId = 1

        renderer = ToolbarRenderer(app=mock_app)
        result = renderer.render()
        assert isinstance(result, HTML)
        # With all required data present, should NOT return "No data yet..."
        assert "No data" not in str(result)

    def test_render_with_empty_quotes_shows_client_id(self):
        """When no quotes but account data is populated, clientId appears in output."""
        from prompt_toolkit.formatted_text import HTML

        mock_app = MagicMock()
        mock_app.updates = 5
        mock_app.updatesReconnect = 5
        mock_app.localvars = {}
        mock_app.accountStatus = {
            "SMA": 0.0,
            "MaintMarginReq": 0.0,
        }
        mock_app.quoteStateSorted = []
        mock_app.quoteState = {}
        mock_app.ib.openTrades.return_value = []
        mock_app.ib.portfolio.return_value = []
        mock_app.ib.fills.return_value = []
        mock_app.clientId = 42

        renderer = ToolbarRenderer(app=mock_app)
        result = renderer.render()
        assert "42" in str(result)
