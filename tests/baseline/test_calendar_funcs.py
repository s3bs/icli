"""Tests for icli.engine.calendar — standalone functions extracted from cli.py."""

import datetime
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd

from icli.engine.calendar import (
    mkcolor, mkPctColor, sortQuotes, invertstr,
    tradingDaysRemainingInMonth, tradingDaysRemainingInYear,
    tradingDaysNextN, readableHTML,
    goodCalendarDate,
)


class TestTradingDaysRemainingInMonth:
    def test_returns_valid_count(self):
        result = tradingDaysRemainingInMonth()
        assert isinstance(result, int)
        assert 0 <= result <= 23


class TestTradingDaysRemainingInYear:
    def test_returns_valid_count(self):
        result = tradingDaysRemainingInYear()
        assert isinstance(result, int)
        assert 0 <= result <= 253


class TestTradingDaysNextN:
    def test_excludes_weekends(self):
        result = tradingDaysNextN(10)
        assert isinstance(result, list)
        assert len(result) > 0
        for ts in result:
            assert ts.weekday() < 5, f"Weekend date found: {ts}"


class TestMkcolor:
    def test_zero_returns_unchanged(self):
        result = mkcolor(0, "test", [-0.5, 0, 0.5])
        assert result == "test"

    def test_list_input_returns_list(self):
        result = mkcolor(0.5, ["a", "b"], [-0.5, 0, 0.5])
        assert isinstance(result, list)
        assert len(result) == 2


class TestMkPctColor:
    def test_zero_returns_unchanged(self):
        result = mkPctColor(0, "0%")
        assert result == "0%"


class TestReadableHTML:
    def test_strips_tags(self):
        result = readableHTML("<b>bold</b>")
        assert result.strip() == "bold"

    def test_preserves_text_content(self):
        result = readableHTML("plain text")
        assert "plain text" in result

    def test_empty_string_returns_empty(self):
        result = readableHTML("")
        assert result == ""

    def test_nested_tags_flattened(self):
        result = readableHTML("<div><p>text</p></div>")
        assert "text" in result


class TestInvertstr:
    def test_inverts_lowercase(self):
        assert invertstr("abc") == "zyx"
        assert invertstr("z") == "a"

    def test_non_alpha_unchanged(self):
        assert invertstr("123") == "123"
