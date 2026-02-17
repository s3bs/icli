"""Characterisation tests for futsexchanges.py — futures exchange mappings."""

import pytest
from icli.futsexchanges import FUTS_EXCHANGE, FutureSymbol


class TestFutsExchange:
    def test_structural_invariants(self):
        """Every entry must have symbol, exchange, currency, and bool hasOptions."""
        assert len(FUTS_EXCHANGE) > 50, "Mapping looks truncated"
        for key, val in FUTS_EXCHANGE.items():
            assert isinstance(val, FutureSymbol), f"{key}: not FutureSymbol"
            assert val.exchange, f"{key}: missing exchange"
            assert val.currency, f"{key}: missing currency"
            assert isinstance(val.hasOptions, bool), f"{key}: hasOptions not bool"

    def test_key_matches_symbol_except_vx_weeklies(self):
        """Dict key should match .symbol, except for known exceptions.

        Most entries have key == symbol. Known exceptions: VX weeklies (VX01..VX52)
        where key != symbol='VIX', and many EUREX/international futures that use
        short codes like '6A' (AUD), 'EH' (AC), '6E' (EUR), etc.
        """
        # These are known exception categories where key != symbol is expected
        known_exception_symbols = {
            "VIX",      # VX01-VX52 all have symbol='VIX'
            "AUD", "CAD", "CHF", "EUR", "GBP", "JPY", "MXP", "NZD", "ZAR",  # Currency futures
            "AC", "ACDX", "AIGCI", "BQX", "BRE", "BRR", "BTCEURRR", "BTP", "BTS",  # Various EUREX/crypto
            "CLP", "DA", "DAX", "DBEA", "DDAX", "DDXDIVPT", "DESX5", "DJ200", "DJ200L",
            "DJ200S", "DJ600", "DJES", "DJESL", "DJESM", "DJESS", "DJSD", "DJUBS", "DJUSRE",
            "EJ", "EO", "ESA", "ESE", "ESF", "ESI", "ESM", "ESTR", "ESTX50", "ESU",
            "ETHEURRR", "ETHUSDRR", "EU3", "GB", "GBL", "GBM", "GBS", "GBX", "GSCI", "IBAA",
            "IBXXIBHY", "IBXXIBIG", "IXB", "IXE", "IXI", "IXR", "IXU", "IXV", "IXY",
            "K200M.EU", "KS200.EU", "KWY.EU", "LRC30APR", "M1CN", "M1EF", "M1EU", "M1IN",
            "M1JP", "M1MS", "M1MSA", "M1TW", "M1WO", "M7EU", "MBWO", "MDAX", "MHNG",
            "MXEA", "MXEF", "MXEU", "MXJPUS", "MXUS", "MXWO", "NF", "NIY", "NYFANG",
            "OAT", "OMXH25", "RUR *", "RZ", "SD3ED", "SGUF", "SGX", "SIXM", "SIXRE",
            "SIXT", "SLI", "SMI", "SMIDP", "SMIM", "SOFR1", "SOFR3", "SONIA", "SPXDIVAN",
            "SPXESUP", "ST3", "STX", "SVX", "SX3E", "SX3P", "SX4E", "SX4P", "SX6P",
            "SX7E", "SX7P", "SX86P", "SX8E", "SX8P", "SXAP", "SXDE", "SXDP", "SXEP",
            "SXFP", "SXIP", "SXKE", "SXKP", "SXMP", "SXNE", "SXNP", "SXOE", "SXOP",
            "SXPE", "SXPP", "SXQE", "SXQP", "SXRE", "SXRP", "SXTE", "SXTP", "SXXPESGX",
            "TDX", "V2TX", "YC", "YK", "YW", "ZY",
        }

        # Collect actual unexpected mismatches
        unexpected_mismatches = []
        for key, val in FUTS_EXCHANGE.items():
            if key != val.symbol and val.symbol not in known_exception_symbols:
                unexpected_mismatches.append(f"{key} -> {val.symbol}")

        assert not unexpected_mismatches, f"Unexpected key/symbol mismatches: {unexpected_mismatches}"

    def test_critical_contracts_present(self):
        """Contracts you'd actually trade — if these are missing the mapping is broken."""
        critical = {
            "ES": "CME",      # E-mini S&P
            "NQ": "CME",      # E-mini Nasdaq
            "CL": "NYMEX",    # Crude oil
            "GC": "COMEX",    # Gold
            "ZB": "CBOT",     # Treasury bond
            "ZN": "CBOT",     # 10-year note
        }
        for symbol, expected_exchange in critical.items():
            assert symbol in FUTS_EXCHANGE, f"{symbol} missing from mapping"
            assert FUTS_EXCHANGE[symbol].exchange == expected_exchange, \
                f"{symbol}: expected {expected_exchange}, got {FUTS_EXCHANGE[symbol].exchange}"
