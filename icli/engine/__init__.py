"""icli engine layer — pure trading logic with no UI dependency.

This package contains the extracted, testable core of icli.
All modules use ``from __future__ import annotations`` and modern
Python typing (``str | None``, ``@dataclass(slots=True)``, etc.).

Modules
-------
exchanges
    Futures exchange mappings and contract metadata.
    - ``FutureSymbol``: immutable dataclass (symbol, exchange, name, tradingClass, currency, hasOptions)
    - ``FUTS_EXCHANGE``: dict[str, FutureSymbol] — 500+ IBKR futures keyed by symbol *and* tradingClass

technicals
    Time-weighted EMA and ATR calculations.
    - ``TWEMA``: time-weighted EMA with diff/score tracking across multiple durations
    - ``ATR``: bar-based Average True Range (O(1) rolling RMA)
    - ``ATRLive``: ATR adapter for live tick streams (keeps a small price buffer)
    - ``rmsnorm``: RMS normalization helper for EMA score calculations
    - ``RTH_EMA_VWAP``: constant (23,400s = 6.5h Regular Trading Hours)
    - ``analyze_trend_strength``: multi-timeframe trend direction and strength analysis
    - ``generate_trend_summary``: human-readable trend summary from EMA data

orders
    Order type factory for ib_async.
    - ``IOrder``: dataclass wrapping common order params; ``.order(type_str)`` dispatches
      to 20+ IBKR order types (limit, adaptive, trailing, peg, MOO/MOC, combos, etc.)
    - ``CLIOrderType``: enum of all IBKR order type strings
    - ``markOrderNotGuaranteed``: tag a combo order for independent leg execution

audio
    Text-to-speech and sound alert client.
    - ``AwwdioClient``: async httpx client for the Awwdio TTS service

primitives
    Pure types, constants, and utility functions (stdlib-only, no ib_async).
    - Constants: ``ALGOMAP``, ``D100``, ``DN1``, ``DP1``, ``FUTS_MONTH_MAPPING``, ``PQ``, ``nan``
    - Type aliases: ``BuySell``, ``ContractId``, ``FPrice``, ``MaybePrice``, ``PercentAmount``, ``Seconds``
    - Dataclasses: ``FillReport``, ``PaperLog``, ``QuoteSizes``, ``LevelLevels``, ``LevelBreacher``,
      ``QuoteFlowPoint``, ``QuoteFlow``, ``Bracket``, ``LadderStep``, ``PriceOrQuantity``
    - Functions: ``fmtmoney``, ``convert_futures_code``, ``find_nearest``, ``sortLocalSymbol``,
      ``portSort``, ``tradeOrderCmp``, ``boundsByPercentDifference``, ``split_commands``,
      ``convert_time``, ``as_duration``

contracts
    Contract creation, parsing, and utilities for ib_async contracts.
    - ``contractForName``: parses text formats ("AAPL", "/ES", "I:SPX", OCC options) into Contract objects
    - ``nameForContract``: generates readable text descriptions from Contract objects
    - ``contractFromTypeId``, ``contractFromSymbolDescriptor``: reconstruct contracts from cached keys
    - ``contractToSymbolDescriptor``: generates unique cache keys for contracts
    - ``tickFieldsForContract``: returns CSV tick field IDs for market data subscriptions
    - ``parseContractOptionFields``: extracts option-specific fields (date, strike, right)
    - ``isset``: checks if IBKR float value is real vs UNSET_DOUBLE sentinel
    - ``lookupKey``: generates immutable lookup keys for contract caching
    - ``TradeOrder``, ``FullOrderPlacementRecord``: order result containers
    - ``CompleteTradeNotification``: async event wrapper for order completion
    - ``getExpirationsFromTradier``: fetches option expirations from Tradier API
    - Module-level ``FUT_EXP``: must be set before calling ``contractForName()``

algobinder
    WebSocket client for external algo data feed ingestion.
    - ``AlgoBinder``: connects to external WebSocket, saves results to dot-addressable dict
"""

# Convenience re-exports for common usage:
# from icli.engine import ATR, TWEMA, IOrder, FUTS_EXCHANGE
from icli.engine.technicals import ATR, ATRLive, TWEMA
from icli.engine.orders import IOrder, CLIOrderType
from icli.engine.exchanges import FUTS_EXCHANGE, FutureSymbol
from icli.engine.audio import AwwdioClient

__all__ = [
    "ATR",
    "ATRLive",
    "TWEMA",
    "IOrder",
    "CLIOrderType",
    "FUTS_EXCHANGE",
    "FutureSymbol",
    "AwwdioClient",
]
