"""Predicate data binding for the ifthen conditional system.

Extracted from cli.py: predicateSetup() and dataExtractorForTicker().

These functions bind ifthen predicate descriptions to live data sources
(ITicker fields, EMA values, ATR, greeks, portfolio positions, algobinder)
so predicates can evaluate their conditions against real-time data.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Coroutine, Hashable
from typing import Any

from loguru import logger

import tradeapis.ifthen as ifthen

from icli.helpers import ITicker
from icli.engine.technicals import TWEMA


class PredicateEngine:
    """Binds ifthen predicates to live market data sources.

    Dependencies are injected as callbacks so the engine has no direct coupling
    to IBKRCmdlineApp, IB, or diskcache.

    Parameters
    ----------
    quoteState:
        Live dict mapping symbol key -> ITicker.
    positional_quote_repopulate:
        async (sym, exchange?) -> (name, Contract|None).  Resolves positional
        references (`:0`, `:1`) and subscribes if needed.
    add_quote_from_contract:
        (Contract) -> str.  Subscribe to market data, return symbol key.
    name_for_contract:
        (Contract) -> str.  Human-readable name for a contract.
    algobinder_read:
        (field: str) -> value.  Read a value from the algo binder.  May be
        None if algo system is not active.
    ensure_algobinder:
        () -> None.  Ensure the algo binder task is running (lazy start).
    portfolio_position:
        (conId: int) -> PortfolioItem | None.  Look up a live position by
        contract ID.  Returns an object with .averageCost and .position attrs.
    """

    def __init__(
        self,
        *,
        quoteState: dict[str | Hashable, ITicker],
        positional_quote_repopulate: Callable,
        add_quote_from_contract: Callable,
        name_for_contract: Callable[[Any], str],
        algobinder_read: Callable[[str], Any] | None,
        ensure_algobinder: Callable[[], None],
        portfolio_position: Callable[[int], Any],
    ):
        self.quoteState = quoteState
        self._positional_quote_repopulate = positional_quote_repopulate
        self._add_quote_from_contract = add_quote_from_contract
        self._name_for_contract = name_for_contract
        self._algobinder_read = algobinder_read
        self._ensure_algobinder = ensure_algobinder
        self._portfolio_position = portfolio_position

    def data_extractor_for_ticker(
        self, iticker: ITicker, field: str, timeframe: int
    ) -> Callable | None:
        """Return a zero-argument function querying the live 'iticker' for 'field' and potentially 'timeframe' updates."""
        fetcher = None

        # a dot in the field means we HAVE AN ALGO! ALGO ALERT! ALGO ALERT!
        # TODO: maybe move this to an algo: prefix instead of just any dots?
        if "." in field:
            self._ensure_algobinder()

            algobinder_read = self._algobinder_read
            assert algobinder_read is not None

            # Note: it's up to the user ensuring a 100% correct algo field description for the full 3, 5, 8+ level depth they expect...
            return lambda *args: algobinder_read(field)

        def emaByField(subtype):
            parts = subtype.split(":")

            # match first component to the instance variable names of ITicker
            match parts[0]:
                case "price" | "p":
                    src = "ema"
                case "trade" | "tr":
                    src = "emaTradeRate"
                case "volume" | "vol":
                    src = "emaVolumeRate"
                case "iv":
                    src = "emaIV"
                case "delta" | "d":
                    src = "emaDelta"
                case "vega" | "v":
                    src = "emaVega"
                case _:
                    src = "ema"
                    logger.warning(
                        "No EMA source provided, defaulting to 'price' (other choices: 'delta' or 'iv' or 'vega' or 'trade' or 'volume')"
                    )

            # fetch ITicker instance variable by name
            base: TWEMA = getattr(iticker, src)

            # fetch sub-components of the EMA object
            match ":".join(parts[1:]):
                case "ema":
                    # our time-weighted ema
                    fetcher = lambda *args: base[timeframe]
                case "rms":
                    fetcher = lambda *args: base.rms()[timeframe]
                case "ema:prev:log":
                    # difference between period N and period N-1 expressed as percentage returns decayed by 'timeframe' ema
                    # (populated for all EMA durations)
                    fetcher = lambda *args: base.diffPrevLog[timeframe]
                case "ema:prev:score":
                    # CURRENT weighted sum of every period (N, N-1) pair between 0 and 6.5 hours of EMA
                    # (ranged-returns are weighted more heavily towards lower timeframes as (1/15, 1/30, 1/60, ...))
                    fetcher = lambda *args: base.diffPrevLogScore
                case "ema:prev:score:ema":
                    # EMA of weighted sum of every period (N, N-1) pair between 0 and 6.5 hours of EMA decayed by 'timeframe' ema
                    # (ranged-returns are weighted more heavily towards lower timeframes as (1/15, 1/30, 1/60, ...))
                    # Note: score:ema of [0] is the same as just the instantenous 'prev:score' too.
                    fetcher = lambda *args: base.diffPrevLogScoreEMA[timeframe]
                case "ema:vwap:log":
                    # difference between the 6.5 hour EMA and the current price expressed as percentage returns
                    # (populated for all EMA durations)
                    fetcher = lambda *args: base.diffVWAPLog[timeframe]
                case "ema:vwap:score":
                    # CURRENT weighted sum of every period (N, N-1) pair against the 6.5 hour EMA
                    fetcher = lambda *args: base.diffVWAPLogScore
                case "ema:vwap:score:ema":
                    # EMA of weighted sum of every period (N, N-1) pair against the 6.5 hour EMA decayed by 'timeframe' ema
                    fetcher = lambda *args: base.diffVWAPLogScoreEMA[timeframe]
                case _:
                    assert None, (
                        f"Invalid EMA sub-fields requested? Full request was for: {subtype}"
                    )

            return fetcher

        name_for_contract = self._name_for_contract
        portfolio_position = self._portfolio_position

        # We _can_ do case insenstiive matches for these:
        match field.lower():
            case "bid" | "ask":
                # Note: we use quote() bid/ask because the directly .bid/.ask values
                #       on the ticker may not be updating if IBKR breaks spread quotes
                fetcher = lambda *args: getattr(iticker.quote(), field)
            case "mid" | "midpoint" | "live":
                fetcher = lambda *args: iticker.quote().current
            case "last" | "high" | "low" | "open" | "close":
                fetcher = lambda *args: getattr(iticker.ticker, field)
            case "atr":
                # selectable as 90, 180, 300, 600, 1800 second ATRs
                fetcher = lambda *args: iticker.atrs[timeframe].atr.current
            case "sym" | "symbol":
                # just return symbol as string...
                fetcher = lambda *args: name_for_contract(iticker.ticker.contract)  # type: ignore
            case "half":
                # half way between high and low
                fetcher = lambda *args: (iticker.ticker.high + iticker.ticker.low) / 2
            case "vwap":
                # vwap is a special case because if VWAP doesn't exist, we want to use our 6.5 hour EMA instead
                fetcher = lambda *args: iticker.ticker.vwap or iticker.ema[23_400]
            case "cost":
                # fetch live averageCost for position as reported by portfolio reporting
                # TODO: if contract is a bag with no single contract id, just add averageCost of ALL internal contract ids together?

                # Note: the IBKR portfolio API lists contracts at their multiplier-adjusted price, be want the quoted contract price,
                #       so we de-adjust them by multipliers back to contract prices again.
                # Also note: IBKR portfolio prices have positive prices but negative quantities for shorts, but we want negative prices
                #            for short positions too, so we also adjust accordingly.
                c = iticker.ticker.contract
                assert c

                contractId = c.conId
                mul = float(c.multiplier or 1)
                fetcher = lambda *args: (
                    portfolio_position(contractId).averageCost
                    / math.copysign(portfolio_position(contractId).position, mul)
                )
            case "qty":
                # fetch live qty for position as reported by portfolio reporting
                assert iticker.ticker.contract

                contractId = iticker.ticker.contract.conId
                fetcher = lambda *args: portfolio_position(contractId).position
            case "theta" | "delta" | "iv" | "gamma" | "d" | "g" | "t" | "v" | "vega":
                # allow some shorthand back to actual property names
                match field:
                    case "iv":
                        field = "impliedVol"
                    case "d":
                        field = "delta"
                    case "t":
                        field = "theta"
                    case "g":
                        field = "gamma"
                    case "v":
                        field = "vega"

                fetcher = lambda *args: getattr(iticker.modelGreeks, field)
            case parts if ":" in parts:
                fetcher = emaByField(parts)
            case _:
                logger.warning("Unexpected field requested? This won't work: {}", field)

        return fetcher

    async def setup(
        self,
        prepredicate: ifthen.CheckableRuntime,
        *,
        task_create_fn: Callable,
        extension_fns: dict[str, Callable],
    ) -> None:
        """Attach data extractors and custom functions to all predicates inside 'prepredicate'.

        The ifthen predicate language only _describes_ what to check, but we must provide the predicate(s)
        with the actual data sources and custom functions so the predicate(s) can execute their checks
        with live data on every update.

        Parameters
        ----------
        prepredicate:
            The runtime predicate object containing inner predicates to bind.
        task_create_fn:
            Callable to create a background task: (name, coroutine) -> None.
        extension_fns:
            Maps function names to their implementations.  Known keys:
            "verticalput"/"vp", "verticalcall"/"vc", "position"/"pos"/"p", "abs".
        """

        # Now we need to traverse the entire predicate condition hierarchy so we can bind
        # individual DataExtractor instances to live data elements to use for value extraction.

        symbolToTickerMap: dict[Hashable, ITicker] = {}

        pid = prepredicate.id

        # a CheckableRuntime predicate condition may have multiple inner predicates, so we must
        # introspect all _inner_ predicates to properly attach their data accessors.
        for predicate in prepredicate.actives:
            extractedSymbols = set()

            # Upon return of PARSED result, we need to:
            #  For each symbol in parsed.symbols, get contracts via: await self.state.positionalQuoteRepopulate(sym)
            # We need to pass in a unified 'datasource' capable of looking up (symbol (string or contract id or ticker (?????)), FIELD (SMA), TIMEFRAME (35s), SUBFIELD (5 lookback or 10 lookback))
            #  - though, instead of needing string/id/ticker binding, we attach a custom closure already attached to the target symbol so it is always bound to the correct data source provider itself.
            #     - but then we have to pick between: Are we looking up a field on the ITicker or a field in the Algo Feed?
            #        - for Algo Feed, we also then need to save each {Symbol -> {Duration -> {algo -> {result: value}}}} and repopulate {Symbol -> {Duration...}} when a new Duration change is received.
            for symbol in predicate.actuals:
                assert isinstance(symbol, str)
                foundsym, c = await self._positional_quote_repopulate(symbol)

                logger.info("[{}] Tracking contract: {}", pid, c)

                # subscribe if not subscribed (no-op if already subscribed, but returns symkey either way)
                try:
                    symkey = self._add_quote_from_contract(c)
                except:
                    logger.warning("[{}] Live contract not found?", c)
                    continue

                extractedSymbols.add(symkey)

                # now fetch subscribed ticker
                iticker = self.quoteState.get(symkey)
                assert iticker
                # logger.info("Tracking ticker: {}", iticker)

                # record ORIGINAL symbol to ticker map so we can re-bind them after this loop
                symbolToTickerMap[symbol] = iticker

            # Replace potentially positional symbols with full symbol details we use for lookups
            predicate.symbols = frozenset(extractedSymbols)

            for extractor in predicate.extractors():
                # We prefer 'actual 'here because if this is being _repopulated_ after a reconnect(),
                # the original 'symbol' isn't valid (if it's a position request) but we added 'actual'
                # when the predicate was first created to represent the _real_ symbol data needed.
                if iticker := symbolToTickerMap.get(extractor.actual or extractor.symbol):
                    # Note: DO NOT .lower() 'extractor.field' because the fields are case sensitive when doing algo lookups...
                    datafield = extractor.datafield
                    timeframe = extractor.timeframe

                    # TODO: allow 'iticker' to generate text descriptions of spreads for usage in places...
                    # TODO: create helper which takes a contract and generates a compatible description we can re-parse
                    #        e.g. Future ESZ4 -> /ESZ4, options generation, index generation, ........
                    extractor.actual = self._name_for_contract(iticker.contract)

                    logger.info(
                        "[{}] Assigning field extractor: {} ({}) @ {} {}",
                        pid,
                        extractor.symbol,
                        extractor.actual,
                        datafield,
                        timeframe or "",
                    )

                    assert datafield
                    fetcher = self.data_extractor_for_ticker(iticker, datafield, timeframe or 0)

                    extractor.datafetcher = fetcher

            # now do the same for functions (if any)
            fnfetcher: (
                Callable[[dict[str, Any], str, float, float], Coroutine[Any, Any, Any]]
                | Callable[[str], Any]
            )

            for fn in predicate.functions():
                match fn.datafield.lower():
                    case "verticalput" | "vp":
                        fnfetcher = extension_fns["verticalput"]
                    case "verticalcall" | "vc":
                        fnfetcher = extension_fns["verticalcall"]
                    case "position" | "pos" | "p":
                        fnfetcher = extension_fns["position"]
                    case "abs":
                        fnfetcher = extension_fns["abs"]

                fn.scheduler = functools.partial(
                    task_create_fn, f"[{pid}] predicate executor for {fn.datafield}"
                )
                fn.datafetcher = fnfetcher
