"""Quote management extracted from IBKRCmdlineApp (cli.py).

Handles subscribing to market data, resolving positional quote references,
and keeping bag/leg ticker relationships in sync.
"""
from __future__ import annotations

import asyncio
import math
import re
from typing import Iterable

from loguru import logger

from ib_async import Bag, Contract, Future, FuturesOption

from icli.engine.contracts import contractForName, lookupKey, tickFieldsForContract
from icli.engine.calendar import sortQuotes
from icli.engine.primitives import as_duration, nan



class QuoteManager:
    """Manages live market-data subscriptions and positional quote lookups.

    Extracted from IBKRCmdlineApp so it can be tested without a full app instance.

    Parameters
    ----------
    ib:
        Active ib_async IB connection.
    quoteState:
        Shared dict mapping symbol key -> ITicker.  Mutated in place.
    quotesPositional:
        Shared list of (symkey, iticker) sorted by position.  Mutated in place.
    contractIdsToQuoteKeysMappings:
        Shared dict mapping conId -> quote key.  Mutated in place.
    conIdCache:
        Contract cache used by sortQuotes.
    idb:
        Instrument database (used for roundOrNothing and load).
    ol:
        buylang.OrderLanguage parser.
    app:
        Back-reference to IBKRCmdlineApp for cross-component calls
        (qualify, bagForSpread, contractForOrderRequest, decimals).
    """

    def __init__(
        self,
        ib,
        quoteState: dict,
        quotesPositional: list,
        contractIdsToQuoteKeysMappings: dict,
        conIdCache,
        idb,
        ol,
        app=None,
    ):
        self.ib = ib
        self.quoteState = quoteState
        self.quotesPositional = quotesPositional  # mutable list, shared by reference
        self.contractIdsToQuoteKeysMappings = contractIdsToQuoteKeysMappings
        self.conIdCache = conIdCache
        self.idb = idb
        self.ol = ol
        self._app = app  # back-reference for cross-component calls (qualify, bagForSpread, etc.)

    # ------------------------------------------------------------------
    # Positional resolution
    # ------------------------------------------------------------------

    def quoteResolve(self, lookup: str) -> tuple[str, Contract] | tuple[None, None]:
        """Resolve a local symbol alias like ':33' to current symbol name for the index."""

        # TODO: this doesn't work for futures symbols. Probably need to read the contract type
        #       to re-apply or internal formatting? futs: /; CFD: CFD; crypto: C; ...
        # TODO: fix this lookup if the number doesn't exist. (e.g. deleting :40 when quote 40 isn't valid
        #       results in looking up ":"[1:] which is just empty and it breaks.
        #       Question though: what do we return when a quote doesn't exist? Does the code using this method accept None as a reply?

        # extract out the number only here... (_ASSUMING_ we were called correct with ':33' etc and not just '33')
        lookupId = lookup[1:]

        if not lookupId:
            return None, None

        try:
            lookupInt = int(lookupId)
            _lookupsymbol, ticker = self.quotesPositional[lookupInt]
        except:
            # either the input wasn't ':number' or the index doesn't exist...
            return None, None

        # now we passed the integer extraction and the quote lookup, so return the found symbol for the lookup id
        assert ticker.contract
        name = (ticker.contract.localSymbol or ticker.contract.symbol).replace(" ", "")

        return name, ticker.contract

    def scanStringReplacePositionsWithSymbols(self, query: str) -> str:
        """Take an input string having any number of :N references and replace them with symbol names in the output.

        e.g. "Checking :32 for updates" -> "Checking AAPL for updates"
        """

        # We match to INCLUDE the ":" because `.quoteResolve()` _strips_ the leading `:` itself.
        return re.sub(
            r"(:\d+)",
            lambda match: self.quoteResolve(match.group(1))[0] or "NOT_FOUND",
            query,
        )

    async def positionalQuoteRepopulate(
        self, sym: str, exchange: str | None = "SMART"
    ) -> tuple[str | None, Contract | None]:
        """Given a symbol request which may contain :N replacement indicators, return resolved symbol instead."""

        assert sym

        # single symbol positional request
        if sym[0] == ":":
            foundSymbol, contract = self.quoteResolve(sym)
            # assert foundSymbol and contract and contract.symbol
            return foundSymbol, contract

        # single symbol no spaces
        if " " not in sym:
            try:
                contract = contractForName(sym, exchange=exchange)
            except Exception as e:
                # Note: don't make this logger.exception() except for temporary debugging.
                #       (because logger.exception pauses for 30+ when drawing stack trace during live sessions)
                logger.error("Contract creation failed: {}", str(e))
                return None, None

            (contract,) = await self._app.qualify(contract)
            assert contract and contract.conId

            return sym, contract

        def symFromContract(c):
            if isinstance(c, FuturesOption):
                # Need to construct OCC-like format so the symbol
                # parser can deconstruct it back into a contract:
                # symbol[date][right][strike]
                fsym = c.symbol

                # remove the leading "20"
                fdate = c.lastTradeDateOrContractMonth[2:]
                fright = c.right
                fstrike = c.strike
                tradingClass = c.tradingClass

                tradingClassExtension = f"-{tradingClass}" if tradingClass else ""
                return f"/{fsym}{fdate}{fright}{int(fstrike * 1000):08}{tradingClassExtension}"

            return c.localSymbol.replace(" ", "")

        # a symbol request with spaces which could require replacing resolved quotes inside of it
        rebuild: list[str] = []
        for part in sym.split():
            if part[0] == ":":
                foundSymbol, contract = self.quoteResolve(part)

                # logger.info("resolved: {} {}", foundSymbol, contract)

                # if we are adding a spread, combine each leg in their current order
                # (i.e. we aren't respecting the user buy/sell request, but rather just replacing
                #       the current spread as it exists from a different quote into this quote)
                if isinstance(contract, Bag):
                    # Look up all legs of this spread/bag
                    legs = contract.comboLegs
                    contracts = await self._app.qualify(
                        *[Contract(conId=leg.conId) for leg in legs]
                    )

                    # remove the previous two elements of the current rebuild list because they are
                    # just the "buy 1" before this ":nn" field.
                    rebuild = rebuild[:-2]

                    # now append all leg add commands based on their sides and ratios in the spread description
                    for leg, contract in zip(legs, contracts):
                        # We need to use our futures option syntax because we pass strings to the bag constructor,
                        # but futures options symbols aren't OCC symbols
                        rebuild.append(leg.action.lower())
                        rebuild.append(str(leg.ratio))
                        rebuild.append(symFromContract(contract))
                else:
                    assert foundSymbol
                    rebuild.append(symFromContract(contract))
            else:
                rebuild.append(part)

        # now put it back together again...
        sym = " ".join(rebuild)
        logger.info("Using add request: {}", sym)
        orderReq = self.ol.parse(sym)

        return sym, await self._app.bagForSpread(orderReq)

    # ------------------------------------------------------------------
    # Quote subscription
    # ------------------------------------------------------------------

    def addQuoteFromContract(self, contract) -> str:
        """Add live quote by providing a resolved contract"""
        # logger.info("Adding quotes for: {} :: {}", ordReq, contract)

        # just verify this contract is already qualified (will be a cache hit most likely)
        if isinstance(contract, Bag):
            assert all(
                [x.conId for x in contract.comboLegs]
            ), f"Sorry, your bag doesn't have qualified contracts inside of it? Got: {contract}"
        else:
            assert (
                contract.conId or contract.lastTradeDateOrContractMonth
            ), f"Sorry, we only accept qualified contracts for adding quotes, but we got: {contract}"

        # remove spaces from OCC-like symbols for consistent key reference
        symkey = lookupKey(contract)

        DELAYED = False
        # don't double-subscribe to symbols! If something is already in our quote state, we have an active subscription!
        if symkey not in self.quoteState:
            tickFields = tickFieldsForContract(contract)

            if isinstance(contract, Future):
                # enable delayed data quotes for VIX/VX/VXM quotes because they are in a non-default quote package
                # (Note: even though we mark delayed data here, I still get no results. TBD.)
                if contract.tradingClass.startswith("VX"):
                    # https://interactivebrokers.github.io/tws-api/market_data_type.html
                    DELAYED = True
                    self.ib.reqMarketDataType(3)

            # defend against some simple contracts not being qualified before reaching here
            if not contract.exchange:
                contract.exchange = "SMART"

            # logger.info("[{}] Adding new live quote: {}", symkey, contract)
            from icli.helpers import ITicker
            ticker = self.ib.reqMktData(contract, tickFields)
            self.quoteState[symkey] = ITicker(ticker, self._app)

            # Note: IBKR uses the same 'contract id' for all bags, so this is invalid for bags...
            self.contractIdsToQuoteKeysMappings[contract.conId] = symkey

            # if we enabled a delayed quote for a single reqMktData() call, return back to Live+Frozen quotes for regular requests
            if DELAYED:
                DELAYED = False
                self.ib.reqMarketDataType(2)

            # This is a nice debug helper just showing the quote key name to the attached contract subscription:
            # logger.info("[{}]: {}", symkey, contract)

            # re-comply all tickers when anything is added
            self.complyITickersSharedState()

        return symkey

    async def addQuotes(self, symbols: Iterable[str]):
        """Add quotes by a common symbol name"""
        if not symbols:
            return

        ors: list = []
        sym: str
        for sym in symbols:
            sym = sym.upper()
            # don't attempt to double subscribe
            # TODO: this only checks the named entry, so we need to verify we aren't double subscribing /ES /ESZ3 etc
            if sym in self.quoteState:
                continue

            # if this is a spread quote, attempt to replace any :N requests with the actual symbols...
            sym, _contract = await self.positionalQuoteRepopulate(sym)  # type: ignore

            # if creation failed, we can't process it...
            if not sym:
                continue

            orderReq = self.ol.parse(sym)
            ors.append(orderReq)

            # if this is a multi-part spread order, also add quotes for each leg individually!
            if orderReq.isSpread():
                for oo in orderReq.orders:
                    osym = oo.symbol
                    ors.append(self.ol.parse(osym))

        # technically not necessary for quotes, but we want the contract
        # to have the full '.localSymbol' designation for printing later.
        cs: list[Contract | None] = await asyncio.gather(
            *[self._app.contractForOrderRequest(o) for o in ors]
        )

        # logger.info("Resolved contracts: {}", cs)

        # the 'contractForOrderRequest' qualifies contracts before it returns, so
        # all generated contracts already have their fields populated correctly here.

        for ordReq, contract in zip(ors, cs):
            if not contract:
                logger.error(
                    "Failed to find live contract for: {} :: {}", ordReq, contract
                )
                continue

            symkey = self.addQuoteFromContract(contract)

        # check if all contracts exist in the instrumentdb (and schedule creating them if not)
        self.idb.load(*cs)

        # return contracts added
        return list(filter(None, cs))

    # ------------------------------------------------------------------
    # Quote state accessors
    # ------------------------------------------------------------------

    @property
    def quoteStateSorted(self):
        """Return the EXACT toolbar ticker/quote content in position-accurate iteration order.

        This can be used for iterating quotes/tickers by position if we need to elsewhere."""
        tickersSortedByPosition = sorted(
            self.quoteState.items(), key=lambda x: sortQuotes(x, self.conIdCache)
        )

        # replace the global mapping each time too
        self.quotesPositional = tickersSortedByPosition

        return tickersSortedByPosition

    def quoteExists(self, contract) -> bool:
        return lookupKey(contract) in self.quoteState

    # ------------------------------------------------------------------
    # Live quote retrieval
    # ------------------------------------------------------------------

    def currentQuote(self, sym, show=True) -> tuple[float | None, float | None]:
        # TODO: maybe we should refactor this to only accept qualified contracts as input (instead of string symbol names) to avoid naming confusion?
        q = self.quoteState.get(sym)

        # if quote did not exist, then we need to check it...
        assert q and q.contract, f"Why doesn't {sym} exist in the quote state?"

        # use our potentially synthetically-derived live quotes for spreads (if IBKR isn't quoting us bid/ask for some reason)
        current = q.quote()

        # (we use 'is not None' here because with spreads, a bid or ask of $0 _is_ valid for credit spreads)
        hasBid = current.bid is not None
        hasAsk = current.ask is not None

        hasQuotes = hasBid or hasAsk

        # only optionally print the quote because printing technically requires extra time
        # for all the formatting and display output
        if show and hasQuotes:
            nowpy = self._app.nowpy
            ago = (
                "now"
                if q.time >= nowpy
                else as_duration((nowpy - (q.time or nowpy)).total_seconds())
            )

            if q.lastTimestamp:
                agoLastTrade = (
                    "now"
                    if q.lastTimestamp >= nowpy
                    else as_duration((nowpy - q.lastTimestamp).total_seconds())
                )
            else:
                agoLastTrade = "never received"

            digits = self._app.decimals(q.contract)

            assert current.bid is not None or current.ask is not None

            # if no bid, just use ask or last, whichever exists first
            if not hasBid:
                mid = current.ask or q.last
            else:
                # another redundant check to help the type checker stop complaining
                assert current.bid is not None and current.ask is not None
                mid = (current.bid + current.ask) / 2

            # don't pass zero bid/ask calculation to the DB or else it gets upset
            # NOTE: If there is zero tick width between bid/ask (e.g. $0.80 bid, $0.85 ask on a $0.05 tick),
            #       then the midpoint will be either the bid or the ask directly because it can't actually
            #       divide them in half.
            if mid:
                mid = self.idb.roundOrNothing(q.contract, mid) or mid

            # fmt: off
            show = [
                f"{q.contract.secType} :: {q.contract.localSymbol or q.contract.symbol}:",
                f"bid {current.bid:,.{digits}f} x {current.bidSize}" if q.bid else "bid NONE",
                f"mid {mid:,.{digits}f}",
                f"ask {current.ask:,.{digits}f} x {current.askSize}",
                f"last {q.last:,.{digits}f} x {q.lastSize}" if q.last else "(no last trade)",
                f"last trade {agoLastTrade}" if q.lastTimestamp else "(no last trade timestamp)",
                f"updated {ago}" if q.time else "(no last update time)",
            ]
            # fmt: on
            logger.opt(depth=1).info("    ".join(show))

        # updated price picking logic: if we have a live bid/ask, return them.
        # else, if we don't have a bid/ask, use the last reported price (if it exists).
        # else else, return nothing because there's no actual price we can read anywhere.

        # if no quote yet (or no prices available), return last seen price...
        if hasQuotes:
            return current.bid, current.ask

        # if last exists, use it.
        if q.last is not None:
            last = q.last
            return last, last

        # else, we found no valid price option here.
        return None, None

    # ------------------------------------------------------------------
    # Ticker relationship compliance
    # ------------------------------------------------------------------

    def complyITickersSharedState(self) -> None:
        """Iterate all subscribed tickers looking to attach bags to their legs and legs to their bags."""
        # We need to evalute all subscribed bags so their .legs match.
        # Also we need to attach each contract leg to bag(s) it belongs to.
        # This should be run after any bag addition or removal because the tickers themselves don't have
        # access to the full quote state to read other tickers (so we must manually attach related tickers).

        idToTicker = lambda x: self.quoteState.get(
            self.contractIdsToQuoteKeysMappings.get(x)  # type: ignore
        )

        # first, reset all membership, then re-add all membership...
        for symkey, iticker in self.quoteState.items():
            iticker.legs = tuple()
            iticker.bags.clear()

        # now attach each leg to its owning bags, and populate each bag with its leg tickers
        for symkey, iticker in self.quoteState.items():
            t = iticker.ticker

            if isinstance(t.contract, Bag):
                width = 0.0
                legs = list()
                for leg in t.contract.comboLegs:
                    legTicker = idToTicker(leg.conId)

                    # add this bag as being used by the current leg
                    if legTicker:
                        legTicker.bags.add(iticker)

                    # generate leg and ticker descriptors for the bag
                    match leg.action:
                        case "BUY":
                            legs.append((leg.ratio, legTicker))
                            if legTicker:
                                width += (
                                    leg.ratio
                                    * (legTicker.contract.strike or nan)
                                    * (-1.0 if legTicker.contract.right == "C" else 1.0)
                                )
                        case "SELL":
                            legs.append((-leg.ratio, legTicker))
                            if legTicker:
                                width += (
                                    leg.ratio
                                    * (legTicker.contract.strike or nan)
                                    * (-1.0 if legTicker.contract.right == "P" else 1.0)
                                )
                        case _:
                            logger.warning("Unexpected action? Got: {}", leg.action)

                # attach leg descriptors to bag ticker
                assert (
                    len(legs) == len(t.contract.comboLegs)
                ), "Why didn't we populate all legs into the legs descriptors? Our math is invalid if this happens."

                iticker.legs = tuple(legs)
                iticker.width = width
                iticker.updateGreeks()
