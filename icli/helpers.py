"""A refactor-base for splitting out common helpers between cli and lang"""

from __future__ import annotations

import asyncio
import bisect
import enum
import functools
import locale
import math
import platform
import re
import statistics
import time
import types

import dateutil
import numpy as np
import pandas as pd

ourjson: types.ModuleType
# Only use orjson under CPython, else use default json (because `json` under pypy is faster than orjson)
# (also we are doing this "import name, assign name to global" to get around mypy complaining about double importing with the same alias)
if platform.python_implementation() == "CPython":
    import orjson

    ourjson = orjson
else:
    import json

    ourjson = json

import datetime
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from decimal import Decimal
from functools import cached_property
from typing import *

import httpx
import ib_async  # just for UNSET_DOUBLE
import questionary
import tradeapis.cal as tcal
import websockets
from cachetools import cached
from dotenv import dotenv_values
from ib_async import (
    CFD,
    Bag,
    Bond,
    Commodity,
    ContFuture,
    Contract,
    Crypto,
    Forex,
    Future,
    FuturesOption,
    Index,
    MutualFund,
    Option,
    OptionComputation,
    Order,
    Stock,
    Ticker,
    Trade,
    Warrant,
)
from loguru import logger
from questionary import Choice
from tradeapis.orderlang import (
    DecimalCash,
    DecimalPercent,
    DecimalShares,
    OrderIntent,
)

from icli.engine.algobinder import AlgoBinder
from icli.engine.exchanges import FUTS_EXCHANGE

from icli.engine.technicals import ATRLive, RTH_EMA_VWAP, TWEMA, analyze_trend_strength, generate_trend_summary

from icli.engine.primitives import (
    nan, ALGOMAP, D100, DN1, DP1, FUTS_MONTH_MAPPING, PQ,
    BuySell, ContractId, FPrice, MaybePrice, PercentAmount, Seconds,
    FillReport, PaperLog, QuoteSizes, LevelLevels, LevelBreacher,
    QuoteFlowPoint, QuoteFlow, Bracket, LadderStep, PriceOrQuantity,
    fmtmoney, convert_futures_code, find_nearest, sortLocalSymbol,
    portSort, tradeOrderCmp, boundsByPercentDifference,
    split_commands, convert_time, as_duration,
)

from icli.engine.contracts import (
    TradeOrder, FullOrderPlacementRecord,
    nameForContract, contractForName, contractToSymbolDescriptor,
    contractFromTypeId, contractFromSymbolDescriptor,
    tickFieldsForContract, parseContractOptionFields,
    strFromPositionRow, isset, lookupKey,
    CompleteTradeNotification, getExpirationsFromTradier,
)

# auto-detect next index futures expiration month based on roll date
# we add some padding to the futs exp to compensate for having the client open a couple days before
# (which will be weekends or sunday night, which is fine)
futexp: Final = tcal.nextFuturesRollDate(
    datetime.datetime.now().date() + datetime.timedelta(days=2)
)

# Also compare: https://www.cmegroup.com/trading/equity-index/rolldates.html
logger.info("Futures Next Roll-Forward Date: {}", futexp)
FU_DEFAULT = dict(ICLI_FUT_EXP=f"{futexp.year}{futexp.month:02}")  # YM like: 202309
FU_CONFIG = {**FU_DEFAULT, **dotenv_values(".env.icli"), **os.environ}  # type: ignore

# TEMPORARY OVERRIDE FOR EXPIRATION WEEK OPTION PROBLEMS
# TODO: for futures options, we need to reead the detail "uynderlying contract" to use for distance and quotes instead of the live symbol. sigh.
FUT_EXP = FU_DEFAULT["ICLI_FUT_EXP"]
# FUT_EXP = "202409"
import icli.engine.contracts as _contracts
_contracts.FUT_EXP = FUT_EXP

@dataclass(slots=True)
class ITicker:
    """Our own version of a ticker with more composite and self-reporting details."""

    # underlying live-updating Ticker object
    ticker: Ticker

    # map of cached contract ids to contracts
    # (globally shared contract cache)
    state: Any

    # proper contract name is populated on instance creation
    name: str = "UNDEFINED"

    # just a recentHistoryAnchor price history for every new price update.
    # length assumes we get 4 price updates per second and we want 1 minute of history.
    history: deque[float] = field(default_factory=lambda: deque(maxlen=60 * 4))

    # hold EMA of instrument price over different time periods.
    ema: TWEMA = field(default_factory=TWEMA)

    # For options, also track some extra fields over time...
    emaIV: TWEMA = field(default_factory=TWEMA)
    emaDelta: TWEMA = field(default_factory=TWEMA)
    emaVega: TWEMA = field(default_factory=TWEMA)

    # ticker stats history
    emaTradeRate: TWEMA = field(default_factory=TWEMA)
    emaVolumeRate: TWEMA = field(default_factory=TWEMA)

    # synthetic ATR using our own accounting.
    # calculate live ATR based on quote updates
    atrs: dict[int, ATRLive] = field(default_factory=dict)

    # if this is a spread, we track individual legs.
    # The format is (ratio, ITicker).
    # Short legs will have a negative ratio number so the math works out.
    # Note: these are tuple() because they must all be updated at the same time instead of mutated.
    legs: tuple[tuple[int, ITicker | None], ...] = tuple()

    # if this is a single contract CURRENTLY PARTICIPATING IN OPEN BAG QUOTES, track all bags here too.
    # We want to track the bags so when a leg contract ticks, we update all bag quotes at the same time.
    bags: set[ITicker] = field(default_factory=set)

    # if spread, the "width" of the spread (may not be valid for more than 2 legs, but we try)
    width: float | int = 0

    alerts: dict[int | str, dict[float, tuple[float, float] | None]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(None.__class__))
    )

    alertDelta: float = 0.0
    alertIV: float = 0.0

    # 'levels' map from a bar duration (2 minute, 5 minute, 30 minute, 1 hour, 1 day, 1 week) to the level records holder
    levels: dict[int, LevelBreacher] = field(default_factory=dict)

    # just track an estimate of current direction given short-term EMA crossovers (or maybe log scores too)
    prevDirUp: bool | None = None

    quoteflow: QuoteFlow = field(default_factory=QuoteFlow)

    created: float = field(default_factory=time.time)

    def __post_init__(self):
        assert self.ticker.contract
        self.name = self.state.nameForContract(self.ticker.contract)

        # by default, give all instruments a delta=1 greek designation
        # (if the instrument is a live option, this will be overwritten immediately with live data)
        if not isinstance(self.ticker.contract, (Option, FuturesOption)):
            self.ticker.modelGreeks = OptionComputation(0, 0, 1, 0, 0, 0, 0, 0, 0)

        # init ATRs at our default allowances
        # (the .25 is because fully active quotes update in 250 ms intervals, so we normalize "events per second" by update frequency)
        # (ergo, this is a 90 second vs. 45 second ATR)
        for lookback in (90, 120, 180, 300, 420, 600, 840, 900, 1260, 1800):
            self.atrs[lookback] = ATRLive(
                int(lookback / 0.25), int(lookback / 2 / 0.25)
            )

    def __hash__(self) -> int:
        return hash(self.ticker)

    def __getattr__(self, key):
        """For any non-existing property requested, just fetch from the inner Ticker itself."""
        return getattr(self.ticker, key)

    @property
    def age(self) -> float:
        return time.time() - self.created

    def quote(self) -> QuoteSizes:
        t = self.ticker
        bid = t.bid
        ask = t.ask
        bidSize = t.bidSize
        askSize = t.askSize
        last = t.last
        close = t.close

        # if this is a bag, we can generate some more synthetic details than is provided elsewhere...
        if isinstance(t.contract, Bag):
            if False:
                # We *could* do this, but is the current method of EMA-if-no-VWAP okay enough?
                vwap = 0

                # always set ticker vwap for spread  into ticker object
                for ratio, quote in self.legs:
                    vwap += (quote.vwap or nan) * ratio

                t.vwap = vwap

            # optionally populate bid/ask and size details if current spread isn't getting quotes for some reason
            if bid is None and ask is None:
                bid = 0
                ask = 0
                bidSize = float("inf")
                askSize = float("inf")
                for ratio, quote in self.legs:
                    # if a quote doesn't exist, we need to abandon trying to generate any part of this synthetic quote
                    # because we don't have all the data we need so just combining partial values would be wrong.
                    if not (quote and quote.ask and quote.bid):
                        bid = 0
                        ask = 0
                        vwap = nan
                        bidSize = 0
                        askSize = 0
                        usePrice = None
                        break

                    # SELL legs have opposite signs and positions because they are credits
                    bid += quote.bid * ratio
                    ask += quote.ask * ratio

                    # the "quantity" of a spread is the smallest number available for the combinations
                    bidSize = min(bidSize, quote.askSize)
                    askSize = min(askSize, quote.bidSize)

        # default: return current values
        return QuoteSizes(bid, ask, bidSize, askSize, last, close)

    @property
    def vwap(self) -> float:
        """Override "vwap" paramter option for ITicker allowing us to return our synthetic ema vwap if vwap doesn't exist."""

        if vwap := self.ticker.vwap:
            return vwap

        # return our "daily hours EMA" which should _approximate_ a VWAP for most instruments.
        return self.ema[RTH_EMA_VWAP]

    @property
    def current(self) -> float | None:
        return self.quote().current

    def processTickerUpdate(self) -> None:
        """Update data for this ticker and any dependent tickers when we received new data."""

        q = self.quote()
        current = q.current

        # if no bid/ask/last/close exists, we don't have anything else useful to do here...
        if current is None:
            return

        # log current price update into history...
        # (this also handles updating abandon bag quotes with synthetic quotes from the active legs)
        self.history.append(current)

        for atr in self.atrs.values():
            atr.update(current)

        self.quoteflow.update(self.ticker.bid, self.ticker.ask, self.ticker.timestamp)

        # IBKR spreads only update high/low values when the exact spread is executed, but we can track
        # more detailed high/lows based on current midpoints as they occur (at least until a restart
        # then we begin the process again).
        # Note: the IBKR API can still overwrite these high/low values with active trades when they happen
        #       which may be lower highs or higher lows than the midpoints we've tracked along the way.
        # We could also do this for regular Option and FuturesOption contracts too, but for now we're leaving those as live high/low reports.
        if isinstance(self.ticker.contract, Bag):
            # Don't allow bad negative quotes to pollute the high/low feed counter to the direction of the spread.
            # Logic is: only allow setting positive prices on positive spreads, negative prices on negative spreads, etc
            ema = self.ema[60]

            # also limit these updates to if bid and ask are within 40% of each other to avoid misquoted bouncing on extreme ranges
            # (positive spreads have bid < ask; negative spreads have bid > ask)
            if (
                # current > 0 and ema > 0 and (q.bid and q.ask and q.bid / q.ask >= 0.60)
                current > 0
                and ema > 0
                and q.bid
                and q.ask
                and (abs(q.bid - q.ask) <= 5)
            ) or (
                # current < 0 and ema < 0 and (q.bid and q.ask and q.ask / q.bid >= 0.60)
                current < 0
                and ema < 0
                and q.bid
                and q.ask
                and (abs(q.bid - q.ask) <= 5)
            ):
                # Note: for spreads, we always use only prices we've SEEN SINCE THE QUOTE WAS ADDED and we ignore any
                #       officialy reported "high/low" of the spread (since the "official" high/low data for a spread
                #       is just IBKR reporting if a certain combination traded, so it doesn't represent actual high/low
                #       potential transactions of the legs; and for our own usage, it's easier to see how much a spread
                #       has changed against the highest and lowest value we've seen since we added it).
                # ALSO NOTE: to avoid guessing unseen prices in the middle of a spread, we assume the worst outcome where the low is the highest ask and the high is the smallest bid.
                self.ticker.low = min(self.ticker.low or float("inf"), q.ask)
                self.ticker.high = max(self.ticker.high or float("-inf"), q.bid)

        # if we belong to bags, update the greeks inside the bags
        for bag in self.bags:
            bag.updateGreeks()

        # update EMAs
        # (one problem we have here: when IBKR sometimes refuses to quote spreads, the spread never gets populated in a
        #  ticker update, so the spread will never populate all the metadata (as it has no bid/ask quote and we would
        #  have to generate it synthetically from the legs for additional EMA updating)
        ts = self.ticker.timestamp
        assert ts
        self.ema.update(current, ts)

        name = self.ticker.contract.symbol  # type: ignore
        if isinstance(self.ticker.contract, Future):
            content = ""
            e60 = self.ema[60]
            e300 = self.ema[300]
            diff = current - e300
            d60 = current - e60
            prefix = ""
            if name in {"ES", "RTY"}:
                if diff >= 2 and d60 > 0:
                    if diff > 4:
                        prefix = "MAJOR"
                    content = f"RAPID TICKER UP {name}"
                elif diff <= -2 and d60 < 0:
                    if diff < -4:
                        prefix = "MAJOR"
                    content = f"RAPID TICKER DOWN {name}"
            elif name in {"NQ"}:
                if diff >= 10 and d60 > 0:
                    if diff > 20:
                        prefix = "MAJOR"
                    content = f"RAPID TICKER UP {name}"
                elif diff <= -10 and d60 < 0:
                    if diff < 20:
                        prefix = "MAJOR"
                    content = f"RAPID TICKER DOWN {name}"

            # yes, this is a weird way to use a 'prefix' but it works for information delivery density

            # TODO: fix this logic, it is never triggering
            if content and self.current and self.low:
                atr300 = self.atrs[300].current
                inRangeLow = (
                    "(NEAR LOW ATR)" if (self.current - self.low) <= atr300 else ""
                )
                inRangeHigh = (
                    "(NEAR HIGH ATR)" if (self.high - self.current) <= atr300 else ""
                )
                content = " ".join([content, prefix, inRangeLow, inRangeHigh]).replace(
                    "  ", " "
                )
                self.state.task_create(
                    content,
                    self.state.speak.say(
                        say=content, suppress=60, deadline=time.time() + 5
                    ),
                )

        # check level breaches for alerting
        # compare against ema 30 second to use as the directional bias anchor
        recentHistoryAnchor = round(self.ema[30], 3)
        newer = current
        vw = round(self.vwap, 3)

        # for now, only run VWAP reporting on clients where we have other levels established!
        if self.levels:
            if isinstance(self.ticker.contract, (Future, Index)):
                if recentHistoryAnchor > 1 and vw > 1:
                    # Let's speak VWAP alerts too using local data...
                    if recentHistoryAnchor > vw and newer < vw:
                        # logger.info("down because: {} > {} and {} < {}", recentHistoryAnchor, vw, newer, vw)
                        content = f"{self.name} VW DOWN"
                        self.state.task_create(
                            content,
                            self.state.speak.say(
                                say=content, suppress=60, aux=f" @ {vw:.2f}"
                            ),
                        )
                    elif recentHistoryAnchor < vw and newer > vw:
                        # logger.info("up because: {} < {} and {} > {}", recentHistoryAnchor, vw, newer, vw)
                        content = f"{self.name} VW UP"
                        self.state.task_create(
                            content,
                            self.state.speak.say(
                                say=content, suppress=60, aux=f" @ {vw:.2f}"
                            ),
                        )

        # TODO: also compare against previous daily high and previous daily low
        for duration, breacher in self.levels.items():
            if not breacher.enabled:
                continue

            for level in breacher.levels:
                l = level.level
                if recentHistoryAnchor >= l and newer <= l:
                    # for SMA, we need to say: SMA DURATION SOURCE (e.g. SMA 5 1-day) but we don't want to say "close 1 day 1 day" if duration and lookback are the same
                    addendum = (
                        f", {breacher.durationName}"
                        if breacher.durationName != level.lookbackName
                        else ""
                    )
                    content = f"{self.name} DOWN {level.levelType} {level.lookbackName}{addendum}"
                    self.state.task_create(
                        content, self.state.speak.say(say=content, suppress=60)
                    )
                elif recentHistoryAnchor <= l and newer >= l:
                    addendum = (
                        f", {breacher.durationName}"
                        if breacher.durationName != level.lookbackName
                        else ""
                    )
                    content = f"{self.name} UP {level.levelType} {level.lookbackName}{addendum}"
                    self.state.task_create(
                        content, self.state.speak.say(say=content, suppress=60)
                    )

        self.emaTradeRate.update(self.ticker.tradeRate or 0, ts)
        self.emaVolumeRate.update(self.ticker.volumeRate or 0, ts)

        # update greeks-specific EMAs because why not?
        if g := self.ticker.modelGreeks:
            self.emaIV.update(g.impliedVol, ts)
            self.emaDelta.update(g.delta, ts)
            self.emaVega.update(g.vega, ts)

        # logger.info("[{}] EMAs: {}", self.ticker.contract.localSymbol, self.ema)
        if isinstance(self.ticker.contract, (Future, FuturesOption, Option)):
            name = self.ticker.contract.localSymbol

            # We don't have a clean/concise way of alerting bag names, so avoid for now (bag contracts have no names themselves...)
            if name:
                # fire a special alert if EMA cross is changing directions
                if self.prevDirUp:
                    # if previously up, but now down (fast < slow) , report.
                    # NOTE: we have turned off the DOWN alerts because we typically have equal call and put strikes and there ends up
                    #       being equal reports of DOWN as UP, so we can only report UP (which we tend to care about more anyway).
                    if False and self.ema[60] - self.ema[120] < -0.05:
                        self.prevDirUp = False
                        self.state.task_create(
                            "DOWN",
                            self.state.speak.say(say=f"DOWN {name}", suppress=10),
                        )
                else:
                    if self.ema[60] - self.ema[120] > 0.05:
                        self.prevDirUp = True
                        self.state.task_create(
                            "UP",
                            self.state.speak.say(say=f"UP {name}", suppress=10),
                        )

        # check alert requests for... alerting, I guess.
        baseline = round(self.ema[15], 2)
        prevcompare = 0.0
        if isinstance(self.ticker.contract, (Option, FuturesOption, Bag)):
            # first, check if the delta is sweeping upward
            # (yes, we are only tracking UPWARD motion currently. for downward alerts, just watch an opposite P/C position)
            if mg := self.ticker.modelGreeks:
                curDelta = abs(mg.delta or 0)
                if deltaAlert := self.alertDelta:
                    nextDeltaAlert = curDelta + 0.05
                    if curDelta >= deltaAlert:
                        logger.info(
                            "[{} :: {} :: {:>5}] {:.0f} % from {:>4.02f} -> {:>4.02f} (next: {:>4.02f})",
                            self.ticker.contract.localSymbol,
                            self.name,
                            "DELTA",
                            100 * ((curDelta - deltaAlert) / deltaAlert),
                            deltaAlert,
                            curDelta,
                            nextDeltaAlert,
                        )
                        self.alertDelta = nextDeltaAlert
                else:
                    self.alertDelta = abs(curDelta) + 0.05

            # now alert on the actual premium value being reported by the quote/ticker updates
            for start in [1800]:  # [23400, 3900, 1800, 300]:
                # don't alert if the previous historical value is the same as current
                startval = self.ema[start]
                if startval == prevcompare:
                    continue

                prevcompare = startval

                for r in [0.10]:
                    # TODO: proper digit rounding using instrument db
                    nextalert = round(baseline * (1 + r), 2)
                    if saved := self.alerts[start][r]:
                        (prev, found) = saved
                        if abs(baseline) > abs(found):
                            self.alerts[start][r] = (startval, nextalert)
                            # guard this notice from alerting too much if we add a new quote with unpopluated EMAs
                            if prev > 0.50:
                                logger.info(
                                    "[{} :: {:>5}] {:>5,.0f}% from {:>6.02f}  -> {:>6.02f}  (next: {:>6.02f})",
                                    self.ticker.contract.localSymbol or self.name,
                                    start,
                                    100 * ((baseline - prev) / prev),
                                    prev,
                                    baseline,
                                    nextalert,
                                )
                    else:
                        self.alerts[start][r] = (baseline, round(startval * (1 + r), 2))

    def updateGreeks(self):
        """For bags/spreads, we calculate greeks for the entire spread by combining greeks for each leg.

        Greeks are combined by adding longs and subtracting shorts, all in magnitude of ratio-per-leg.
        """

        final = None
        vwap = 0.0

        # TODO: we should also be checking the "freshness" of the legs quotes so we aren't using stale data.
        #       (e.g. if we remove a quote but still have it as a leg, it isn't getting update anymore...)
        # i.e. composite bag quotes are invalid if all legs don't have live quotes being populated.
        for ratio, leg in self.legs:
            # if any leg is missing, all our values are invalid so do nothing.
            if not leg:
                self.ticker.modelGreeks = None
                self.ticker.vwap = nan
                return

            try:
                if final:
                    final += leg.modelGreeks * ratio
                else:
                    final = leg.modelGreeks * ratio
            except:
                # don't update if any of the leg greeks don't exist.
                self.ticker.modelGreeks = None
                return

        # we set a synthetic modelGreeks object on the bag which normally doesn't exist (but we make it here anyway)
        self.ticker.modelGreeks = final

        # for spread "vwap" we use our live tracked EMA instead of combining VWAP of legs because the VWAP for kinda far OTM
        # legs doesn't reflect underlying price changes over time to because our auto-detected OTM strikes may not have many trades
        # (so: OTM strikes with few trades means the VWAP isn't getting updated over time as price actually moves, but our live tracked
        #      EMA will track the price changes as they float over time (as long as quotes started long ago enough in the past))
        self.ticker.vwap = self.ema[RTH_EMA_VWAP]

        # logger.info("Legs ({}): {}", len(self.legs), pp.pformat(self.legs))
        # logger.info("Updated greeks: {}", final)

    def percentAmtFrom(self, base: float) -> PercentAmount:
        """Return a tuple of current price percent change from 'base' as well as numeric difference between current price and 'base'"""
        c = self.quote().current
        if c and base:
            return (((c - base) / base) * 100, c - base)

        return None, None

    # Note: don't make these properties or else __getattr__ will receive the requests instead of the methods.
    def percentAmtFromHigh(self) -> PercentAmount:
        return self.percentAmtFrom(self.ticker.high)

    def percentAmtFromLow(self) -> PercentAmount:
        return self.percentAmtFrom(self.ticker.low)

    def percentAmtFromOpen(self) -> PercentAmount:
        return self.percentAmtFrom(self.ticker.open)

    def percentAmtFromClose(self) -> PercentAmount:
        return self.percentAmtFrom(self.ticker.close)

    def percentAmtFromVWAP(self) -> PercentAmount:
        # use "real vwap" or, if vwap doesn't exist, use our synthetic 6.5 hour EMA instead.
        return self.percentAmtFrom(self.ticker.vwap or self.ema[RTH_EMA_VWAP])


@dataclass(slots=True)
class IPosition:
    """A representation of a position with order representation capability for accumulation and distribution."""

    # contract for this ENTIRE POSITION
    # i.e. if this is a spread, .contract here is the BAG for the spread, but Contract in the qtycost and updates will be legs.
    #      otherwise, it will be the same contract for all fields if just representing a single instrument.
    contract: Contract

    # quantity and cost per contract as (qty, cost) for ease of iterating
    qtycost: dict[Contract, tuple[float, float]] = field(default_factory=dict)

    # count how many updates we receive per contract.
    # For spreads, this helps us not make decisions if we have an un-equal number of updates (one leg updated, another not yet)
    # to protect against making decisions with not 100% up-to-date field data.
    updates: dict[Contract, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, contract: Contract, qty: float, cost: float):
        """Replace position for contract given quantity and cost.

        Note: for shorts, qty is input as negative (IBKR default behavior),
            but we turn _cost_ negative for our calculations instead.

        Also note: IBKR reports 'cost' as the full dollar cost, but we store cost as per-share cost.
        """

        self.updates[contract] += 1
        self.qtycost[contract] = (qty, cost)

    @property
    def dataIsSafe(self) -> bool:
        """Verify all contracts have an equal update count.

        If all contracts do not have the same update count, we are in the middle
        of an async update cycle and we can't trust the data completely.

        Also, we must have received at least ONE update for every leg in the contract (if this is a bag)
        for the data to be considered safe/complete.
        """

        # data is okay if we don't have any values populated yet
        if not self.updates:
            return True

        # if bag, we must have all contracts populated with at least one position update
        if isinstance(self.contract, Bag):
            values = list(self.updates.values())
            return all([v == values[0] for v in values]) and len(self.updates) == len(
                self.contract.comboLegs
            )

        # else, not a bag, so we only have one update basically
        assert len(self.updates) == 1

        return True

    @property
    def totalSpend(self) -> float | None:
        """Return the total value spent to acquire this position.

        Note: doesn't account for margin holdings (e.g. if you have 20 point wide credit spread you received $4 credit on, you still are at risk of (20 - 4) margin fill-up.
        """
        if self.dataIsSafe:
            pq = 0.0
            for contract, (qty, price) in self.qtycost.items():
                pq += qty * price

            return pq

        return None

    @property
    def averageCost(self) -> float | None:
        """Return average cost of position in per-share prices.

        Note: for spreads, we just add all the legs together (and this also means we store short legs with negative prices)
        """

        # generate average cost as (position qty * position cost per contract) / (total qty)
        # Note: this won't _exactly_ equal your execution price because IBKR will update your cost basis to be reduced by commissions.
        #       e.g. if you shorted a spread for -$17 credit, you may see your average cost is actually -$16.94 after commissions
        if self.dataIsSafe:
            pq = 0.0
            q = float("inf")
            for contract, (qty, price) in self.qtycost.items():
                pq += qty * price / float(contract.multiplier or 1)

                # quantity is the minimum value of any quantity seen
                # (e.g. for spreads, a butterfly of qty 1:2:1 is butterfly qty 1 for buying/selling,
                #       and a vertical spread qty of +70 long and -70 short is a qty 70 spread (not '-70 + 70 == 0')
                q = min(q, abs(qty))

            return pq / q

        return None

    def closePercent(self, percent: float) -> float | None:
        """Return a price above (+) or below (-) the average cost.

        Can be used to generate take profit price with a positive [0, 1+] percent
        or can be used to generate a stop loss price with a negative [-1, 0] percent.

        Note: the price naturally matches positive percents to ALWAYS be profit and negative
              percents to ALWAYS be loss regardless of the long/short position side.
        """
        if self.dataIsSafe:
            ac = self.averageCost
            assert ac

            return ac + (abs(ac) * percent)

        return None

    def closeCash(self, cash: float) -> float | None:
        """Return a price above (+) or below (-) the average cost.

        Can be used to generate take profit price with a positive price growth
        or can be used to generate a stop loss price with a negative price growth.
        """
        if self.dataIsSafe:
            ac = self.averageCost
            assert ac

            tq = self.totalQty
            assert tq

            # we need to project the average cost price back into a full spent cost basis (ac per share * tq * multiplier) == total spend,
            # then we can add the total profit/lost cash price requested, then divide by (quantity * multiplier) to get the limit price
            # yielding the requested profit/loss limit price given the total spend.
            multiplier = float(self.contract.multiplier or 1)
            actualGrowthAdjuster = tq * multiplier

            # our position 'ac' is postive for longs and negative for shorts/credits, so:
            # POSITIVE CASH on LONG  will MAKE  BIGGER POSITIVE PRICE (+, +) == PROFIT
            # NEGATIVE CASH on LONG  will MAKE SMALLER POSITIVE PRICE (+, -) == LOSS
            # POSITIVE CASH on SHORT will MAKE SMALLER NEGATIVE PRICE (-, +) == PROFIT
            # NEGATIVE CASH on SHORT will MAKE  BIGGER NEGATIVE PRICE (-, -) == LOSS
            targetPrice = ((ac * actualGrowthAdjuster) + cash) / actualGrowthAdjuster

            return targetPrice

        return None

    def limitLoss(self, percent: float) -> float | None:
        if self.dataIsSafe:
            ac = self.averageCost
            assert ac

            closeSide = -1 if ac > 0 else 1
            return ac - (ac * percent) * closeSide

        return None

    @property
    def totalQty(self) -> float | None:
        """Return total quantity for this position.

        Note: quantity for spreads is the smallest quantity of any leg.

        Also note: we only report positive quantities so .isLong is needed for a directional quantity check.
        """

        if self.dataIsSafe:
            return (
                min([abs(x[0]) for x in self.qtycost.values()]) if self.qtycost else 0
            )

        return None

    @property
    def closeQty(self) -> float | None:
        """Return quantity for closing (negative if long, positive if short)"""

        if self.dataIsSafe:
            if ac := self.averageCost:
                closeSide = -1 if ac > 0 else 1
                # return negative quantity if currently long, positive quantity if currently short
                return min([abs(x[0]) for x in self.qtycost.values()]) * closeSide

        return None

    @property
    def isLong(self) -> bool | None:
        if ac := self.averageCost:
            return ac > 0

        return None

    @property
    def name(self) -> str:
        """Generate a text name for this contract we can use to discover the same contract again."""
        return nameForContract(self.contract)

    def percentComplete(self, goal: OrderIntent) -> float | None:
        qty = self.totalQty
        amt = self.totalSpend
        avg = self.averageCost

        # if any of the metadata reporters say we aren't ready yet, we can't do any math.
        if any([x is None for x in (qty, amt, avg)]):
            return None

        # if we have no goal, we are 100% complete because there is nothing to do.
        if not goal.qty:
            return 1

        match goal.qty:
            case qtyTarget if isinstance(qtyTarget, DecimalShares):
                assert qty is not None
                return qty / float(qtyTarget)
            case cashTarget if isinstance(cashTarget, DecimalCash):
                assert amt is not None
                assert avg is not None
                # We estimate the impact of adding half of one more quantity to see
                # if acquiring one more is likely to put us over the goal or not.
                return (amt + (avg / 2)) / float(cashTarget)

        return None

    async def accumulate(self, qty: float | str) -> str:
        if self.isLong:
            assert (isinstance(qty, (float, int, Decimal)) and qty > 0) or (
                isinstance(qty, str) and "-" not in qty
            )
        else:
            assert (isinstance(qty, (float, int, Decimal)) and qty < 0) or (
                isinstance(qty, str) and "-" in qty
            )

        cmd = f"buy '{self.name}' {qty} AF"
        return cmd

    async def distribute(self, qty: float | str) -> str:
        if self.isLong:
            assert (isinstance(qty, (float, int, Decimal)) and qty < 0) or (
                isinstance(qty, str) and "-" in qty
            )
        else:
            assert (isinstance(qty, (float, int, Decimal)) and qty > 0) or (
                isinstance(qty, str) and "-" not in qty
            )

        cmd = f"buy '{self.name}' {qty} AF"
        return cmd


# Note: DO NOT use slots=True here because we use @cached_property which doesn't work with slots=True
@dataclass(slots=False)
class Ladder:
    """A description of how to order successively higher or lower prices and quantities for an instrument.

    We also call this a "scale" operation for scaling in and out of positions with partial quantities at
    different prices over time.

    One important feature of our ladder/scale operations is the ability to sumbit them all as connected orders
    then attach a final-level average cost stop-out at the end. This allows us to accumulate at a better cost
    basis during mild volatility, but if volatility exceeds our expectations, we stop our position to prevent
    more damage than we expected from occurring.
    """

    # Details Required:
    #  - N Steps
    #    - A step is a combined (qty + limit price)
    #  - Total stop percentage loss acceptable
    #  - After the final step, we calculate a final stop-limit order to cancel this position if it continues
    #    going against our interests.

    # note: percentages here are 0.xx based
    stopPct: Decimal | None = None
    profitPct: Decimal | None = None

    stopAlgo: str = "STP"
    profitAlgo: str = "LMT"

    steps: tuple[LadderStep, ...] = tuple()

    @classmethod
    def fromOrderIntent(self, oi: OrderIntent) -> Ladder:
        """Convert an OrderIntent with embedded scale ladder into an icli ladder format.

        The main reason we don't use OrderIntent ladder directly is the OrderIntent scale
        is just a list of OrderIntent items at different price, quantity levels with no
        collective profit/stop/average cost embedded.

        We could technically create an OrderIntent group scale object too, but also currently
        the placeOrderForContract() doesn't know about OrderIntent objects directly, so we would
        need to refactor placeOrderForContract() to stop using the custom PriceOrQuantity class
        and instead derive all its values from the OrderIntent object (which _is_ better, we just
        haven't had time to properly refactor it yet, so continuing to add extra custom additions
        on top is cleaner for now).
        """
        steps = tuple([LadderStep(qty=o.qty, limit=o.limit) for o in oi.scale])  # type: ignore

        # 0.xx based percentages
        stopPct: Decimal | None = None
        profitPct: Decimal | None = None

        if isinstance(oi.bracketProfit, DecimalPercent):
            profitPct = oi.bracketProfit / D100
            if oi.isShort:
                profitPct = -profitPct

        if isinstance(oi.bracketLoss, DecimalPercent):
            stopPct = oi.bracketLoss / D100
            if oi.isShort:
                stopPct = -stopPct

        return Ladder(
            profitPct=profitPct,
            stopPct=stopPct,
            steps=steps,
            profitAlgo=ALGOMAP.get(oi.bracketProfitAlgo, None),  # type: ignore
            stopAlgo=ALGOMAP.get(oi.bracketLossAlgo, None),  # type: ignore
        )

    def __bool__(self) -> bool:
        """Ladder only exists if it has order steps inside"""
        return len(self.steps) > 0

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        yield from self.steps

    @cached_property
    def qty(self) -> Decimal:
        return sum([x.qty for x in self.steps])  # type: ignore

    @cached_property
    def avgPrice(self) -> Decimal:
        tpq = Decimal()
        tq = Decimal()
        for x in self.steps:
            tpq += x.qty * x.limit
            tq += x.qty

        return tpq / (tq or 1)

    @cached_property
    def stop(self) -> Decimal | None:
        if not self.stopPct:
            return None

        avg = self.avgPrice

        # calculate average stop loss condition which is LOWER for longs but HIGHER for shorts
        return avg - (avg * self.stopPct)

    @cached_property
    def profit(self) -> Decimal | None:
        if not self.profitPct:
            return None

        avg = self.avgPrice

        # calculate average profit condition which is HIGHER for longs but LOWER for shorts
        return avg + (avg * self.profitPct)


@dataclass
class Q:
    """Self-asking series of prompts."""

    name: str = ""
    msg: str = ""
    choices: Sequence[str | Choice] | None = None
    value: str = field(default_factory=str)

    def __post_init__(self):
        # Allow flexiblity with assigning msg/name if they are just the same
        if not self.msg:
            self.msg = self.name

        if not self.name:
            self.name = self.msg

    def ask(self, **kwargs):
        """Prompt user based on types provided."""
        if self.choices:
            # Note: no kwargs on .select() because .select()
            #       is injecting its own bottom_toolbar for error reporting,
            #       even though it never seems to use it?
            #       See: questionary/prompts/common.py create_inquier_layout()
            return questionary.select(
                message=self.msg,
                choices=self.choices,
                use_indicator=True,
                use_shortcuts=True,
                use_arrow_keys=True,
                use_jk_keys=False,
                # **kwargs,
            ).ask_async()

        return questionary.text(self.msg, default=self.value, **kwargs).ask_async()


@dataclass
class CB:
    """Self-asking series of prompts."""

    name: str = ""
    msg: str = ""
    choices: Sequence[Choice] | None = None

    def __post_init__(self):
        # Allow flexiblity with assigning msg/name if they are just the same
        if not self.msg:
            self.msg = self.name

        if not self.name:
            self.name = self.msg

    def ask(self, **kwargs):
        """Prompt user based on types provided."""
        if self.choices:
            # Note: no kwargs on .select() because .select()
            #       is injecting its own bottom_toolbar for error reporting,
            #       even though it never seems to use it?
            #       See: questionary/prompts/common.py create_inquier_layout()
            return questionary.checkbox(
                message=self.msg,
                choices=self.choices,
                use_jk_keys=False,
                # **kwargs,
            ).ask_async()

        return questionary.text(self.msg, **kwargs).ask_async()

