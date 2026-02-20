"""Bottom toolbar renderer extracted from IBKRCmdlineApp (cli.py).

Generates the prompt_toolkit bottom toolbar HTML, including:
- Account status fields (balance, PnL, margin, etc.)
- Live quote rows for all subscribed symbols
- SPX circuit-breaker levels
- Open orders / positions / executions counters
- Market-close countdown and trading-day counters
"""
from __future__ import annotations

import bisect
import shutil
import statistics
from typing import TYPE_CHECKING

import numpy as np
import whenever
from loguru import logger
from prompt_toolkit.formatted_text import HTML
from ib_async import Bag, Future, FuturesOption, Option, Stock

from icli.engine.contracts import lookupKey
from icli.engine.primitives import QuoteSizes, as_duration, convert_time, nan
from icli.engine.calendar import (
    mkPctColor,
    fetchEndOfMarketDayAtDate,
    tradingDaysRemainingInMonth,
    tradingDaysRemainingInYear,
    sortLeg,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Quote tier classification (mirrors sortQuotes() grouping in calendar.py)
# ---------------------------------------------------------------------------

# Named ETFs that sort alongside futures/indexes in tier 0
_TIER0_ETFS = {
    "SPY", "UPRO", "SPXL", "SOXL", "SOXS",
    "QQQ", "TQQQ", "SQQQ", "IWM", "DIA",
}


def _quote_tier(contract) -> int:
    """Return the display-group tier for a contract (0–3).

    Tiers mirror the first tuple element from sortQuotes() in calendar.py:
      0 = Futures, Indexes, named ETFs, Crypto, Forex
      1 = Equities (stocks, warrants, other ETFs)
      2 = Single-leg options (OPT, FOP, EC)
      3 = Spreads / bags (multi-leg combos)
    """
    st = contract.secType
    if st in {"FUT", "IND", "CONTFUT", "CRYPTO", "CASH"}:
        return 0
    if st == "BAG":
        return 3
    if st in {"OPT", "FOP", "EC"}:
        return 2
    # tier 0 named ETFs: only when symbol == localSymbol (avoids option legs)
    if contract.symbol == contract.localSymbol and contract.symbol in _TIER0_ETFS:
        return 0
    return 1


# ---------------------------------------------------------------------------
# Column header strings for each tier (shown when `set headers true`)
# ---------------------------------------------------------------------------
# These are aligned to match the fixed-width format strings in formatTicker().
# Enable via: set headers true
#
# Equity / Futures / Index columns (tier 0 and 1):
#   Symbol      9 chars    Ticker symbol (localSymbol with spaces stripped)
#   EMA-15s    10 chars    Exponential moving average over ~15 s (60 ticks × 250 ms)
#   (±Now)      8 chars    currentPrice − EMA-15s (positive = price above its short EMA)
#   >                      Trend arrow: EMA-15s vs EMA-75s direction (> rising, < falling, = flat)
#   EMA-75s    10 chars    Exponential moving average over ~75 s (300 ticks × 250 ms)
#   (±Now)      8 chars    currentPrice − EMA-75s
#   Price      10 chars    Current price: bid/ask midpoint (or last trade if `set last true`)
#   ±Sprd       6 chars    Half the bid-ask spread (ask − midpoint)
#   (%Hi ±Hi)  17 chars    % and $ below session high (negative = below high)
#   (%Lo ±Lo)  14 chars    % and $ above session low (positive = above low)
#   (%EOD ±EOD)17 chars    % and $ change from previous session close
#   High       10 chars    Session high
#   Low        10 chars    Session low
#   Bid×Size / Ask×Size    NBBO: bid price × size, ask price × size (purple background)
#   (ATR3m)     7 chars    Average True Range (atrs[180], count-based — see docs/atr-time-fix.md)
#   (%VWAP ±VWAP) 17 chars % and $ from VWAP (IBKR native VWAP when available; otherwise a
#                          synthetic 6.5-hour / 390-minute EMA approximating one RTH session.
#                          IBKR VWAP resets at session open; the EMA fallback decays continuously.)
#   EOD        10 chars    Previous session close price
#   (Age)       9 chars    Time since last quote update
#   @ (Trade)              Time since last trade (shown only when recent)
#
# Single-leg option columns (tier 2):
#   Symbol     22 chars    Option localSymbol (OCC format or decoded FOP)
#   [Undly (ITM Dist%)]    Underlying price, ITM flag, % distance strike-to-underlying
#   [IV]                   Implied volatility
#   [Delta]                Option delta
#   EMA-15s     6 chars    Short-term EMA of option mark
#   > / < / =              Trend direction
#   EMA-75s     6 chars    Longer-term EMA of option mark
#   Mark ±Sprd             Mark price (mid or model) and half-spread
#   (%Hi ±Hi High)         % return from high, $ diff, session high
#   (%Lo ±Lo Low)          % return from low, $ diff, session low
#   (%EOD ±EOD EOD)        % return from close, $ diff, previous close
#   Bid×Size / Ask×Size    Option NBBO
#   ±VWAP                  $ distance from VWAP
#   (Age)                  Time since last quote update
#   (ShortBE @ Dist)       Short break-even price @ distance from underlying
#   (DTE)                  Days (or duration) to expiration
#
# Spread / bag columns (tier 3):
#   Legs                   Multi-line: action, right, ratio, symbol for each leg
#   [IV] [Delta] [Theta] [Vega] [Width]   Combined greeks and spread width
#   EMA-15s > EMA-75s      Short/long EMAs with trend
#   Mark ±Sprd             Mark and half-spread
#   (%Hi ±Hi High)         % return from high, $ diff, high
#   (%Lo ±Lo Low)          % return from low, $ diff, low
#   Bid×Size / Ask×Size    Spread NBBO
#   ±VWAP                  $ from VWAP
#   (Age)                  Time since last quote
#   :: [Quantiles]         Price quintile buckets with [X] marking current position
#   (r) (s) (w)            Range, std dev, width-to-mark ratio

# Build headers using the SAME field widths as formatTicker() so alignment is guaranteed.
# Each label is padded to match the exact format spec of its data column.
# IMPORTANT: parenthesized fields use literal "(" and ")" at boundaries so the "("
# aligns with the "(" in data rows (e.g. data "( -0.12%  -3.25)" → header "( %Hi      ±Hi)").
# Verified: total width = 234 chars, matching a real equity data row.
# fmt: off
_HEADER_EQUITY = " ".join([
    f"{'Symbol':<9}",              #  9  f"{ls:<9}"
    f"{'EMA-15s':>10}",            # 10  f"{e100:>10}"
    f"({'±Now':>6})",              #  8  f"({e100diff:>6})"
    f"{'':>1}",                    #  1  f"{trend}"
    f"{'EMA-75s':>10}",            # 10  f"{e300:>10}"
    f"({'±Now':>6})",              #  8  f"({e300diff:>6})"
    f"{'Price':>10} {'±Sprd':>7}", # 18  f"{usePrice:>10} ±{spread:<6}"
    f"({'%Hi':>7} {'±Hi':>8})",    # 18  f"({pct:>6.2f% + amt:>8})"
    f"({'%Lo':>6} {'±Lo':>6})",    # 15  f"({pct:>5.2f% + amt:>6})"
    f"({'%EOD':>7} {'±EOD':>8})",  # 18  f"({pct:>6.2f% + amt:>8})"
    f"{'High':>10}",               # 10  f"{high:>10}"
    f"{'Low':>10}",                # 10  f"{low:>10}"
    f"{'Bid':>10} x {'Size':>6} {'Ask':>10} x {'Size':>6}",  # 39  bid×size ask×size
    f"({'ATR3m':>5})",             #  7  f"({atr:>5})"
    f"({'%VWAP':>7} {'±VWAP':>8})",  # 18  f"({pct:>6.2f% + amt:>8})"
    f"{'EOD':>10}",                # 10  f"{close:>10}"
    f"({'Age':>7})",               #  9  f"({ago:>7})"
])

# Option header — some fields have variable width (nan vs value), so alignment is best-effort.
_HEADER_OPTION = " ".join([
    f"{'Option Symbol':<22}",              # 22  f"{rowName:<21}:"
    f"[{'Undly':>8} ({'ITM':>1} {'Dist%)]':>8}",  # 25  [u undPrice (itm diff%)]
    f"[iv {'IV]':>5}",                     #  9  [iv X.XX]
    f"[d {'Dlt]':>5}",                     #  9  [d -X.XX]
    f"{'EMA15':>6}",                       #  6  fmtPriceOpt:>6
    f"{'':>1}",                            #  1  trend
    f"{'EMA75':>6}",                       #  6  fmtPriceOpt:>6
    f"{'Mark':>6} ±{'Sprd':>4}",          # 13  mark:>6 ±half:>5
    f"({'%Hi':>7} {'±Hi':>7} {'High':>5})",    # 25  "(pct amt high)"
    f"({'%Lo':>7} {'±Lo':>7} {'Low':>5})",     # 25
    f"({'%EOD':>7} {'±EOD':>7} {'EOD':>5})",   # 25
    f" {'Bid':>6} x {'Size':>6}   {'Ask':>6} x {'Size':>6}",  # 34  " " + bid×size + "   " + ask×size
    f"{'±VWAP':>6}",                       #  6  amtVWAPColor visible chars
    f" ({'Age':>7})",                      # 10  " ({ago:>7})"
    f" ({'ShortBE':>8} @ {'Dist':>5})",    # 22  " (s comp @ diff)"
    f" ({'DTE':>5} {'':>1})",              #  9  " (X.XX d)"
])

# Spread header — spread rows are multi-line so first-line width varies.
_HEADER_SPREAD = " ".join([
    f"{'Spread Legs':<21}",                # ~21  rowName first line
    f"[iv {'IV]':>5}",                     #  9  [iv X.XX]
    f"[d {'Dlt]':>5}",                     #  9  [d -X.XX]
    f"[t {'Tht]':>6}",                     # 10  [t  -X.XX]
    f"[v {'Vga]':>5}",                     #  9  [v X.XX]
    f"[w {'W]':>3}",                       #  6  [w X.X]
    f"{'EMA15':>6}",                       #  6  fmtPriceOpt:>6
    f"{'':>1}",                            #  1  trend
    f"{'EMA75':>6}",                       #  6  fmtPriceOpt:>6
    f" {'Mark':>5} ±{'Sprd':>4}",         # 13  " " mark:>5 ±half:<4
    f" ({'%Hi':>7} {'±Hi':>7} {'High':>5})",   # 26  " (pct amt high)"
    f"({'%Lo':>7} {'±Lo':>7} {'Low':>5})",     # 25
    f" {'Bid':>6} x {'Size':>6}   {'Ask':>6} x {'Size':>6}",  # 34
    f"{'±VWAP':>6}",                       #  6
    f" ({'Age':>7})",                      # 10
])
# fmt: on

_TIER_HEADERS = {
    0: _HEADER_EQUITY,
    1: _HEADER_EQUITY,   # equities use the same column layout as futures/indexes
    2: _HEADER_OPTION,
    3: _HEADER_SPREAD,
}


# ---------------------------------------------------------------------------
# Module-level formatting helpers (no closure state — easily testable)
# ---------------------------------------------------------------------------


def fmtPrice2(n: float) -> str:
    """Format a dollar value for account-status display.

    Returns a 10-character right-aligned string.  Values >= $1 million are
    shown without cents so they fit within the fixed column width.
    """
    # Some prices may not be populated if they haven't
    # happened yet (e.g. PNL values if no trades for the day yet, etc)
    if not n:
        n = 0

    # if GTE $1 million, stop showing cents.
    if n > 999_999.99:
        return f"{n:>10,.0f}"

    return f"{n:>10,.2f}"


def fmtEquitySpread(n, digits: int = 2) -> str:
    """Format a bid/ask spread value for equity quote display."""
    if isinstance(n, (int, float)):
        if n < 1000:
            return f"{n:>6.{digits}f}"

        return f"{n:>6,.0f}"

    return f"{n:>5}"


def fmtPriceOpt(n, digits: int = 2) -> str:
    """Format an option price, substituting nan when the value is falsy."""
    return f"{n or nan:>5,.{digits}f}"


# ---------------------------------------------------------------------------
# ToolbarRenderer
# ---------------------------------------------------------------------------


class ToolbarRenderer:
    """Renders the prompt_toolkit bottom toolbar for IBKRCmdlineApp.

    All application state is accessed through the ``_app`` back-reference so
    this class stays consistent with the pattern used by other extracted
    modules (events.py, quotemanager.py, placement.py, etc.).

    Parameters
    ----------
    app:
        Back-reference to the live IBKRCmdlineApp instance.
    """

    def __init__(self, app) -> None:
        self._app = app

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self):
        """Build and return the bottom-toolbar HTML for prompt_toolkit.

        Called at ``toolbarUpdateInterval`` by prompt_toolkit's refresh loop.
        Returns an ``HTML`` object on success, or ``HTML("No data yet...")``
        when the app has not yet received any market data.
        """
        app = self._app

        app.updates += 1
        app.updatesReconnect += 1
        tz = app.localvars.get("timezone", "US/Eastern")
        app.now = whenever.ZonedDateTime.now(tz)
        app.nowpy = app.now.py_datetime()

        # -- Settings read from app localvars --------------------------------
        useLast = app.localvars.get("last")
        hideSingleLegs = app.localvars.get("hide")
        hideMissing = app.localvars.get("hidemissing")
        showHeaders = app.localvars.get("headers", "").lower() in ("true", "on", "1", "yes")

        # -- Nested formatting helpers (use closure over useLast etc.) -------

        def formatTicker(c):
            ls = lookupKey(c.contract)

            # ibkr API keeps '.close' as the previous full market day close until the next
            # full market day, so for example over the weekend where there isn't a new "full
            # market day," the '.close' is always Thursday's close, while '.last' will be the last
            # traded value seen, equal to Friday's last after-hours trade.
            # But when a new market day starts (but before trading begins), the 'c.last' becomes
            # nan and '.close' becomes the actual expected "previous market day" close we want
            # to use.
            # In summary: '.last' is always the most recent traded price unless it's a new market
            # day before market open, then '.last' is nan and '.close' is the previous most accurate
            # (official) close price, but doesn't count AH trades (we think).
            # Also, this price is assuming the last reported trade is accurate to the current
            # NBBO spread because we aren't checking "if last is outside of NBBO, use NBBO midpoint
            # instead" because these are for rather active equity symbols (we do use the current
            # quote midpoint as price for option pricing though due to faster quote-vs-trade movement)

            # We switched from using "lastPrice" as the shown price to the current midpoint
            # as the shown price because sometimes we were getting price lags when midpoints
            # shifted faster than buying or selling, so we were looking at outdated "prices"
            # for some decisions.
            match c.quote():
                case (
                    QuoteSizes(
                        bid=bid,
                        ask=ask,
                        bidSize=bidSize,
                        askSize=askSize,
                        last=last,
                        close=close,
                    ) as qs
                ):
                    usePrice = qs.last if useLast else qs.current

            high = c.high
            low = c.low
            vwap = c.vwap
            decimals: int | None

            # short circuit the common case of simple quotes
            if isinstance(c.contract, (Stock, Option, Bag)):
                # NOTE: this can potentially miss the case of Bags of FuturesOption having higher precision, but not a priority at the moment.
                decimals = 2
            else:
                try:
                    # NOTE: decimals *can* be zero, so our decimal fetcher returns None on failure to load, so None means "wait for data to populate"
                    if (decimals := app.idb.decimals(c.contract)) is None:
                        return f"WAITING TO POPULATE METADATA FOR: {c.contract.localSymbol}"

                    # for DISPLAY purposes, don't allow one digit decimals (things like /RTY trade in $0.1 increments, but we still want to show $0.10 values)
                    # NOTE: don't make this min(2, decimals) because we _DO_ want to allow 0 decimals, but deny only 1 decimals.
                    if decimals == 1:
                        decimals = 2

                except Exception as e:
                    # logger.exception("WHY?")
                    return f"METADATA LOOKUP FAILED {e}, WAITING TO TRY AGAIN FOR: {c.contract.localSymbol}"

            # assert decimals >= 0, f"Why bad decimals here for {c.contract}?"

            if (bid is None and ask is None) and (usePrice is None):
                name = c.contract.localSymbol
                if isinstance(c.contract, Bag):
                    try:
                        name = " :: ".join(
                            [
                                f"{z.action:<5} {z.ratio:>3} {app.conIdCache.get(z.conId).localSymbol.replace(' ', ''):>20}"
                                for z in c.contract.comboLegs
                            ]
                        )
                    except:
                        # just give it a try. if it doesn't work, no problem.
                        pass

                if hideMissing:
                    return None

                return f"WAITING FOR LIVE MARKET DATA: {name:>12}  ::  {bid=} x {bidSize=}  {ask=} x {askSize=}  {last=} {close=} {usePrice=} {high=} {low=}"

            if usePrice is None:
                if hideMissing:
                    return None

                return f"WAITING FOR DATA UPDATE: {c.contract.localSymbol}"

            if c.lastTimestamp:
                agoLastTrade = as_duration(
                    (app.nowpy - c.lastTimestamp).total_seconds()
                )
            else:
                agoLastTrade = None

            if c.time:
                ago = as_duration((app.nowpy - c.time).total_seconds())
            else:
                ago = "NO LIVE DATA"

                # since no live data, use our synthetic midpoint to update quote history for now
                # (This should only apply to spreads because if a single leg or single stock has no quote data, we have no data at all...)
                # (for some reason, sometimes IBKR just refuses to quote spreads even though all the legs have live offerings)
                # quotekey = lookupKey(c.contract)
                # self.quotehistory[quotekey].append(usePrice)

            percentVWAP, amtVWAP = c.percentAmtFromVWAP()
            percentUnderHigh, amtHigh = c.percentAmtFromHigh()
            percentUpFromLow, amtLow = c.percentAmtFromLow()
            percentUpFromClose, amtClose = c.percentAmtFromClose()

            # If there are > 1,000 point swings, stop displaying cents.
            # also the point differences use the same colors as the percent differences
            # because having fixed point color offsets doesn't make sense (e.g. AAPL moves $2
            # vs DIA moving $200)

            # if bidsize or asksize are > 100,000, just show "100k" instead of breaking
            # the interface for being too wide
            if not bidSize:
                b_s = f"{'X':>6}"
            elif 0 < bidSize < 1:
                # there's a bug here when 'bidSize' is 'inf' and it's triggering here??
                b_s = f"{bidSize:>6.4f}"
            elif bidSize < 100_000:
                b_s = f"{int(bidSize):>6,}"
            else:
                b_s = f"{int(bidSize // 1000):>5,}k"

            if not askSize:
                a_s = f"{'X':>6}"
            elif 0 < askSize < 1:
                a_s = f"{askSize:>6.4f}"
            elif askSize < 100_000 or (askSize != askSize):
                a_s = f"{int(askSize):>6,}"
            else:
                a_s = f"{int(askSize // 1000):>5,}k"

            # use different print logic if this is an option contract or spread
            if isinstance(c.contract, (Option, FuturesOption, Bag)):
                # if c.modelGreeks:
                #     mark = c.modelGreeks.optPrice

                mark = round((bid + ask) / 2, decimals) if bid and ask else 0

                e100 = round(c.ema[60], decimals)
                e300 = round(c.ema[300], decimals)

                # logger.info("[{}] Got EMA for OPT: {} -> {}", ls, e100, e300)
                e100diff = (mark - e100) if e100 else 0

                ediff = e100 - e300
                if ediff > 0:
                    trend = "&gt;"
                elif ediff < 0:
                    trend = "&lt;"
                else:
                    trend = "="

                # For options, instead of using percent difference between
                # prices, we use percent return over the low/close instead.
                # e.g. if low is 0.05 and current is 0.50, we want to report
                #      a 900% multiple, not a 163% difference between the
                #      two numbers as we would report for normal stock price changes.
                # Also note: we use 'mark' here because after hours, IBKR reports
                # the previous day open price as the current price, which clearly
                # isn't correct since it ignores the entire most recent day.
                bighigh = (((mark / high) - 1) * 100) if high else None

                # only report low if current mark estimate is ABOVE the registered
                # low for the day, else we report it as currently trading AT the low
                # for the day instead of potentially BELOW the low for the day.
                biglow = (((mark / low) - 1) * 100) if low else None
                bigclose = (((mark / close) - 1) * 100) if close else None

                emptyFieldA = "       "
                emptyFieldB = "        "
                und = nan
                underlyingStrikeDifference = None
                iv = None
                delta = None
                theta = None
                try:
                    iv = c.modelGreeks.impliedVol
                    delta = c.modelGreeks.delta
                    theta = c.modelGreeks.theta

                    # Note: keep underlyingStrikeDifference the LAST attempt here because if the user doesn't
                    #       have live market data for this option, then 'und' is 0 and this math breaks,
                    #       but if it breaks _last_ then the greeks above still work properly.
                    strike = c.contract.strike
                    und = c.modelGreeks.undPrice
                    underlyingStrikeDifference = -(strike - und) / und * 100
                except:
                    pass

                # Note: we omit OPEN price because IBKR doesn't report it (for some reason?)
                # greeks available as .bidGreeks, .askGreeks, .lastGreeks, .modelGreeks each as an OptionComputation named tuple.
                rowName: str

                # Note: we generate a manual legsAdjust to de-adjust our width measurements if this is
                #       an (assumed) equal width IC spread. (e.g. 2 legs are 1 width, but 4 legs are the width of only *1 pair*)
                legsAdjust: float = 1.0

                # For all combos, we cache the ID to original symbol mapping
                # after the contractId is resolved.
                if c.contract.comboLegs:
                    legsAdjust = len(c.contract.comboLegs) / 2

                    # generate rows to look like:
                    # B  1 AAPL 212121 C 000...
                    # S  2 ....
                    rns = []

                    # for puts, we want legs listed HIGH to LOW
                    # for calls, we want legs listed LOW to HIGH
                    for idx, x in enumerate(
                        sorted(
                            c.contract.comboLegs,
                            # TODO: move this to a leg-sort-lookup cache so it's less work to run every time
                            key=lambda leg: sortLeg(leg, app.conIdCache),
                        )
                    ):
                        try:
                            contract = app.conIdCache[x.conId]
                        except:
                            # cache is broken for this contract id...
                            return f"[CACHE BROKEN FOR {x=} in {c.contract=}]"

                        padding = "    " if idx > 0 else ""
                        action = " "
                        right = " "

                        # localSymbol is either:
                        #  - AAPL  240816C00220000 (contract leg)
                        #  - AAPL (stock leg)

                        # We want to split out some of the details with spaces if it's a full option symbol because
                        # we show spreads as vertically stacked and it makes it easier to pick out which legs are long
                        # vs short vs their strikes and dates as compared to the default OCC showing we use elsewhere.
                        name = contract.localSymbol
                        # TODO: turn this into a top-level @cache function? It just reformats the contract output every time.
                        if isinstance(contract, (Option, Future, FuturesOption)):
                            date = contract.lastTradeDateOrContractMonth
                            right = contract.right or "U"
                            strike = contract.strike
                            action = x.action[0]

                            # TODO: fix padding length between HAS digits (longer) and NOT HAS digits (shorter)
                            #       like if strikes are 225.50 and 230.
                            # Need to: detect decimals for _all_ strikes then do max(legdigits)
                            legstrikedigits = 0 if int(strike) == strike else 2

                            # highlight BUY strikes as bold so we can easily pick them out of spreads
                            # https://python-prompt-toolkit.readthedocs.io/en/stable/pages/advanced_topics/styling.html
                            if strike:
                                if action == "B":
                                    strikeFormatted = f"<aaa bg='ansibrightblue'>{strike:>5,.{legstrikedigits}f}</aaa>"
                                else:
                                    strikeFormatted = f"{strike:>5,.{legstrikedigits}f}"
                            else:
                                strikeFormatted = ""

                            name = (
                                f"{contract.symbol:<4} {date} {right} {strikeFormatted}"
                            )

                        rns.append(f"{padding}{action} {right} {x.ratio:2} {name}")

                    rowName = "\n".join(rns)

                    if False:
                        logger.info(
                            "Contract and vals for combo: {}  -> {} -> {} -> {} -> {}",
                            c.contract,
                            ls,
                            e100,
                            e300,
                            (usePrice, c.bid, c.ask, c.high, c.low),
                        )

                    # show a recent range of prices since spreads have twice (or more) the bid/ask volatility
                    # of a single leg option (due to all the legs being combined into one quote dynamically)
                    src = c.history

                    # typically, a low stddev indicates temporary low volatility which is
                    # the calm before the storm when a big move happens next (in either direction,
                    # but direction prediction can be augmented with moving average crossovers).
                    try:
                        std = statistics.stdev(src)
                    except:
                        std = 0

                    try:
                        parts: list[str | float] = [
                            round(x, 2)
                            for x in statistics.quantiles(
                                src,
                                n=5,
                                method="inclusive",
                            )
                        ]

                        # TODO: benchmark if this double min/max run is faster than looping over it once and just
                        #       checking for min/max in one loop instead of two loops here.
                        minmax = max(src) - min(src)
                    except:
                        # 'statistics' throws an exception if there's not enough data points yet...
                        parts = sorted(src)
                        minmax = 0

                    # add marker where curent price goes in this range...
                    # (sometimes this complains for some reason, but it clears up eventually)
                    try:
                        bpos = bisect.bisect_left(parts, mark)
                        parts.insert(bpos, "[X]")
                    except:
                        logger.info("Failed parts on: {}", parts)
                        pass

                    partsFormatted = ", ".join(
                        [
                            f"{x:>7.2f}" if isinstance(x, (float, int)) else x
                            for x in parts
                        ]
                    )

                    bighigh, amtHigh = c.percentAmtFromHigh()
                    biglow, amtLow = c.percentAmtFromLow()
                    percentCollectiveVWAP, amtCollectiveVwap = c.percentAmtFromVWAP()

                    pctBigHigh, amtBigHigh = (
                        mkPctColor(
                            bighigh,
                            [
                                f"{bighigh:>7.2f}%"
                                if bighigh < 10_000
                                else f"{bighigh:>7,.0f}%",
                                f"{amtHigh:>7.2f}"
                                if amtHigh < 1000
                                else f"{amtHigh:>7,.0f}",
                            ],
                        )
                        if bighigh is not None
                        else (emptyFieldA, emptyFieldB)
                    )
                    pctBigLow, amtBigLow = (
                        mkPctColor(
                            biglow,
                            [
                                f"{biglow:>7.2f}%"
                                if biglow < 10_000
                                else f"{biglow:>7,.0f}%",
                                f"{amtLow:>7.2f}"
                                if amtLow < 1000
                                else f"{amtLow:>7,.0f}",
                            ],
                        )
                        if biglow is not None
                        else (emptyFieldA, emptyFieldB)
                    )
                    _pctBigVWAP, amtBigVWAPColor = (
                        mkPctColor(
                            percentCollectiveVWAP,
                            ["", f"{amtCollectiveVwap:>6.{decimals}f}"],
                        )
                        if percentCollectiveVWAP is not None
                        else (
                            "      ",
                            "      ",
                        )
                    )

                    # Some of the daily values seem to exist for spreads: high and low of day, but previous day close just reports the current price.
                    # this is OPTION BAG/SPREAD ROWS
                    # fmt: off
                    g = c.ticker.modelGreeks
                    if g:
                        collectiveIV = g.impliedVol
                        collectiveDelta = g.delta
                        collectiveTheta = g.theta
                        collectiveVega = g.vega
                        collectiveWidth = c.width
                        wpts = c.width - mark
                        if (
                            collectiveWidth == collectiveWidth
                            and int(collectiveWidth) == collectiveWidth
                        ):
                            widthCents = 0
                        else:
                            widthCents = 2
                    else:
                        collectiveIV = nan
                        collectiveDelta = nan
                        collectiveTheta = nan
                        collectiveVega = nan
                        collectiveWidth = nan
                        widthCents = 0
                        wpts = nan

                    # Spread / bag row — column reference (see also _HEADER_SPREAD):
                    #   Legs | [IV] [Delta] [Theta] [Vega] [Width] | EMA-15s | trend |
                    #   EMA-75s | Mark ±Sprd | (%Hi ±Hi High) | (%Lo ±Lo Low) |
                    #   Bid×Size Ask×Size | ±VWAP | (Age) | :: [Quantiles] (range) (stdev) (w/mark)
                    return " ".join(
                        [
                            rowName,                        # Multi-line leg descriptions (action, right, ratio, symbol)
                            f"[iv {collectiveIV or 0:>5.2f}]",   # Combined implied volatility
                            f"[d {collectiveDelta or 0:>5.2f}]", # Combined delta
                            f"[t {collectiveTheta or 0:>6.2f}]", # Combined theta
                            f"[v {collectiveVega or 0:>5.2f}]",  # Combined vega
                            f"[w {collectiveWidth or 0:>3.{widthCents}f}]",  # Spread width (strike distance)
                            f"{fmtPriceOpt(e100):>6}",     # EMA-15s of spread mark
                            f"{trend}",                     # > / < / =
                            f"{fmtPriceOpt(e300):>6}",     # EMA-75s of spread mark
                            f" {fmtPriceOpt(mark):>5} ±{fmtPriceOpt((ask or nan) - mark, decimals):<4}",  # Mark ±Sprd
                            f" ({pctBigHigh} {amtBigHigh} {fmtPriceOpt(high):>6})" if high else " (                       )",  # %Hi ±Hi High
                            f"({pctBigLow} {amtBigLow} {fmtPriceOpt(low):>6})" if low else "(                       )",      # %Lo ±Lo Low
                            f" {fmtPriceOpt(bid):>6} x {b_s}   {fmtPriceOpt(ask):>6} x {a_s}",  # Bid×Size / Ask×Size
                            f"{amtBigVWAPColor}",           # ±VWAP: $ from VWAP
                            f" ({ago:>7})",                 # Age: time since last quote
                            f"  :: {partsFormatted}  (r {minmax:.2f}) (s {std:.2f}) (w {wpts / legsAdjust:.2f}; {collectiveWidth / (mark or 1) / legsAdjust:,.1f}x)",  # Quantiles, range, stdev, width/mark
                            "HALTED!" if c.halted else "",
                        ]
                    )
                    # fmt: on
                else:
                    if hideSingleLegs:
                        return None

                    if isinstance(c.contract, FuturesOption):
                        strike = c.contract.strike
                        if strike == (istrike := int(c.contract.strike)):
                            strike = istrike

                        fparts = c.contract.localSymbol.split()
                        tradingClass = fparts[0][:-2]
                        month = fparts[0][3:]
                        expiration = fparts[1][1:]
                        rowBody = f"{tradingClass} {month} {expiration} {c.contract.right} {c.contract.lastTradeDateOrContractMonth[2:]} {strike}"
                        rowName = f"{rowBody:<21}:"
                    else:
                        rowName = f"{c.contract.localSymbol:<21}:"

                    # color spreads using our CUSTOM synthetic high/low indicators
                    pctBigHigh, amtBigHigh = (
                        mkPctColor(
                            bighigh,
                            [
                                f"{bighigh:>7.2f}%"
                                if bighigh < 10_000
                                else f"{bighigh:>7,.0f}%",
                                f"{amtHigh:>7.2f}"
                                if amtHigh < 1000
                                else f"{amtHigh:>7,.0f}",
                            ],
                        )
                        if bighigh is not None
                        else (emptyFieldA, emptyFieldB)
                    )
                    pctBigLow, amtBigLow = (
                        mkPctColor(
                            biglow,
                            [
                                f"{biglow:>7.2f}%"
                                if biglow < 10_000
                                else f"{biglow:>7,.0f}%",
                                f"{amtLow:>7.2f}"
                                if amtLow < 1000
                                else f"{amtLow:>7,.0f}",
                            ],
                        )
                        if biglow is not None
                        else (emptyFieldA, emptyFieldB)
                    )
                    pctBigClose, amtBigClose = (
                        mkPctColor(
                            bigclose,
                            [
                                f"{bigclose:>7.2f}%"
                                if bigclose < 1000
                                else f"{bigclose:>7,.0f}%",
                                f"{amtClose:>7.2f}"
                                if amtClose < 10_000
                                else f"{amtClose:>7,.0f}",
                            ],
                        )
                        if bigclose is not None
                        else (emptyFieldA, emptyFieldB)
                    )

                    if isinstance(c.contract, (Option, FuturesOption)):
                        # has data like:
                        # FuturesOption(conId=653770578, symbol='RTY', lastTradeDateOrContractMonth='20231117', strike=1775.0, right='P', multiplier='50', exchange='CME', currency='USD', localSymbol='R3EX3 P1775', tradingClass='R3E')
                        ltdocm = c.contract.lastTradeDateOrContractMonth
                        y = ltdocm[2:4]
                        m = ltdocm[4:6]
                        d = ltdocm[6:8]
                        pc = c.contract.right
                        price = c.contract.strike
                        # sym = rowName

                    # Note: this dynamic calendar math shows the exact time remaining even accounting for (pre-scheduled) early market close days.
                    when = (
                        fetchEndOfMarketDayAtDate(2000 + int(y), int(m), int(d))
                        - app.now
                    ).in_days_of_24h()

                    # this may be too wide for some people? works for me.
                    # just keep shrinking your terminal font size until everything fits?
                    # currently works nicely via:
                    #   - font: Monaco
                    #   - size: 10
                    #   - terminal width: 275+ characters
                    #   - terminal height: 60+ characters

                    # guard the ITM flag because after hours 'underlying price' isn't populated in option quotes
                    itm = ""
                    if delta and und and mark:
                        if delta > 0 and und >= price:
                            # calls
                            itm = "I"
                        elif delta < 0 and und <= price:
                            # puts
                            itm = "I"

                    # "compensated" is acquisiton price for the underlying if you short this strike.
                    # basically (strike - premium) == price of underlying if you get assigned.
                    # (here, "price"  is the "strike price" in the contract)
                    # provide defaults due to async value population from IBKR (and sometimes we don't have underlying price if we don't have market data)
                    compdiff = 0.0
                    try:
                        match pc:
                            case "P":
                                # for puts, we calculate break-even short prices BELOW the the underlying.
                                # first calculate the premium difference from the strike price,
                                compensated = price - mark
                                # then calculate how far from the underlying for the break-even-at-expiry price.
                                # (here, underlying is ABOV E the (strike - premium break-even))
                                compdiff = und - compensated
                            case "C":
                                # for calls, we calculate break-even short prices ABOVE the the underlying.
                                compensated = price + mark
                                # same as above, but for shorting calls, your break-even is above the underlying.
                                # (here, underlying is BELOW the (strike + premium break-even))
                                compdiff = compensated - und
                    except:
                        pass

                    # signal if the current option midpoint is higher or lower than the IBKR theoretical value
                    # (we _can_ do this, but not sure it's really useful to show)
                    # modelDiff = " "
                    # try:
                    #     modelDiff = "+" if round(c.modelGreeks.optPrice, 2) > mark else "-"
                    # except:
                    #     # ignore model not existing when quotes are first added
                    #     pass

                    _pctVWAP, amtVWAPColor = (
                        mkPctColor(
                            percentVWAP,
                            ["", f"{amtVWAP:>6.{decimals}f}"],
                        )
                        if amtVWAP is not None
                        else (
                            "      ",
                            "      ",
                        )
                    )

                    # Single-leg option row — column reference (see also _HEADER_OPTION):
                    #   Symbol | [Undly (ITM Dist%)] | [IV] | [Delta] | EMA-15s | trend |
                    #   EMA-75s | Mark ±Sprd | (%Hi ±Hi High) | (%Lo ±Lo Low) |
                    #   (%EOD ±EOD EOD) | Bid×Size Ask×Size | ±VWAP | (Age) |
                    #   (ShortBE @ Dist) | (DTE)
                    # Note: %Hi/%Lo/%EOD here are multiplicative returns (e.g. 900% from low),
                    # not additive differences as in equity rows.
                    # fmt: off
                    return " ".join(
                        [
                            rowName,                        # Option symbol (OCC format or decoded FOP)
                            f"[u {und or np.nan:>8,.2f} ({itm:<1} {underlyingStrikeDifference or np.nan:>7,.2f}%)]",  # Underlying price, ITM flag, strike-vs-underlying %
                            f"[iv {iv or np.nan:.2f}]",     # Implied volatility
                            f"[d {delta or np.nan:>5.2f}]", # Delta
                            # do we want to show theta or not? Not useful for intra-day trading and we have it in `info` output anyway too.
                            # f"[t {theta or np.nan:>5.2f}]",
                            f"{fmtPriceOpt(e100):>6}",     # EMA-15s of mark
                            f"{trend}",                     # > / < / =
                            f"{fmtPriceOpt(e300):>6}",     # EMA-75s of mark
                            f"{fmtPriceOpt(mark or (c.modelGreeks.optPrice if c.modelGreeks else 0)):>6} ±{fmtPriceOpt((ask or np.nan) - mark, decimals):<4}",  # Mark ±Sprd
                            f"({pctBigHigh} {amtBigHigh} {fmtPriceOpt(high):>6})" if high else "(                       )",  # %Hi ±Hi High
                            f"({pctBigLow} {amtBigLow} {fmtPriceOpt(low):>6})" if low else "(                       )",    # %Lo ±Lo Low
                            f"({pctBigClose} {amtBigClose} {fmtPriceOpt(close):>6})" if close else "(                       )",  # %EOD ±EOD EOD
                            f" {fmtPriceOpt(bid or np.nan):>6} x {b_s}   {fmtPriceOpt(ask or np.nan):>6} x {a_s}",  # Bid×Size / Ask×Size
                            f"{amtVWAPColor}",              # ±VWAP: $ from VWAP
                            f" ({ago:>7})",                 # Age: time since last quote
                            f" (s {compensated:>8,.2f} @ {compdiff:>6,.2f})",  # ShortBE: break-even if short @ distance from underlying
                            f" ({when:>3.2f} d)" if when >= 1 else f" ({as_duration(when * 86400)})",  # DTE: days/duration to expiration
                            "HALTED!" if c.halted else "",
                        ]
                    )
                    # fmt: on

            # TODO: pre-market and after-market hours don't update the high/low values, so these are
            #       not populated during those sessions.
            #       this also means during after-hours session, the high and low are fixed to what they
            #       were during RTH and are no longer valid. Should this have a time check too?
            pctVWAP, amtVWAPColor = (
                mkPctColor(
                    percentVWAP,
                    [
                        f"{percentVWAP:>6.2f}%",
                        f"{amtVWAP:>8.{decimals}f}"
                        if amtVWAP < 1000
                        else f"{amtVWAP:>8.0f}",
                    ],
                )
                if amtVWAP is not None
                else (
                    "       ",
                    "        ",
                )
            )

            pctUndHigh, amtUndHigh = (
                mkPctColor(
                    percentUnderHigh,
                    [
                        f"{percentUnderHigh:>6.2f}%",
                        f"{amtHigh:>8.{decimals}f}"
                        if amtHigh < 1000
                        else f"{amtHigh:>8.0f}",
                    ],
                )
                if amtHigh is not None
                else ("       ", "        ")
            )

            pctUpLow, amtUpLow = (
                mkPctColor(
                    percentUpFromLow,
                    [
                        f"{percentUpFromLow:>5.2f}%",
                        f"{amtLow:>6.{decimals}f}"
                        if amtLow < 1000
                        else f"{amtLow:>6.0f}",
                    ],
                )
                if amtLow is not None
                else ("      ", "      ")
            )

            # high and low are only populated after regular market hours, so allow nan to show the
            # full float value during pre-market hours.
            pctUpClose, amtUpClose = (
                mkPctColor(
                    percentUpFromClose,
                    [
                        f"{percentUpFromClose:>6.2f}%",
                        f"{amtClose:>8.{decimals}f}"
                        if amtClose < 1000
                        else f"{amtClose:>8.0f}",
                    ],
                )
                if amtClose is not None
                else ("      ", "         ")
            )

            # ATR3m: Average True Range. Note: count-based, not time-based (see docs/atr-time-fix.md).
            # atrs[180] = 720-sample buffer, 360-sample decay. "3 min" assumes 4 Hz ticks.
            atrval = c.atrs[180].atr.current

            # if ATR > 100, omit cents so it fits in the narrow column easier
            if atrval > 100:
                atr = f"{atrval:>5.0f}"
            else:
                # else, we can print a full width value since it will fit in the 5 character width column
                atr = f"{atrval:>5.2f}"

            e100 = round(c.ema[60], decimals)
            e300 = round(c.ema[300], decimals)

            # for price differences we show the difference as if holding a LONG position
            # at the historical price as compared against the current price.
            # (so, if e100 is $50 but current price is $55, our difference is +5 because
            #      we'd have a +5 profit if held from the historical price.
            #      This helps align "price think" instead of showing difference from historical
            #      vs. current where "smaller historical vs. larger current" would cause negative
            #      difference which is actually a profit if it were LONG'd in the past)
            # also don't show differences for TICK because it's not really a useful number (and it's too big breaking formatting)
            if ls == "TICK-NYSE":
                e100diff = 0
                e300diff = 0
            else:
                e100diff = (usePrice - e100) if e100 else 0
                e300diff = (usePrice - e300) if e300 else 0
            # logger.info("[{}] e100 e300: {} {} {} {}", ls, e100, e300, e100diff, e300diff)

            # also add a marker for if the short term trend (1m) is GT, LT, or EQ to the longer term trend (3m)
            ediff = e100 - e300
            if ediff > 0:
                trend = "&gt;"
            elif ediff < 0:
                trend = "&lt;"
            else:
                trend = "="

            # Equity / Futures / Index row — column reference (see also _HEADER_EQUITY):
            #   Symbol | EMA-15s | (±Now) | trend | EMA-75s | (±Now) | Price ±Sprd |
            #   (%Hi ±Hi) | (%Lo ±Lo) | (%EOD ±EOD) | High | Low |
            #   Bid×Size Ask×Size | (ATR3m) | (%VWAP ±VWAP) | EOD | (Age) | @ (Trade)
            # fmt: off
            return " ".join(
                [
                    f"{ls:<9}",                             # Symbol
                    f"{e100:>10,.{decimals}f}",             # EMA-15s (60 ticks × 250 ms)
                    f"({e100diff:>6,.2f})" if e100diff else "(      )",  # ±Now: price − EMA-15s
                    f"{trend}",                             # > / < / = (EMA-15s vs EMA-75s)
                    f"{e300:>10,.{decimals}f}",             # EMA-75s (300 ticks × 250 ms)
                    f"({e300diff:>6,.2f})" if e300diff else "(      )",  # ±Now: price − EMA-75s
                    f"{usePrice:>10,.{decimals}f} ±{fmtEquitySpread(ask - usePrice, decimals) if (ask and ask >= usePrice) else '':<6}",  # Price ±Sprd (half-spread)
                    f"({pctUndHigh} {amtUndHigh})",         # %Hi ±Hi: % and $ below session high
                    f"({pctUpLow} {amtUpLow})",             # %Lo ±Lo: % and $ above session low
                    f"({pctUpClose} {amtUpClose})",         # %EOD ±EOD: % and $ from previous close
                    f"{high or np.nan:>10,.{decimals}f}",   # Session high
                    f"{low or np.nan:>10,.{decimals}f}",    # Session low
                    f"<aaa bg='purple'>{c.bid or np.nan:>10,.{decimals}f} x {b_s} {ask or np.nan:>10,.{decimals}f} x {a_s}</aaa>",  # Bid×Size / Ask×Size (NBBO)
                    f"({atr})",                             # ATR3m: 3-minute ATR (atrs[180], 720-tick buf, 360-tick decay)
                    f"({pctVWAP} {amtVWAPColor})",          # %VWAP ±VWAP: % and $ from VWAP (see ITicker.vwap for source)
                    f"{close or np.nan:>10,.{decimals}f}",  # EOD: previous session close
                    f"({ago:>7})",                          # Age: time since last quote update
                    # Only show "last trade ago" if it is recent enough
                    f"@ ({agoLastTrade})" if agoLastTrade else "",  # Trade: time since last trade
                    "     HALTED!" if c.halted else "",
                ]
            )
            # fmt: on

        try:
            rowlen, _ = shutil.get_terminal_size()

            rowvals: list[list[str]] = [[]]
            currentrowlen = 0
            DT = []
            for cat, val in app.accountStatus.items():
                # if val == 0:
                #    continue

                # Note: if your NLV is >= $25,000 USD, then the entire
                #       DayTradesRemaining{,T+{1,2,3,4}} sections do not
                #       show up in self.accountStatus anymore.
                #       This also means if you are on the border of $25k ± 0.01,
                #       the field will keep vanishing and showing up as your
                #       account values bounces above and below the PDT threshold
                if cat.startswith("DayTrades"):
                    # the only field we treat as just an integer

                    # skip field if is -1, meaning account is > $25k so
                    # there is no day trade restriction
                    if val == -1:
                        continue

                    DT.append(int(val))

                    # wait until we accumulate all 5 day trade indicators
                    # before printing the day trades remaining count...
                    if len(DT) < 5:
                        continue

                    section = "DayTradesRemaining"
                    # If ALL future day trade values are equal, only print the
                    # single value.
                    if all(x == DT[0] for x in DT):
                        value = f"{section:<20} {DT[0]:>14}"
                    else:
                        # else, there is future day trade divergence,
                        # so print all the days.
                        csv = ", ".join([str(x) for x in DT])
                        value = f"{section:<20} ({csv:>14})"
                else:
                    # else, use our nice formatting
                    # using length 14 to support values up to 999,999,999.99
                    value = f"{cat:<20} {fmtPrice2(val):>14}"

                vlen = len(value)
                # "+ 4" because of the "    " in the row entry join
                # ALSO, limit each row to 7 elements MAX so we always have the same status block
                # alignment regardless of console width (well, if consoles are wide enough for six or seven columns
                # at least; if your terminal is smaller than six status columns the entire UI is probably truncated anyway).
                totLen = currentrowlen + vlen + 4
                if (totLen < rowlen) and (totLen < 271):
                    # append to current row
                    rowvals[-1].append(value)
                    currentrowlen += vlen + 4
                else:
                    # add new row, reset row length
                    rowvals.append([value])
                    currentrowlen = vlen

            balrows = "\n".join(["    ".join(x) for x in rowvals])

            # RegT overnight margin means your current margin balance must be less than your SMA value.
            # Your SMA account increases with deposits and when your positions grow profit, so the minimum
            # overnight you can hold is 50% of your deposited cash, while the maximum you can hold is your
            # 4x margin if your SMA has grown larger than your total BuyingPower.
            # After you trade for a while without withdraws, your profits will grow your SMA value to be larger
            # than your full 4x BuyingPower, so eventually you can hold 4x margin overnight with no liquidations.
            # (note: the SMA margin calculations are only for RegT and do not apply to portfolio margin / SPAN accounts)
            overnightDeficit = app.accountStatus["SMA"]

            onc = ""
            if overnightDeficit < 0:
                # You must restore your SMA balance to be positive before:
                # > Whenever you have a position change on a trading day,
                # > we check the balance of your SMA at the end of the US trading day (15:50-17:20 ET),
                # > to ensure that it is greater than or equal to zero.
                onc = f" (OVERNIGHT REG-T MARGIN CALL: ${-overnightDeficit:,.2f})"

            # some positions have less day margin than overnight margin, and we can see the difference
            # where 'FullMaintMarginReq' is what is required after RTH closes and 'MaintMarginReq' is required for the current session.
            # Just add a visible note if our margin requirements will increase if we don't close out live positions.
            fmm = app.accountStatus.get("FullMaintMarginReq", 0)
            mm = app.accountStatus["MaintMarginReq"]

            if fmm > mm:
                onc += f" (OVERNIGHT MARGIN LARGER THAN DAY: ${fmm:,.2f} (+${fmm - mm:,.2f}))"

            qs = app.quoteStateSorted

            spxbreakers = ""

            try:
                spx = app.quoteState.get("SPX")
                if spx:
                    # hack around IBKR quotes being broken over weekends/holdays
                    # NOTE: this isn't valid across weekends because until Monday morning, the "close" is "Thursday close" not frday close. sigh.
                    #       also the SPX symbol never has '.open' value so we can't detect "stale vs. current quote from last close"
                    spxl = spx.last
                    spxc = spx.close

                    def undX(spxd, spxIn):
                        return (spxd / spxIn) * 100

                    spxc7 = round(spxc / 1.07, 2)
                    spxcd7 = round(spxl - spxc7, 2)

                    spxc13 = round(spxc / 1.13, 2)
                    spxcd13 = round(spxl - spxc13, 2)

                    spxc20 = round(spxc / 1.20, 2)
                    spxcd20 = round(spxl - spxc20, 2)

                    spxbreakers = "   ".join(
                        [
                            f"7%: {spxc7:,.2f} ({spxcd7:,.2f}; {undX(spxcd7, spxc7):.2f}%)",
                            f"13%: {spxc13:,.2f} ({spxcd13:,.2f}; {undX(spxcd13, spxc13):.2f}%)",
                            f"20%: {spxc20:,.2f} ({spxcd20:,.2f}; {undX(spxcd20, spxc20):.2f}%)",
                        ]
                    )
            except:
                # the data will populate eventually
                # logger.exception("cant update spx?")
                pass

            # TODO: we may want to iterate these to exclude "Inactive" or orders like:
            # [x.log[-1].status == "Inactive" for x in self.ib.openTrades()]
            # We could also exclude waiting bracket orders when status == 'PreSubmitted' _and_ has parentId
            ordcount = len(app.ib.openTrades())
            openorders = f"open orders: {ordcount:,}"

            positioncount = len(app.ib.portfolio())
            openpositions = f"positions: {positioncount:,}"

            executioncount = len(app.ib.fills())
            todayexecutions = f"executions: {executioncount:,}"

            # TODO: We couold also flip this between a "time until market open" vs "time until close" value depending
            #       on if we are out of market hours or not, but we aren't bothering with the extra logic for now.
            untilClose = (
                fetchEndOfMarketDayAtDate(app.now.year, app.now.month, app.now.day)
                - app.now
            )
            todayclose = f"mktclose: {as_duration(untilClose.in_seconds())}"
            daysInMonth = f"dim: {tradingDaysRemainingInMonth()}"
            daysInYear = f"diy: {tradingDaysRemainingInYear()}"

            # this weird thing lets is optionally remove tickers by letting formatTicker() return None, then we drop None results from showing.
            rows = []

            # this extra processing loop for format inclusion lets us _optionally hide_ ticker rows
            # from appearing in the toolbar numbered list.
            # If you enable icli env var 'hide', currently all single-leg option rows get removed from
            # printing (if you are trading speads only and single legs are taking up most of the screen, this
            # helps save your screen space a bit).
            # We could extend this "show/hide" system to different categories or symbols in the future.
            altrowColor = app.altrowColor
            # safety: only use color if it looks like a valid #hex to avoid
            # an infinite prompt_toolkit renderer crash loop
            if altrowColor and not (altrowColor.startswith("#") and len(altrowColor) in (4, 7)):
                altrowColor = ""

            # Track tier transitions so we can inject group header rows
            # when `set headers true` is enabled (beginner-friendly column labels).
            prevTier = None

            for qp, (sym, quote) in enumerate(qs):
                # Optionally inject a column-description header before each new group
                if showHeaders:
                    tier = _quote_tier(quote.contract)
                    if tier != prevTier:
                        hdr = _TIER_HEADERS.get(tier, "")
                        # Pad with spaces so the background extends to the terminal edge.
                        # In prompt_toolkit's reverse-styled toolbar: fg→bg, bg→text color.
                        rows.append(f"<aaa fg='#333333' bg='#aaaaaa'>    {hdr}{' ' * rowlen}</aaa>")
                        prevTier = tier

                if niceticker := formatTicker(quote):
                    row = f"{qp:>2}) " + niceticker
                    if altrowColor and qp % 2 == 1:
                        # The toolbar uses prompt_toolkit's default 'reverse'
                        # style, which swaps fg↔bg. So to change the rendered
                        # *background* of a row, we must set fg (not bg).
                        # Pad with spaces so the stripe extends to terminal edge.
                        row = f"<zebra fg='{altrowColor}'>{row}{' ' * rowlen}</zebra>"
                    rows.append(row)

            # basically, if we've never reconnected, then only show one update count
            if app.updates == app.updatesReconnect:
                updatesFmt = f"[{app.updates:,}]"
            else:
                # else, the total CLI refresh count has diverged from the same-session reconnect count, so show both.
                # (why is this useful? our internal _data_ resets on a reconnect, so all our client-side moving averages, etc, go back
                #  to baseline with no history after a reconnect (because we don't know how long we were disconnected for technically),
                #  so having a "double count" can help show users to wait a little longer for the client-side derived metrics to catch up again).
                updatesFmt = f"[{app.updates:,}; {app.updatesReconnect:,}]"

            return HTML(
                # all these spaces look weird, but they (kinda) match the underlying column-based formatting offsets
                f"""[{app.clientId}] {app.nowpy.strftime('%Y-%m-%d %H:%M:%S %Z'):<28}{onc} {updatesFmt}          {spxbreakers}          {openorders}    {openpositions}    {todayexecutions}      {todayclose}   ({daysInMonth} :: {daysInYear})\n"""
                + "\n".join(rows)
                + "\n"
                + balrows
            )
        except:
            logger.exception("qua?")
            return HTML("No data yet...")  # f"""{self.now:<40}\n""")
