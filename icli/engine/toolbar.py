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
        app.now = whenever.ZonedDateTime.now("US/Eastern")
        app.nowpy = app.now.py_datetime()

        # -- Settings read from app localvars --------------------------------
        useLast = app.localvars.get("last")
        hideSingleLegs = app.localvars.get("hide")
        hideMissing = app.localvars.get("hidemissing")

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

                    return " ".join(
                        [
                            rowName,
                            f"[iv {collectiveIV or 0:>5.2f}]",
                            f"[d {collectiveDelta or 0:>5.2f}]",
                            f"[t {collectiveTheta or 0:>6.2f}]",
                            f"[v {collectiveVega or 0:>5.2f}]",
                            f"[w {collectiveWidth or 0:>3.{widthCents}f}]",
                            f"{fmtPriceOpt(e100):>6}",
                            f"{trend}",
                            f"{fmtPriceOpt(e300):>6}",
                            f" {fmtPriceOpt(mark):>5} ±{fmtPriceOpt((ask or nan) - mark, decimals):<4}",
                            f" ({pctBigHigh} {amtBigHigh} {fmtPriceOpt(high):>6})" if high else " (                       )",
                            f"({pctBigLow} {amtBigLow} {fmtPriceOpt(low):>6})" if low else "(                       )",
                            f" {fmtPriceOpt(bid):>6} x {b_s}   {fmtPriceOpt(ask):>6} x {a_s}",
                            f"{amtBigVWAPColor}",
                            f" ({ago:>7})",
                            f"  :: {partsFormatted}  (r {minmax:.2f}) (s {std:.2f}) (w {wpts / legsAdjust:.2f}; {collectiveWidth / (mark or 1) / legsAdjust:,.1f}x)",
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

                    # this is SINGLE LEG OPTION ROWS
                    # fmt: off
                    return " ".join(
                        [
                            rowName,
                            f"[u {und or np.nan:>8,.2f} ({itm:<1} {underlyingStrikeDifference or np.nan:>7,.2f}%)]",
                            f"[iv {iv or np.nan:.2f}]",
                            f"[d {delta or np.nan:>5.2f}]",
                            # do we want to show theta or not? Not useful for intra-day trading and we have it in `info` output anyway too.
                            # f"[t {theta or np.nan:>5.2f}]",
                            f"{fmtPriceOpt(e100):>6}",
                            f"{trend}",
                            f"{fmtPriceOpt(e300):>6}",
                            f"{fmtPriceOpt(mark or (c.modelGreeks.optPrice if c.modelGreeks else 0)):>6} ±{fmtPriceOpt((ask or np.nan) - mark, decimals):<4}",
                            f"({pctBigHigh} {amtBigHigh} {fmtPriceOpt(high):>6})" if high else "(                       )",
                            f"({pctBigLow} {amtBigLow} {fmtPriceOpt(low):>6})" if low else "(                       )",
                            f"({pctBigClose} {amtBigClose} {fmtPriceOpt(close):>6})" if close else "(                       )",
                            f" {fmtPriceOpt(bid or np.nan):>6} x {b_s}   {fmtPriceOpt(ask or np.nan):>6} x {a_s}",
                            f"{amtVWAPColor}",
                            f" ({ago:>7})",
                            f" (s {compensated:>8,.2f} @ {compdiff:>6,.2f})",
                            f" ({when:>3.2f} d)" if when >= 1 else f" ({as_duration(when * 86400)})",
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

            # somewhat circuitous logic to format NaNs and values properly at the same string padding offsets
            # Showing the 3 minute ATR by default. We have other ATRs to choose from. See per-symbol 'info' output for all live values.
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

            # fmt: off
            return " ".join(
                [
                    f"{ls:<9}",
                    f"{e100:>10,.{decimals}f}",
                    f"({e100diff:>6,.2f})" if e100diff else "(      )",
                    f"{trend}",
                    f"{e300:>10,.{decimals}f}",
                    f"({e300diff:>6,.2f})" if e300diff else "(      )",
                    f"{usePrice:>10,.{decimals}f} ±{fmtEquitySpread(ask - usePrice, decimals) if (ask and ask >= usePrice) else '':<6}",
                    f"({pctUndHigh} {amtUndHigh})",
                    f"({pctUpLow} {amtUpLow})",
                    f"({pctUpClose} {amtUpClose})",
                    f"{high or np.nan:>10,.{decimals}f}",
                    f"{low or np.nan:>10,.{decimals}f}",
                    f"<aaa bg='purple'>{c.bid or np.nan:>10,.{decimals}f} x {b_s} {ask or np.nan:>10,.{decimals}f} x {a_s}</aaa>",
                    f"({atr})",
                    f"({pctVWAP} {amtVWAPColor})",
                    f"{close or np.nan:>10,.{decimals}f}",
                    f"({ago:>7})",
                    # Only show "last trade ago" if it is recent enough
                    f"@ ({agoLastTrade})" if agoLastTrade else "",
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
            todayclose = f"mktclose: {convert_time(untilClose.in_seconds())}"
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
            for qp, (sym, quote) in enumerate(qs):
                if niceticker := formatTicker(quote):
                    row = f"{qp:>2}) " + niceticker
                    if qp % 2 == 1:
                        # The toolbar uses prompt_toolkit's default 'reverse'
                        # style, which swaps fg↔bg. So to change the rendered
                        # *background* of a row, we must set fg (not bg).
                        # Pad with spaces so the stripe extends to terminal edge.
                        row = f"<zebra fg='#c0c0c0'>{row}{' ' * rowlen}</zebra>"
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
                f"""[{app.clientId}] {str(app.now):<44}{onc} {updatesFmt}          {spxbreakers}          {openorders}    {openpositions}    {todayexecutions}      {todayclose}   ({daysInMonth} :: {daysInYear})\n"""
                + "\n".join(rows)
                + "\n"
                + balrows
            )
        except:
            logger.exception("qua?")
            return HTML("No data yet...")  # f"""{self.now:<40}\n""")
