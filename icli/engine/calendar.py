"""Standalone functions extracted from cli.py — calendar, color, sorting utilities.

These functions have no dependency on IBKRCmdlineApp state.
"""
from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from typing import Final

import bs4
import pandas as pd
import pytz
import seaborn  # type: ignore
import whenever
from cachetools import TTLCache, cached
from pandas.tseries.offsets import MonthEnd, YearEnd

import tradeapis.cal as mcal
from ib_async import Contract

from icli.engine.primitives import nan

# increase calendar cache duration since we provide exact inputs each time,
# so we know the cache doesn't need to self-invalidate to update new values.
mcal.CALENDAR_CACHE_SECONDS = 60 * 60 * 60

warnings.filterwarnings("ignore", category=bs4.MarkupResemblesLocatorWarning)

# setup color gradients we use to show gain/loss of daily quotes
COLOR_COUNT = 100

# palette 'RdYlGn' is a spectrum from low RED to high GREEN which matches
# the colors we want for low/negative (red) to high/positive (green)
MONEY_COLORS = seaborn.color_palette("RdYlGn", n_colors=COLOR_COUNT, desat=1).as_hex()

# only keep lowest 25 and highest 25 elements since middle values are less distinct
MONEY_COLORS = MONEY_COLORS[:25] + MONEY_COLORS[-25:]

# display order we want: RTY / RUT, ES / SPX, NQ / COMP, YM, Index ETFs
FUT_ORD = dict(
    MES=-9,
    ES=-9,
    SPY=-6,
    SPX=-9,
    NANOS=-9,
    RTY=-10,
    RUT=-10,
    M2K=-10,
    IWM=-6,
    NDX=-8,
    COMP=-8,
    NQ=-8,
    QQQ=-6,
    MNQ=-8,
    MYM=-7,
    YM=-7,
    DJI=-7,
    DIA=-6,
)

# A-Z, Z-A, translate between them (lowercase only)
ATOZ = "".join([chr(x) for x in range(ord("a"), ord("z") + 1)])
ZTOA = ATOZ[::-1]
ATOZTOA_TABLE = str.maketrans(ATOZ, ZTOA)


def mkcolor(
    n: float, vals: str | list[str], colorRanges: Sequence[float]
) -> str | list[str]:
    def colorRange(x):
        buckets = len(MONEY_COLORS) // len(colorRanges)
        for idx, crLow in enumerate(colorRanges):
            if x <= crLow:
                return MONEY_COLORS[idx * buckets]

        # else, on the high end of the range, so use highest color
        return MONEY_COLORS[-1]

    # no style if no value (or if nan%)
    if n == 0 or n != n:
        return vals

    # override for high values
    if n >= 0.98:
        useColor = "ansibrightblue"
    else:
        useColor = colorRange(n)

    if isinstance(vals, list):
        return [f"<aaa bg='{useColor}'>{v}</aaa>" for v in vals]

    # else, single thing we can print
    return f"<aaa bg='{useColor}'>{vals}</aaa>"


def mkPctColor(a, b):
    # fmt: off
    colorRanges = [-0.98, -0.61, -0.33, -0.13, 0, 0.13, 0.33, 0.61, 0.98]
    # fmt: on
    return mkcolor(a, b, colorRanges)


@cached(cache={}, key=lambda x, _y: x.conId)
def sortLeg(leg, conIdCache):
    try:
        return (
            conIdCache[leg.conId].right,
            leg.action,
            -conIdCache[leg.conId].strike
            if conIdCache[leg.conId].right == "P"
            else conIdCache[leg.conId].strike,
        )
    except:
        # if cache is broken, just ignore the sort instead of crashing
        return ("Z", leg.action, 0)


# Note: only cache based on the 'x' argument and not the 'contractCache' argument.
@cached(cache={}, key=lambda x, _y: hash(x))
def sortQuotes(x, contractCache: dict[int, Contract] | None = None):
    """Comparison function to sort quotes by specific types we want grouped together."""
    sym, quote = x
    c = quote.contract

    # We want to sort futures first, and sort MES, MNQ, etc first.
    # (also Indexes and Index ETFs first too)
    # This double symbol check is so we don't accidentially sort market ETF options
    # inside the regular equity section.
    if c.secType in {"FUT", "IND", "CONTFUT"} or (
        (c.symbol == c.localSymbol)
        and (
            c.symbol
            in {
                "SPY",
                "UPRO",
                "SPXL",
                "SOXL",
                "SOXS",
                "QQQ",
                "TQQQ",
                "SQQQ",
                "IWM",
                "DIA",
            }
        )
    ):
        priority = FUT_ORD[c.symbol] if c.symbol in FUT_ORD else 0
        return (0, priority, c.secType, c.symbol, c.localSymbol)

    # draw crypto and forex/cash quotes under futures quotes
    if c.secType in {"CRYPTO", "CASH"}:
        priority = 0
        return (0, priority, c.secType, c.symbol, c.localSymbol)

    if c.secType == "OPT":
        # options are medium last because they are wide
        priority = 0
        return (2, priority, c.secType, c.localSymbol, c.symbol)

    if c.secType in {"FOP", "EC"}:
        # future options (and "Event Contracts") are above other options...
        priority = -1

        # Future Options have local symbols like "E4AQ4 C5700" where the full date isn't
        # embedded (just a "Date code" which isn't "sequential in time"), so let's prepend
        # the actual date for sorting these against each other...
        return (
            2,
            priority,
            c.secType,
            c.lastTradeDateOrContractMonth + c.localSymbol,
            c.symbol,
        )

    if c.secType == "BAG":
        # bags are last because their descriptions take up multiple rows
        priority = 0

        # look up PROPERTIES of a bag so we can sort by actual details better...
        if contractCache:
            bagParts = []
            for x in c.comboLegs:
                leg = contractCache.get(x.conId)

                if not leg:
                    break

                bagParts.append(
                    (
                        leg.symbol,
                        leg.lastTradeDateOrContractMonth,
                        x.action[0],
                        leg.right,
                        leg.strike,
                    )
                )

            bagParts = list(sorted(bagParts))

            # logger.info("Bag key: {}", bagKey)
            return (3, priority, c.secType, bagParts, c.symbol)

        return (
            3,
            priority,
            c.secType,
            ":".join(sorted([f"{x.action}-{x.conId}" for x in c.comboLegs])),
            "",
        )

    # else, just by name.
    # BUT we do these in REVERSE order since they
    # are at the end of the table!
    # (We create "reverse order" by translating all
    #  letters into their "inverse" where a == z, b == y, etc).
    priority = 0
    return (1, priority, c.secType, invertstr((c.localSymbol or c.symbol).lower()))


def invertstr(x):
    return x.translate(ATOZTOA_TABLE)


# allow these values to be cached for 10 hours
@cached(cache=TTLCache(maxsize=200, ttl=60 * 90))
def marketCalendar(start, stop):
    return mcal.getMarketCalendar(
        "NASDAQ",
        start=start,
        stop=stop,
    )


# allow these values to be cached for 10 hours
@cached(cache=TTLCache(maxsize=300, ttl=60 * 60 * 10))
def fetchDateTimeOfEndOfMarketDayAtDate(y, m, d):
    """Return the market (start, end) timestamps for the next two market end times."""
    start = pd.Timestamp(y, m, d, tz="US/Eastern")  # type: ignore
    found = marketCalendar(start, start + pd.Timedelta(7, "D"))

    # format returned is two columns of [MARKET OPEN, MARKET CLOSE] timestamps per date.
    soonestStart = found.iat[0, 0]
    soonestEnd = found.iat[0, 1]

    nextStart = found.iat[1, 0]
    nextEnd = found.iat[1, 1]

    return [(soonestStart, soonestEnd), (nextStart, nextEnd)]


def goodCalendarDate():
    """Return the start calendar date we should use for market date lookups.

    Basically, use TODAY if the current time is before liquid hours market close, else use TOMORROW."""
    now = pd.Timestamp("now", tz="US/Eastern")

    # if EARLIER than 4pm, use today.
    if now.hour < 16:
        now = now.floor("D")
    else:
        # else, use tomorrow
        now = now.ceil("D")

    return now


@cached(cache=TTLCache(maxsize=1, ttl=60 * 90))
def tradingDaysRemainingInMonth():
    """Return how many trading days until the month ends...

    NOTE: we are excluding partial/early-close days from the full count..."""
    now = goodCalendarDate()
    found = marketCalendar(now, now + MonthEnd(0))

    found["duration"] = found.market_close - found.market_open
    regularDays = found[found.duration >= pd.Timedelta(hours=6)]

    # just length because the 'found' calendar has one row for each market day in the result set...
    distance = len(regularDays)

    # return ONE LESS THAN trading days found because on the LAST DAY of the month, we want
    # to say there are "0 days" remaining in month, not "1 day remaining" on the last day.
    return distance - 1


@cached(cache=TTLCache(maxsize=1, ttl=60 * 90))
def tradingDaysRemainingInYear():
    """Return how many trading days until the year ends..."""
    now = goodCalendarDate()
    found = marketCalendar(now, now + YearEnd(0))

    distance = len(found)

    # same "minus 1" reason as the days in month
    # (i.e. for the last day in the year, report "0 days remaining in year" not "1 days remaining")
    return distance - 1


@cached(cache=TTLCache(maxsize=20, ttl=60 * 90))
def tradingDaysNextN(days: int):
    """Return calendar dates for the next N trading days"""
    now = goodCalendarDate()

    periods = pd.date_range(now, periods=days)
    found = marketCalendar(periods[0], periods[-1])

    return list(found["market_open"])


# expire this cache once every 15 minutes so we only have up to 15 minutes of wrong dates after EOD
@cached(cache=TTLCache(maxsize=128, ttl=60 * 15))
def fetchEndOfMarketDayAtDate(y, m, d):
    """Return the timestamp of the next end-of-day market timestamp.

    This is currently only used for showing the "end of day" countdown timer in the toolbar,
    so it's okay if we return an expired date for a little while (the 15 minute cache interval),
    so the toolbar will just report a negative closing time for up to 15 minutes.

    The cache structure is because the toolbar refresh code is called anywhere from 1 to 10 times
    _per second_ so we want to minimize as much math and logic overhead as possible for non-changing
    values.

    We could potentially place an event timer somewhere to manually clear the cache at EOD,
    but we just aren't doing it yet."""
    [(soonestStart, soonestEnd), (nextStart, nextEnd)] = (
        fetchDateTimeOfEndOfMarketDayAtDate(y, m, d)
    )

    # this logic just helps us across the "next day" barrier when this runs right after a normal 4pm close
    # so we immediately start ticking down until the next market day close (which could be 3-4 days away depending on holidays!)
    if soonestEnd.timestamp() > whenever.ZonedDateTime.now("US/Eastern").timestamp():
        return whenever.ZonedDateTime.from_timestamp(
            soonestEnd.timestamp(), tz="US/Eastern"
        )

    return whenever.ZonedDateTime.from_timestamp(nextEnd.timestamp(), tz="US/Eastern")


def readableHTML(html) -> str:
    """Return contents of 'html' with tags stripped and in a _reasonably_
    readable plain text format.

    This is used for printing "IBKR Realtime Status Updates/News" from the API.
    The API sends news updates as HTML, so we convert it to text for terminal display.
    """

    return re.sub(
        r"(\n[\s]*)+", "\n", bs4.BeautifulSoup(html, features="html.parser").get_text()
    )
