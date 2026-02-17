"""Pure types, constants, and utility functions — no external dependencies beyond stdlib."""

from __future__ import annotations

import bisect
import datetime
import enum
import functools
import locale
import math
import re
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Final, Literal

nan: Final = float("nan")

# Also map for user typing shorthand on command line order entry.
# Values abbreviations are allowed for easier command typing support.
# NOTE: THIS LIST IS USED TO TRIGGER ORDERS IN order.py:IOrder().order() so THESE NAMES MUST MATCH
#       THE OFFICIAL IBKR ALGO NAME MAPPINGS THERE.
# This is a TRANSLATION TABLE between our "nice" names like 'AF' and the IBKR ALGO NAMES USED FOR ORDER PLACING.
# NOTE: DO NOT SEND IOrder.order() requests using 'AF' because it must be LOOKED UP HERE FIRST.
# TODO: on startup, we should assert each of these algo names match an actual implemented algo order method in IOrder().order()
ALGOMAP: Final = dict(
    LMT="LMT",
    LIM="LMT",
    LIMIT="LMT",
    AF="LMT + ADAPTIVE + FAST",
    AS="LMT + ADAPTIVE + SLOW",
    MID="MIDPRICE",
    MIDPRICE="MIDPRICE",
    SNAPMID="SNAP MID",
    SNAPMKT="SNAP MKT",
    SNAPREL="SNAP PRIM",
    SNAPPRIM="SNAP PRIM",
    MTL="MTL",  # MARKET-TO-LIMIT (execute at top-of-book, but don't sweep, just set a limit for remainder)
    PRTMKT="MKT PRT",  # MARKET-PROTECT (futs only), triggers immediately
    PRTSTOP="STP PRT",  # STOP WITH PROTECTION (futs only), triggers when price hits
    PRTSTP="STP PRT",  # STOP WITH PROTECTION (futs only), triggers when price hits
    PEGMID="PEG MID",  # Floating midpoint peg, must be directed IBKRATS or IBUSOPT
    REL="REL",
    STOP="STP",
    STP="STP",
    STPLMT="STP LMT",
    STP_LMT="STP LMT",
    TSL="TRAIL LIMIT",
    MKT="MKT",
    MIT="MIT",
    LIT="LIT",
    AFM="MKT + ADAPTIVE + FAST",
    AMF="MKT + ADAPTIVE + FAST",
    ASM="MKT + ADAPTIVE + SLOW",
    AMS="MKT + ADAPTIVE + SLOW",
    MOO="MOO",
    MOC="MOC",
)

D100: Final = Decimal("100")
DN1: Final = Decimal("-1")
DP1: Final = Decimal("1")

# Also compare: https://www.cmegroup.com/trading/equity-index/rolldates.html
FUTS_MONTH_MAPPING: Final = {
    "F": "01",  # January
    "G": "02",  # February
    "H": "03",  # March
    "J": "04",  # April
    "K": "05",  # May
    "M": "06",  # June
    "N": "07",  # July
    "Q": "08",  # August
    "U": "09",  # September
    "V": "10",  # October
    "X": "11",  # November
    "Z": "12",  # December
}

PQ: Final = enum.Enum("PQ", "PRICE QTY")

type BuySell = Literal["BUY", "SELL"]

type ContractId = int

type FPrice = float
type MaybePrice = FPrice | None
type PercentAmount = tuple[MaybePrice, MaybePrice]
type Seconds = int


@functools.lru_cache(maxsize=16)
def convert_time(seconds):
    """Converts the given seconds into a human-readable time format"""

    # Calculate weeks, days, hours, minutes and seconds
    weeks, remainder = divmod(seconds, 604800)
    days, remainder = divmod(remainder, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Create a list to store the formatted time units
    time_units = []

    # Check if each unit is greater than zero and add it to the list if so
    if weeks > 0:
        time_units.append(f"{weeks:.0f} week{'s' if weeks > 1 else ''}")

    if days > 0:
        time_units.append(f"{days:.0f} day{'s' if days > 1 else ''}")

    if hours > 0:
        time_units.append(f"{hours:.0f} hour{'s' if hours > 1 else ''}")

    if minutes > 0:
        time_units.append(f"{minutes:.0f} minute{'s' if minutes > 1 else ''}")

    if seconds > 0 or not time_units:
        time_units.append(f"{seconds:.2f} second{'s' if seconds != 1 else ''}")

    return " ".join(time_units)


def fmtmoney(val: float | int | Decimal):
    """Return a formatted money string _with_ a comma in them for thousands separator."""
    return locale.currency(val, grouping=True)


@dataclass(slots=True)
class FillReport:
    """A commission report of a filled execution.

    We can have multiple executions per 'orderId' through time or even across symbols for bags.
    """

    orderId: int
    conId: int
    sym: str
    side: str  # execution side (buy/sell etc.)
    shares: float  # number of shares traded
    price: float  # fill price
    pnl: float  # realized profit/loss
    commission: float  # commission paid
    when: datetime.datetime  # time the trade was executed

    @property
    def qty(self) -> float:
        """Return +qty for longs and -qty for shorts"""
        if self.side == "BOT":
            return self.shares

        assert self.side == "SLD"
        return -self.shares


@dataclass(slots=True)
class PaperLog:
    """Simplified paper trading log with P&L tracking."""

    _trades: list[dict[str, float]] = field(default_factory=list)

    def log(self, size: float, price: float):
        """
        Record a new paper trade.

        Args:
            size (float): Trade size (positive for long, negative for short)
            price (float): Execution price
        """
        if (
            not isinstance(size, (int, float))
            or not isinstance(price, (int, float))
            or price <= 0
        ):
            raise ValueError("Invalid trade parameters")

        self._trades.append({"size": size, "price": price})

    def report(self, current_price: float | None = None) -> dict[str, float | None]:
        """
        Generate a comprehensive trading report.

        Args:
            current_price (Optional[float]): Current market price for unrealized P&L

        Returns:
            dict: Detailed trading report
        """
        if not self._trades:
            return {
                "total_size": 0,
                "average_price": None,
                "total_cost": 0,
                "realized_pl": 0,
                "unrealized_pl": None,
                "total_pl": 0,
            }

        # Calculate total position and cost
        total_size = sum(trade["size"] for trade in self._trades)
        total_cost = sum(trade["size"] * trade["price"] for trade in self._trades)

        # Calculate average price
        average_price = total_cost / total_size if total_size != 0 else None

        # Realized P&L (profit from closed trades)
        realized_pl = self._calculate_realized_pl()

        # Unrealized P&L
        unrealized_pl = None
        if current_price is not None and total_size != 0:
            unrealized_pl = (current_price - average_price) * total_size  # type: ignore

        return {
            "total_size": total_size,
            "average_price": average_price,
            "total_cost": round(total_cost, 4) if total_cost else None,
            "realized_pl": round(realized_pl, 4) if realized_pl else None,
            "unrealized_pl": round(unrealized_pl, 4) if unrealized_pl else None,
            "total_pl": round(realized_pl + (unrealized_pl or 0), 4),
        }

    def _calculate_realized_pl(self) -> float:
        """
        Calculate realized profit/loss by matching opposite trades.

        Returns:
            float: Total realized profit/loss
        """
        realized_pl = 0

        for t in self._trades:
            realized_pl += t["size"] * t["price"]  # type: ignore

        return -realized_pl

    def reset(self):
        """Clear all trade history."""
        self._trades.clear()


@dataclass(slots=True, frozen=True)
class QuoteSizes:
    """A holder for passing around price, bid, ask, and size details."""

    bid: float | None
    ask: float | None
    bidSize: float | None
    askSize: float | None
    last: float | None
    close: float | None

    @property
    def current(self) -> float | None:
        bid = self.bid
        ask = self.ask

        if bid is not None and ask is not None:
            return (bid + ask) / 2

        # Note: don't use '.last or .close' here because on startup IBKR
        #       loads values aync, so sometimes 'close' appears before 'last'
        #       and we don't want to return potentially 1-3 day old 'close' values
        #       when we want the most recentHistoryAnchorly last traded price (even if we have to
        #       wait an extra update tick or two for it to arrive)
        # Though, there is a weird bug with the SPX ticker where, after hours, when you
        # subscribe it gives you two immediate values: the CORRECT value for the close ast 'last'
        # then an oddly incorrect value off by 1-3 points as a new 'last', so if you subscribe to SPX
        # after hours, you get two 'last' price updates and they conflict. No idea why, but even their
        # official app values show the "incorrect SPX" price after close instead of the final price.
        return ask if ask is not None else self.last


@dataclass(slots=True)
class LevelLevels:
    """Store a mapping of type (sma, volume?, etc?) and lookback duration (seconds) to level breaching price (price)."""

    levelType: str
    lookback: int
    lookbackName: str
    level: float

    def __post_init__(self) -> None:
        # use native python floats instead of allowing numpy floats to sneak in here
        self.level = float(self.level)

        # don't allow names to be "open" because our spoken events use 'OPEN' when positions are created
        # (it's confusing to have a _price alert_ say OPEN as well as an _order alert_ also use the same keyword)
        if self.lookbackName == "open":
            self.lookbackName = "start"


@dataclass(slots=True, frozen=True)
class QuoteFlowPoint:
    bid: float
    ask: float
    timestamp: float


@dataclass(slots=True)
class QuoteFlow:
    """Track the progress of bid/ask levels over time.

    This helps us see how quickly prices are moving.

    The goal is to detect how quickly bids are growing larger than previous asks (or the opposite, when asks are falling below bids).
    """

    # a 1,200 entry history gives us 5 minutes of price history at 250 ms updates
    pairs: deque[QuoteFlowPoint] = field(default_factory=lambda: deque(maxlen=1_200))

    def update(self, bid, ask, timestamp):
        """Save the current bid/ask/timestamp into our history for analyzing price direction."""
        self.pairs.append(QuoteFlowPoint(bid, ask, timestamp))

    def analyze(self):
        """Walk every recorded 'pairs' to figure out how long it takes for either a bid to become the ask or an ask to become a bid."""

        if not self.pairs:
            return defaultdict(float)

        # mapping of price difference to previous point seen for next comparison
        prevpoints: dict[float, QuoteFlowPoint] = {}

        updoot: dict[float, list[float]] = defaultdict(list)
        downdoot: dict[float, list[float]] = defaultdict(list)
        ranges = (0, 0.5, 1, 3, 5, 15)
        for p in self.pairs:
            match p:
                case QuoteFlowPoint(bid=bid, ask=ask, timestamp=timestamp) as qfp:
                    # only attempt to use valid quotes
                    if not (bid and ask):
                        continue

                    # if this is the first attempt, we want to initialize every previous value with the current value
                    if not prevpoints:
                        for r in ranges:
                            prevpoints[r] = qfp

                        continue

                    for r in ranges:
                        prevpoint = prevpoints[r]

                        if bid - prevpoint.ask >= r:
                            # price is RISING because bid is now above PREVIOUS ASK
                            updoot[r].append(timestamp - prevpoint.timestamp)

                            # update previous point since we USED it for date (otherwise the intermediate parts didn't breach)
                            prevpoints[r] = qfp
                        elif prevpoint.bid - ask >= r:
                            # price is FALLING because previous ask is BELOW current BID
                            downdoot[r].append(timestamp - prevpoint.timestamp)
                            prevpoints[r] = qfp

        # return time between last data (most recent) and first data (oldest)
        duration = self.pairs[-1].timestamp - self.pairs[0].timestamp

        upspeeds: dict[float, float] = dict()
        for r, l in updoot.items():
            upspeeds[r] = statistics.mean(l)

        downspeeds: dict[float, float] = dict()
        for r, l in downdoot.items():
            downspeeds[r] = statistics.mean(l)

        # TODO: also include stats about the DISTANCE of the breaches (0.10 cents? $10? we need to be generating better sub-stats per price range)
        # TODO: we ALSO need to populate a metric for "the current trend" if prices are NOW moving up (or the last timestamp a price moved up or down individually)
        # TODO: also try to not double count the same side if conditions remain in-range but not moving?
        # TODO: return this as a dataframe? rows are the price blocks, columns are uplen/upspeed/downlen/downspeed?
        return dict(
            duration=duration,
            uplen={k: len(v) for k, v in updoot.items()},
            upspeed=upspeeds,
            downlen={k: len(v) for k, v in downdoot.items()},
            downspeed=downspeeds,
        )


@dataclass(slots=True)
class LevelBreacher:
    """Store a collection of levels generated from a bar size with appropriate levels.

    e.g. collection: SMA
         bar size: 1 day
         levels: [<5, price>, <10, price>, <20, price>, <60, price>, <220, price>, <325, price>]
    """

    duration: int
    levels: list[LevelLevels] = field(default_factory=list)

    durationName: str = field(init=False)
    enabled: bool = True

    def __post_init__(self) -> None:
        self.durationName = convert_time(self.duration)


@dataclass(slots=True)
class Bracket:
    profitLimit: Decimal | float | None = None
    lossLimit: Decimal | float | None = None

    # Note: IBKR bracket orders must use common exchange order types (no AF, AS, REL, etc)
    orderProfit: str = "LMT"
    orderLoss: str = "STP LMT"
    lossStop: Decimal | float | None = None

    def __post_init__(self) -> None:
        # if no stop trigger provided, set stop equal to the liss limit
        if not self.lossStop:
            self.lossStop = self.lossLimit


@dataclass(slots=True)
class LadderStep:
    qty: Decimal
    limit: Decimal

    def __post_init__(self) -> None:
        assert isinstance(self.qty, Decimal)
        assert isinstance(self.limit, Decimal)


@dataclass(slots=True)
class PriceOrQuantity:
    """A wrapper/box to allow users to provide price OR quantity using one variable based on input syntax.

    e.g. "$300" is ... price $300... while "300" is quantity 300.
    """

    value: str | int | float | Decimal
    qty: float | int | Decimal = field(init=False)

    is_quantity: bool = False
    is_money: bool = False

    is_long: bool = True

    # hack for now. Fix in a better place eventually.
    exchange: str | None = "SMART"

    def __post_init__(self) -> None:
        # TODO: different currency support?

        if isinstance(self.value, (int, float, Decimal)):
            assert (
                self.is_quantity ^ self.is_money
            ), f"If you provided a direct quantity, you must enable only one of quantity or money, but got: {self}"

            self.qty = self.value

            if self.qty < 0:
                self.is_long = False

                # we don't deal in negative quantities here because IBKR sells have a sell action.
                # negative quantities only happen for prices of credit spreads because those are all "BUY [negative money]"
                self.qty = abs(self.qty)  # type: ignore
        else:
            # else, input is a string and we auto-detect money-vs-quantity depending on if the
            # string value starts with '$' (is money value) or not (is direct quantity)

            assert isinstance(self.value, str)

            # allow numbers to use '_' or ',' for any digit breaks
            self.value = self.value.replace("_", "").replace(",", "")

            # if we have a negative sign in the value, consider it a short.
            # allow: -10 -$10 and $-10 to all activate the short detector.
            if self.value.startswith("-") or self.value.startswith("$-"):
                self.is_long = False
                self.value = self.value.replace("-", "")

            if self.value[0] == "$":
                self.is_money = True
                self.qty = float(self.value[1:])
            else:
                self.qty = float(self.value)
                self.is_quantity = True

        # if there's no fractional component, use integer quantity directly
        iqty = int(self.qty)
        if self.qty == iqty:
            self.qty = iqty

    def __repr__(self) -> str:
        if self.is_quantity:
            return f"{self.qty:,.2f}"

        return locale.currency(self.qty, grouping=True)


def convert_futures_code(code: str):
    """Convert a futures-date-format into IBKR date format.

    So input like 'Z3' becomes 202312.
    """

    assert (
        len(code) == 2 and code[-1].isdigit()
    ), f"Futures codes are two characters like F3 for January 2023, but we got: {code}"

    # Mapping for month codes as per future contracts
    try:
        month_code = FUTS_MONTH_MAPPING[code[0].upper()]
    except KeyError:
        raise ValueError("Invalid month code in futures contract")

    # our math accounts for any numbers PREVIOUS to this year are for the NEXT decade,
    # while numbers FOWARD from here are for the current decade.
    current_year = datetime.datetime.now().year
    year_decade_start = current_year - current_year % 10
    year = year_decade_start + int(code[1])

    return str(year) + month_code


def find_nearest(lst, target):
    """
    Finds the nearest number in a sorted list to the given target.

    If there is an exact match, that number is returned. Otherwise, it returns the number with the smallest
    numerical difference from the target within the list.

    Args:
        lst (list): A sorted list of numbers.
        target (int): The number for which to find the nearest value in the list.

    Returns:
        The nearest index to `target` in `lst`.

    Bascially: using ONLY bisection causes rounding problems because if a query is just 0.0001 more than a value
               in the array, then it picks the NEXT HIGHEST value, but we don't want that, we want the NEAREST value
               which minimizes the difference between the input value and all values in the list.

               So, instead of just "bisect and use" we do the bisect then compare the numerical difference between
               the current element and the next element to decide whether to round down or up from the current value.
    """

    # Get the index where the value should be inserted (rounded down)
    idx = bisect.bisect_left(lst, target) - 1

    size = len(lst)

    # If the difference to the current element is less than or equal to the difference to the next element
    try:
        # this is equivalent to MATCHING or ROUNDING DOWN
        if idx >= 0 and abs(target - lst[idx]) <= abs(target - lst[idx + 1]):
            return idx
    except:
        # if list[idx + 1] doesn't exist (beyond the list) just fall through and we'll return "size - 1" which is the maximum position.
        pass

    # If we need to round up, return the next index in the list (or the final element if we've reached beyond the end of the list)
    return idx + 1 if idx < size - 1 else size - 1


def sortLocalSymbol(s):
    """Given tuple of (occ date/right/strike, symbol) return
    tuple of (occ date, symbol)"""

    return (s[0][:6], s[1])


def portSort(p):
    """sort portfolioItem 'p' by symbol if stock or by (expiration, symbol) if option"""
    s = tuple(reversed(p.contract.localSymbol.split()))
    if len(s) == 1:
        return s

    # if option, sort by DATE then SYMBOL (i.e. don't sort by (date, type, strike) complex
    return sortLocalSymbol(s)


def tradeOrderCmp(o):
    """Return the sort key for a trade representing a live order.

    The goal is to sort by:
        - BUY / SELL
        - DATE (if has date, expiration, option, warrant, etc)
        - SYMBOL

    Sorting is also flexible where if no date is available, the sort still works fine.
    """

    # Sort all options by expiration first then symbol
    # (no impact on symbols without expirations)
    useSym = o.contract.symbol
    useName = useSym
    useKey = o.contract.localSymbol.split()
    useDate = -1

    # logger.info("Using to sort: {}", o)

    if useKey:
        useName = useKey[0]
        if len(useKey) == 2:
            useDate = useKey[1]
        else:
            # the 'localSymbol' date is 2 digit years while the 'lastTradeDateOrContractMonth' is
            # four digit years, so to compare, strip the leading '20' from LTDOCM
            useDate = o.contract.lastTradeDateOrContractMonth[2:]

    # logger.info("Generated sort key: {}", (useDate, useSym, useName))

    return (o.log[-1].status, str(useDate), str(useSym), str(useName))


def boundsByPercentDifference(mid: float, percent: float) -> tuple[float, float]:
    """Returns the lower and upper percentage differences from 'mid'.

    Percentage is given as a full decimal percentage.
    Example: 0.25% must be provided as 0.0025"""
    # Obviously doesn't work for mid == 0 or percent == 2, but we shouldn't
    # encounter those values under normal usage.

    # This is just the percentage difference between two prices equation
    # re-solved for a and b from: (a - b) / ((a + b) / 2) = percent difference
    lower = -(mid * (percent - 2)) / (percent + 2)
    upper = -(mid * (percent + 2)) / (percent - 2)
    return (lower, upper)


def split_commands(text):
    """A helper for splitting in-quote commands delimited by semicolons.

    We can't just split the whole string by semicolons because we have to respect the string boundaries
    if there are quoted elements, so we just get to iterate the entire string character by character. yay.
    """
    # Remove comments
    text = re.sub(r"\s+#.*", "", text).strip()

    # Initialize variables
    commands = []
    current_command = ""
    in_quotes = False
    escape_next = False

    for char in text:
        if escape_next:
            current_command += char
            escape_next = False
        elif char == "\\":
            current_command += char
            escape_next = True
        elif char == '"':
            current_command += char
            in_quotes = not in_quotes
        elif char == ";" and not in_quotes:
            commands.append(current_command.strip())
            current_command = ""
        else:
            current_command += char

    if current_command:
        commands.append(current_command.strip())

    return commands


def as_duration(seconds):
    """Converts the given seconds into a human-readable time format

    (more compressed format limited to 'days' versus convert_time())
    """

    # Calculate weeks, days, hours, minutes and seconds
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Create a list to store the formatted time units
    time_units = []

    if days > 0:
        time_units.append(f"{days:.0f} d")

    if hours > 0:
        time_units.append(f"{hours:.0f} hr")

    if minutes > 0:
        time_units.append(f"{minutes:.0f} min")

    if seconds > 0 or not time_units:
        time_units.append(f"{seconds:.2f} s")

    return " ".join(time_units)
