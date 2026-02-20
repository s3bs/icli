#!/usr/bin/env python3

original_print = print
import asyncio
import dataclasses
import datetime
import fnmatch  # for glob string matching!
import functools
import locale  # automatic money formatting
import logging
import math
import os
import pathlib
import re
import sys
from collections import defaultdict
from collections.abc import Awaitable, Callable, Coroutine, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Final, Literal

# http://www.grantjenks.com/docs/diskcache/
import diskcache  # type: ignore
import pandas as pd
import pytz
import whenever
from pandas.tseries.offsets import Week
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory, ThreadedHistory
from prompt_toolkit.shortcuts import set_title
from prompt_toolkit.styles import Style

from icli.completer import CommandCompleter

import icli.engine.audio as awwdio
import icli.calc
import icli.engine.orders as orders

from . import instrumentdb, utils

locale.setlocale(locale.LC_ALL, "")

import ib_async
import prettyprinter as pp  # type: ignore
import tradeapis.buylang as buylang
import tradeapis.cal as mcal
import tradeapis.ifthen as ifthen
import tradeapis.ifthen_templates as ifthen_templates
import tradeapis.orderlang as orderlang
from ib_async import (
    IB,
    IBDefaults,
    Bag,
    ComboLeg,
    Contract,
    Future,
    Index,
    NewsBulletin,
    NewsTick,
    Order,
    PnLSingle,
    Position,
    Stock,
    Ticker,
    Trade,
)
from loguru import logger
from mutil.bgtask import BGSchedule, BGTask, BGTasks
from mutil.timer import Timer
from tradeapis.ordermgr import OrderMgr
from tradeapis.ordermgr import Trade as OrderMgrTrade

from icli import cmds
from icli.cmds.orders.straddle import IOpStraddleQuote
from icli.engine.algobinder import AlgoBinder
from icli.engine.contracts import (
    CompleteTradeNotification,
    FullOrderPlacementRecord,
    contractForName,
    nameForContract,
)
from icli.engine.primitives import (
    Bracket,
    PaperLog,
    PriceOrQuantity,
    split_commands,
)
from icli.engine.calendar import (
    tradingDaysRemainingInMonth,
    tradingDaysRemainingInYear,
    tradingDaysNextN,
)
from icli.engine.technicals import TWEMA
from icli.helpers import FUT_EXP, IPosition, ITicker, Ladder

USEastern: Final = pytz.timezone("US/Eastern")

pp.install_extras(["dataclasses"], warn_on_error=False)

# environment 1 true; 0 false; flag for determining if EVERY QUOTE (4 Hz per symbol) is saved to a file
# for later backtest usage or debugging (note: this uses the default python 'json' module which sometimes
# outputs non-JSON compliant NaN values, so you may need to filter those out if read back using a different
# json parser)
ICLI_DUMP_QUOTES = bool(int(os.getenv("ICLI_DUMP_QUOTES", 0)))


# Fields updated live for toolbar printing.
# Printed in the order of this list (the order the dict is created)
# Some math and definitions for values:
# https://www.interactivebrokers.com/en/software/tws/usersguidebook/realtimeactivitymonitoring/available_for_trading.htm
# https://ibkr.info/node/1445
LIVE_ACCOUNT_STATUS: Final = [
    # row 1
    "AvailableFunds",
    # NOTE: we replaced "BuyingPower" with a 3-way breakdown instead:
    "BuyingPower4",
    "BuyingPower3",
    "BuyingPower2",
    "Cushion",
    "DailyPnL",
    "DayTradesRemaining",
    "DayTradesRemainingT+1",
    "DayTradesRemainingT+2",
    "DayTradesRemainingT+3",
    "DayTradesRemainingT+4",
    # row 2
    "ExcessLiquidity",
    "FuturesPNL",
    "GrossPositionValue",
    "MaintMarginReq",
    "OptionMarketValue",
    "EquityWithLoanValue",
    # row 3
    "NetLiquidation",
    "RealizedPnL",
    "TotalCashValue",
    "UnrealizedPnL",
    "SMA",
    # unpopulated:
    #    "Leverage",
    #    "HighestSeverity",
]

# we need to add extra keys for VERIFICATION, but we don't show these extra keys directly in the status bar...
STATUS_FIELDS_PROCESS: Final = set(LIVE_ACCOUNT_STATUS) | {"BuyingPower"}


stocks = ["QQQ", "SPY", "AAPL"]

# Futures to exchange mappings:
# https://www.interactivebrokers.com/en/index.php?f=26662
# Note: Use ES and RTY and YM for quotes because higher volume
#       also curiously, MNQ has more volume than NQ?
# Volumes at: https://www.cmegroup.com/trading/equity-index/us-index.html
# ES :: MES
# RTY :: M2K
# YM :: MYM
# NQ :: MNQ
sfutures = {
    "CME": ["ES", "RTY", "MNQ", "GBP"],  # "HE"],
    "CBOT": ["YM"],  # , "TN", "ZF"],
    #    "NYMEX": ["GC", "QM"],
}

# Discovered via mainly: https://www.linnsoft.com/support/symbol-guide-ib
# The DJI / DOW / INDU quotes don't work.
# The NDX / COMP quotes require differen't data not included in default packages.
#    Index("COMP", "NASDAQ"),
idxs = [
    Index("SPX", "CBOE"),
    # No NANOS because most brokers don't offer it and it has basically no volume
    # Index("NANOS", "CBOE"),  # SPY-priced index options with no multiplier
    Index("VIN", "CBOE"),  # VIX Front-Month Component (near term)
    Index("VIF", "CBOE"),  # VIX Front-er-Month Component (far term)
    Index("VIX", "CBOE"),  # VIX Currently (a mix of VIN and VIF basically)
    # No VOL-NYSE because it displays billions of shares and breaks our views
    # Index("VOL-NYSE", "NYSE"),
    Index("TICK-NYSE", "NYSE"),
    # > 1 == selling pressure, < 1 == buying pressure; somewhat
    Index("TRIN-NYSE", "NYSE"),
    # Advancing minus Declining (bid is Advance, ask is Decline) (no idea what the bid/ask qtys represent)
    Index("AD-NYSE", "NYSE"),
]

# Note: ContFuture is only for historical data; it can't quote or trade.
# So, all trades must use a manual contract month (quarterly)
# TODO: we should be consuming a better expiration date system because some
#       futures expire end-of-month (interest rates), others quarterly (indexes), etc.
futures = [
    Future(
        symbol=sym,
        lastTradeDateOrContractMonth=FUT_EXP or "",
        exchange=x,
        currency="USD",
    )
    for x, syms in sfutures.items()
    for sym in syms
]


# Common timezone abbreviations -> IANA names for the 'set timezone' command
_TIMEZONE_ALIASES: Final[dict[str, str]] = {
    "ET": "US/Eastern",
    "EST": "US/Eastern",
    "EDT": "US/Eastern",
    "CT": "US/Central",
    "CST": "US/Central",
    "CDT": "US/Central",
    "MT": "US/Mountain",
    "MST": "US/Mountain",
    "MDT": "US/Mountain",
    "PT": "US/Pacific",
    "PST": "US/Pacific",
    "PDT": "US/Pacific",
    "GMT": "GMT",
    "UTC": "UTC",
    "CET": "CET",
    "EET": "EET",
    "JST": "Asia/Tokyo",
    "HKT": "Asia/Hong_Kong",
    "SGT": "Asia/Singapore",
    "IST": "Asia/Kolkata",
    "AEST": "Australia/Sydney",
    "BST": "Europe/London",
}


@dataclass(slots=True)
class IBKRCmdlineApp:
    # Your IBKR Account ID (auto-discovered if not provided)
    accountId: str = ""

    # number of seconds between refreshing the toolbar quote/balance views
    # (more frequent updates requires higher CPU utilization for the faster redrawing)
    toolbarUpdateInterval: float = 2.22

    host: str = "127.0.0.1"
    port: int = 4001

    # global client ID for your IBKR gateway connection (must be unique per client per gateway)
    clientId: int = field(default_factory=lambda: int(os.getenv("ICLI_CLIENT_ID", 0)))
    customName: str = field(default_factory=lambda: os.getenv("ICLI_CLIENT_NAME", ""))

    # initialized to True/False when we first see the account
    # ID returned from the API which will tell us if this is a
    # sandbox ID or True Account ID
    isSandbox: bool | None = None

    # The Connection
    ib: IB = field(
        default_factory=lambda: IB(
            defaults=IBDefaults(emptyPrice=None, emptySize=None, unset=None, timezone=USEastern)
        )
    )

    # count total toolbar refreshes
    updates: int = 0

    # same as 'updates' except this resets to 0 if your session gets disconnected then reconnected
    updatesReconnect: int = 0

    now: whenever.ZonedDateTime = field(
        default_factory=lambda: whenever.ZonedDateTime.now("US/Eastern")
    )
    nowpy: datetime.datetime = field(default_factory=lambda: datetime.datetime.now())

    quotesPositional: list[tuple[str, ITicker]] = field(default_factory=list)
    dispatch: cmds.Dispatch = field(default_factory=cmds.Dispatch)

    # holder for background events being run for some purpose
    tasks: BGTasks = field(init=False)

    # our own order tracking!
    ordermgr: OrderMgr = field(init=False)

    # Timed Events!
    scheduler: BGTasks = field(init=False)

    # use a single calculator instance so we only need to parse the grammar once
    calc: icli.calc.Calculator = field(init=False)

    # generic cache for data usage (strikes, etc)
    cache: Mapping[Any, Any] = field(
        default_factory=lambda: diskcache.Cache("./cache-multipurpose")
    )

    # global state variables (set per-client and per-session currently with no persistence)
    # We also populate the defaults here. We can potentially have these load from a config
    # file instead of being directly stored here.
    localvars: dict[str, str] = field(
        default_factory=lambda: dict(exchange="SMART", loglevel="INFO", altrow_color="#c0c0c0", timezone="US/Eastern")
    )

    # State caches
    quoteState: dict[str, ITicker] = field(default_factory=dict)
    contractIdsToQuoteKeysMappings: dict[int, str] = field(default_factory=dict)
    depthState: dict[Contract, Ticker] = field(default_factory=dict)
    summary: dict[str, float] = field(default_factory=dict)
    pnlSingle: dict[int, PnLSingle] = field(default_factory=dict)
    exiting: bool = False

    # Alternating row color for zebra-striped quote rows (themed via colorset)
    altrowColor: str = field(init=False, default="#c0c0c0")

    # Console log handler (set by setupLogging, used by setConsoleLogLevel)
    _console_handler_id: int = field(init=False, default=0)
    _console_sink: Any = field(init=False, default=None)

    # cache some parsers. yes these names are confusing. sorry.
    ol: buylang.OLang = field(default_factory=buylang.OLang)
    requestlang: orderlang.OrderLang = field(default_factory=orderlang.OrderLang)

    # global ifthenRuntime for all data processing and predicate execution
    ifthenRuntime: ifthen.IfThenRuntime = field(default_factory=ifthen.IfThenRuntime)

    # maps of tempalte names to template executor instances. We have one executor per "template type"
    # we then sub-populate with more concrete symbol/algo details so we can run one template multiple
    # times with different arugments (i.e. multiple symbols trading under the same tempalte logic, etc)
    ifthenTemplates: ifthen_templates.IfThenMultiTemplateManager = field(init=False)

    # cache recent IBKR API event messages so they don't overwhelm the console
    # (some IBKR order warning/error messages repeat aggressively for multiple minutes even though they don't
    #  reall matter, so we report them once every N occurrences or time period instead of printing thousands
    #  of lines of exact duplicate warnings all at once)
    duplicateMessageHandler: utils.DuplicateMessageHandler = field(
        default_factory=utils.DuplicateMessageHandler
    )

    # in-progress: attempt in-process paper trading space for fake order tracking.
    paperLog: dict[str, PaperLog] = field(default_factory=lambda: defaultdict(PaperLog))

    # track our own custom position representations...
    # (NOTE: one 'IPosition' _may_ belong to multiple 'ContractId' values.
    #        For spreads, we assign each contract id leg to the SAME IPosition representating the spread.
    #        This also means you will get weird/broken behavior if you have contracts internal and
    #        external to spreads).
    # Also, we currently only use 'IPosition' objects for _active_ accumulation/distribution sessions,
    # so these values do not need to persist across restarts.
    iposition: dict[Contract, IPosition] = field(default_factory=dict)

    # something any interested party can await on a contract to detect when it has new COMPLETED orders.
    # Note: this only fires when an order has qtyRemaining==0 (so it fires on each ORDER, not each _execution_).
    fillers: dict[Contract, CompleteTradeNotification] = field(
        default_factory=lambda: defaultdict(CompleteTradeNotification)
    )

    # our API-validated price increment database for every instrument so we can determine
    # proper limit order price tick increments before submitting orders.
    idb: instrumentdb.IInstrumentDatabase = field(init=False)

    # Say hello to our 3rd attempt at consuming live externally-generated algo datafeeds into our trading process...
    algobinder: AlgoBinder | None = None
    algobindertask: BGTask | None = None

    speak: awwdio.AwwdioClient = field(default_factory=awwdio.AwwdioClient)

    # Specific dict of ONLY fields we show in the live account status toolbar.
    # Saves us from sorting/filtering self.summary() with every full bar update.
    accountStatus: dict[str, float] = field(
        default_factory=lambda: dict(zip(LIVE_ACCOUNT_STATUS, [0.00] * len(LIVE_ACCOUNT_STATUS)))
    )

    # Cache all contractIds and names to their fully qualified contract object values
    # TODO: replace this with our dual in-memory-disk-passthrough cache.
    conIdCache: diskcache.Cache = field(
        default_factory=lambda: diskcache.Cache("./cache-contracts")
    )

    connected: bool = False
    disableClientQuoteSnapshotting: bool = False
    loadingCommissions: bool = False

    toolbarStyle: Style = field(
        default_factory=lambda: Style.from_dict({"bottom-toolbar": "fg:default bg:default"})
    )

    opstate: Any = field(init=False)

    # Engine modules (initialized in __post_init__ via lazy imports)
    portfolio: Any = field(init=False)
    qualifier: Any = field(init=False)
    quotemanager: Any = field(init=False)
    events: Any = field(init=False)
    placer: Any = field(init=False)
    toolbar: Any = field(init=False)

    def algobinderStart(self) -> bool:
        """Returns True if we started the algobinder.
        Returns False if algobinder was already running.
        """
        if not self.algobinder:
            self.algobinder = AlgoBinder()

        if not self.algobindertask:
            logger.info("Starting algo binder task...")
            self.algobindertask = self.task_create(
                "Algo Binder Data Receiver", self.algobinder.datafeed()
            )

            return True

        return False

    def algobinderStop(self) -> bool:
        """Returns True if we stopped the algobinder live processing task (data remains available, just not updating anymore).
        Returns False if there is no active algobinder to stop.
        """

        if self.algobindertask:
            logger.info("Stopping algo binder task...")
            self.algobindertask.stop()
            self.algobindertask = None
            return True

        return False

    def updateToolbarStyle(self, val: str, altrow_color: str | None = None) -> None:
        """Create new style object when style text"""

        assert isinstance(val, str)

        # note: add 'noreverse' to make bg=bg and fg=fg, otherwise it treats bg=fg and fg=bg
        # bg #33363D is nice, but needs a lighter font
        # kinda nice (camo green-ish) #708C4C
        # https://ethanschoonover.com/solarized/#the-values
        # Solarized(ish): {"bottom-toolbar": "fg:#002B36 bg:#839496"}

        # Want to add your own custom theme? Submit an issue with good color combinations and we'll add it!
        # Each theme is (toolbar_style, altrow_color) for alternating quote rows.
        schemes: Final = dict(
            default=("fg:default bg:default", "#c0c0c0"),
            solar1=("fg:#002B36 bg:#839496", "#003845"),
        )

        # if input is a theme name, use the theme colors
        if theme := schemes.get(val):
            logger.info("Setting toolbar style ({}): {}", val, theme[0])
            altrow_color = altrow_color or theme[1]
            val = theme[0]
        else:
            logger.info("Setting toolbar style: {}", val)

        self.toolbarStyle = Style.from_dict({"bottom-toolbar": val})
        if altrow_color:
            self.altrowColor = altrow_color

    def tradingDays(self, days):
        return tradingDaysNextN(days)

    @property
    def diy(self) -> int:
        """Return remaining trading days in year"""
        return tradingDaysRemainingInYear()

    @property
    def dim(self) -> int:
        """Return remaining trading days in month"""
        return tradingDaysRemainingInMonth()

    def __post_init__(self) -> None:
        # just use the entire IBKRCmdlineApp as our app state!
        self.opstate = self
        self.setupLogging()

        # attach runtime to multi-template manager (since we can't attach the runtime at field() init time)
        self.ifthenTemplates = ifthen_templates.IfThenMultiTemplateManager(self.ifthenRuntime)

        # note: ordermgr is NOT scoped per-client because all clients can see all positions.
        self.ordermgr = OrderMgr("Executed Positions")

        self.tasks = BGTasks(f"icli client {self.clientId} internal")
        self.scheduler = BGTasks(f"icli client {self.clientId} scheduler")

        # provide ourself to the calculator so the calculator can lookup live quote prices and live account values
        self.calc = icli.calc.Calculator(self)

        # provide ourself to instrumentdb so it can also use live API calls
        self.idb = instrumentdb.IInstrumentDatabase(self)

        from icli.engine.portfolio import PortfolioQueries

        self.portfolio = PortfolioQueries(self.ib, self, self.conIdCache, self.idb)

        from icli.engine.qualification import ContractQualifier

        self.qualifier = ContractQualifier(self.ib, self.conIdCache, self.quoteState, self.ol)

        from icli.engine.quotemanager import QuoteManager

        self.quotemanager = QuoteManager(
            self.ib,
            self.quoteState,
            self.quotesPositional,
            self.contractIdsToQuoteKeysMappings,
            self.conIdCache,
            self.idb,
            self.ol,
            app=self,
        )

        from icli.engine.events import IBEventRouter

        self.events = IBEventRouter(
            self.ib,
            self.quoteState,
            self.summary,
            self.accountStatus,
            self.pnlSingle,
            self.iposition,
            self.fillers,
            self.ordermgr,
            self.speak,
            self.duplicateMessageHandler,
            self.ifthenRuntime,
            self.conIdCache,
            self.contractIdsToQuoteKeysMappings,
            app=self,
        )

        from icli.engine.placement import OrderPlacer

        self.placer = OrderPlacer(self.ib, self.conIdCache, self.idb, app=self)

        from icli.engine.toolbar import ToolbarRenderer

        self.toolbar = ToolbarRenderer(app=self)

    def setupLogging(self) -> None:
        # Configure logger where the ib_insync live service logs get written.
        # Note: if you have weird problems you don't think are being exposed
        # in the CLI, check this log file for what ib_insync is actually doing.
        now = pd.Timestamp("now")
        LOGDIR = (
            pathlib.Path(os.getenv("ICLI_LOGDIR", "runlogs")) / f"{now.year}" / f"{now.month:02}"
        )
        LOGDIR.mkdir(exist_ok=True, parents=True)
        LOG_FILE_TEMPLATE = str(
            LOGDIR
            / f"icli-id={self.clientId}-{whenever.ZonedDateTime.now('US/Eastern').py_datetime()}".replace(
                " ", "_"
            )
        )
        logging.basicConfig(
            level=logging.INFO,
            filename=LOG_FILE_TEMPLATE + "-ibkr.log",
            format="%(asctime)s %(message)s",
        )

        logger.info("Logging session with prefix: {}", LOG_FILE_TEMPLATE)

        def asink(x):
            # don't use print_formatted_text() (aliased to print()) because it doesn't
            # respect the patch_stdout() context manager we've wrapped this entire
            # runtime around. If we don't have patch_stdout() guarantees, the interface
            # rips apart with prompt and bottom_toolbar problems during async logging.
            original_print(x, end="")

        logger.remove()
        self._console_sink = asink
        self._console_handler_id = logger.add(asink, colorize=True, level="INFO")

        # new log level to disable color bolding on INFO default
        logger.level("FRAME", no=25)
        logger.level("ARGS", no=40, color="<blue>")

        # Also configure loguru logger to log all activity to its own log file for historical lookback.
        # also, these are TRACE because we log _user input_ to the TRACE facility, but we don't print
        # it to the console (since the user already typed it in the console)
        logger.add(sink=LOG_FILE_TEMPLATE + "-icli.log", level="TRACE", colorize=False)
        logger.add(
            sink=LOG_FILE_TEMPLATE + "-icli-color.log",
            level="TRACE",
            colorize=True,
        )

    @staticmethod
    def _resolve_timezone(val: str) -> str | None:
        """Resolve a timezone string to an IANA name, or None if invalid."""
        alias = _TIMEZONE_ALIASES.get(val.upper())
        if alias:
            return alias

        # Try as direct IANA name
        try:
            whenever.ZonedDateTime.now(val)
            return val
        except Exception:
            return None

    def setConsoleLogLevel(self, level: str) -> None:
        """Change the console log level at runtime.

        Removes the current console handler and re-adds it at the new level."""
        logger.remove(self._console_handler_id)
        self._console_handler_id = logger.add(self._console_sink, colorize=True, level=level)
        logger.info("Console log level set to {}", level)

    async def qualify(self, *contracts, overwrite: bool = False) -> list[Contract]:
        return await self.qualifier.qualify(*contracts, overwrite=overwrite)

    def updateGlobalStateVariable(self, key: str, val: str | None) -> None:
        # 'val' of None means just print the output, while 'val' of empty string means delete the key.

        if val is None:
            from icli.completer import CommandCompleter

            # 'set info' also prints ICLI-prefixed environment variables
            if key.lower() == "info":
                logger.info("ICLI environment variables:")
                for k, v in sorted(os.environ.items()):
                    if k.startswith("ICLI_"):
                        logger.info("  {} = {}", k, v)
                logger.info("")

            # Show all known settings with current values (or default "off")
            all_keys = sorted(set(self.localvars) | set(CommandCompleter._SET_KEY_HELP))
            logger.info("Settings:")
            for k in all_keys:
                v = self.localvars.get(k, "off")
                desc = CommandCompleter._SET_KEY_HELP.get(k, "")
                if desc:
                    logger.info("  {:<16} = {:<16} — {}", k, v, desc)
                else:
                    logger.info("  {:<16} = {}", k, v)

            return

        original = self.localvars.get(key)

        if val:
            # if value provided, set it

            # special handling for alternating row color
            # Note: bare "#hex" values get stripped by the comment parser
            # (split_commands treats " #..." as end-of-line comments), so
            # we accept hex colors without the '#' prefix and prepend it.
            if key.lower() == "altrow_color":
                if val.lower() == "off":
                    self.altrowColor = ""
                    self.localvars[key] = val
                    logger.info("Alternating row color disabled")
                    return

                # normalize: accept bare hex (c0c0c0) and prepend #
                color = val
                if not color.startswith("#"):
                    color = f"#{color}"

                # validate: must be #rgb or #rrggbb with valid hex digits
                hexpart = color[1:]
                if len(hexpart) not in (3, 6) or not all(
                    c in "0123456789abcdefABCDEF" for c in hexpart
                ):
                    logger.error(
                        "Invalid hex color '{}'. Use 3 or 6 hex digits, e.g. c0c0c0 or fff", val
                    )
                    return

                self.altrowColor = color
                self.localvars[key] = color
                logger.info("Alternating row color set to {}", color)
                return

            # special handling for loglevel
            if key.lower() == "loglevel":
                level = val.upper()
                valid = ("TRACE", "DEBUG", "INFO", "WARNING", "ERROR")
                if level not in valid:
                    logger.error("Invalid log level '{}'. Valid: {}", val, ", ".join(valid))
                    return
                self.setConsoleLogLevel(level)
                self.localvars[key] = level
                return

            # special handling for timezone
            if key.lower() == "timezone":
                tz = self._resolve_timezone(val)
                if tz is None:
                    logger.error(
                        "Unknown timezone '{}'. Use IANA names (US/Eastern, "
                        "America/Chicago) or abbreviations (EST, CST, GMT, UTC, PT).",
                        val,
                    )
                    return
                self.localvars[key] = tz
                logger.info("Timezone set to {}", tz)
                return

            # special values if setting dte things
            if key.lower() == "dte":
                now = pd.Timestamp("now")

                if not (val.isnumeric()):
                    # pandas weekdays are indexed by Monday == 0
                    match val.lower():
                        case "monday" | "mon" | "m":
                            weekday = 0
                        case "tuesday" | "tues" | "t":
                            weekday = 1
                        case "wednesday" | "wed" | "w":
                            weekday = 2
                        case "thursday" | "thurs" | "th":
                            weekday = 3
                        case "friday" | "fri" | "f":
                            weekday = 4
                        case _:
                            logger.error("DTE values are weekdays only!")
                            return

                    # calcluate number of calendar days between now and the requested expiration day.
                    # NOTE: Due to how we automatically make "after 4pm == 0dte", this 'dte-by-day-of-week'
                    #       doesn't work correctly after hours because it will always be one extra day ahead.
                    #       (e.g. Monday after hours, our 0dte is tuesday, but calendar tuesday is 1 day away, so 1 dte == wednesday)
                    val = ((now + Week(weekday=weekday)) - now).days  # type: ignore

            self.localvars[key] = val
        else:
            # else, if value not provided, remove key
            self.localvars.pop(key, None)

        if original and not val:
            logger.info("UNSET: {} (previously: {})", key, original)
        elif original:
            logger.info("SET: {} = {} (previously: {})", key, val, original)
        else:
            logger.info("SET: {} = {}", key, val)

    def contractsForPosition(
        self, sym, qty: float | None = None
    ) -> list[tuple[Contract, float, float]]:
        return self.qualifier.contractsForPosition(sym, qty)

    async def contractForOrderRequest(self, oreq) -> Contract | None:
        return await self.qualifier.contractForOrderRequest(oreq)

    async def bagForSpread(self, oreq) -> Contract | Bag | None:
        return await self.qualifier.bagForSpread(oreq)

    def quoteResolve(self, lookup: str) -> tuple[str, Contract] | tuple[None, None]:
        return self.quotemanager.quoteResolve(lookup)

    def decimals(self, contract):
        return self.portfolio.decimals(contract)

    async def tickIncrement(self, contract: Contract) -> Decimal | None:
        return await self.placer.tickIncrement(contract)

    async def comply(
        self, contract: Contract, price: Decimal | float, direction: instrumentdb.ROUND
    ) -> Decimal | None:
        return await self.placer.comply(contract, price, direction)

    async def complyNear(self, contract: Contract, price: Decimal | float) -> Decimal | None:
        return await self.placer.complyNear(contract, price)

    async def complyUp(self, contract: Contract, price: Decimal | float) -> Decimal | None:
        return await self.placer.complyUp(contract, price)

    async def complyDown(self, contract: Contract, price: Decimal | float) -> Decimal | None:
        return await self.placer.complyDown(contract, price)

    async def safeModify(self, contract, order, **kwargs) -> Order:
        return await self.placer.safeModify(contract, order, **kwargs)

    async def fetchContractExpirations(
        self, contract: Contract, fetchDates: list[str] | None = None
    ):
        return await self.qualifier.fetchContractExpirations(contract, fetchDates)

    async def isGuaranteedSpread(self, bag: Contract) -> bool:
        return await self.qualifier.isGuaranteedSpread(bag)

    async def contractForOrderSide(self, order, contract: Contract) -> Contract:
        return await self.qualifier.contractForOrderSide(order, contract)

    async def addNonGuaranteeTagsIfRequired(self, contract, *reqorders):
        return await self.qualifier.addNonGuaranteeTagsIfRequired(contract, *reqorders)

    def createBracketAttachParent(
        self,
        order,
        sideClose,
        qty,
        profitLimit,
        lossLimit,
        lossStopPrice,
        outsideRth,
        tif,
        orderTypeProfit,
        orderTypeLoss,
        config=None,
    ) -> tuple[Order | None, Order | None]:
        return self.placer.createBracketAttachParent(
            order,
            sideClose,
            qty,
            profitLimit,
            lossLimit,
            lossStopPrice,
            outsideRth,
            tif,
            orderTypeProfit,
            orderTypeLoss,
            config,
        )

    async def placeOrderForContract(
        self,
        sym: str,
        isLong: bool,
        contract: Contract,
        qty: PriceOrQuantity,
        limit: Decimal | None,
        orderType: str,
        preview: bool = False,
        bracket: Bracket | None = None,
        config: Mapping[str, Decimal | float] | None = None,
        ladder: Ladder | None = None,
    ) -> FullOrderPlacementRecord | None:
        return await self.placer.placeOrderForContract(
            sym, isLong, contract, qty, limit, orderType, preview, bracket, config, ladder
        )

    async def generatePreviewReport(self, *args, **kwargs):
        return await self.placer.generatePreviewReport(*args, **kwargs)

    def ifthenAbs(self, content: str):
        """Just get an absolute value here..."""
        return abs(float(content))

    def ifthenQuantityForContract(self, symbol: str):
        """Convert input symbol to contract then look up quantity.

        Note: since this is a *non-async* ifthen function, we can just return the result directly."""
        contract = contractForName(symbol)
        found = self.quantityForContract(contract)
        return found

    async def ifthenExtensionVerticalSpreadCall(
        self, mailbox: dict[str, Any], symbol: str, startStrike: float, distance: float
    ):
        return await self.ifthenExtensionVerticalSpread("c", mailbox, symbol, startStrike, distance)

    async def ifthenExtensionVerticalSpreadPut(
        self, mailbox: dict[str, Any], symbol: str, startStrike: float, distance: float
    ):
        return await self.ifthenExtensionVerticalSpread("p", mailbox, symbol, startStrike, distance)

    async def ifthenExtensionVerticalSpread(
        self,
        side: Literal["p", "c"],
        mailbox: dict[str, Any],
        symbol: str,
        startStrike: float,
        distance: float,
    ):
        """Calculate a vertical spread for symbol and also subscribe to spread quote for live price updating."""
        isq = IOpStraddleQuote(
            state=self,
            symbol=symbol,
            overrideATM=startStrike,
            widths=f"v {side} 0 {distance}".split(),
        )

        contracts = await isq.run()

        assert len(contracts) == 3, (
            f"Expected 3 results from spread adding, but got {len(contracts)=} for {contracts=}?"
        )

        # technically, we think these are always in the same order of [bag, buy, sell], but if the order
        # changes, we want to always grab the bag first to populate legIds, then use legIds again.
        # Overall, the 'contracts' is only 3 elements so looping twice isn't a problem.
        for c in contracts:
            if isinstance(c, Bag):
                bag = c
                legIds = {leg.conId: leg.action for leg in bag.comboLegs}

        for c in contracts:
            if lid := legIds.get(c.conId):
                if lid == "BUY":
                    buyLeg = c
                elif lid == "SELL":
                    sellLeg = c

        mailbox["result"] = True
        mailbox["spread"] = self.nameForContract(bag)
        mailbox["buy"] = self.nameForContract(buyLeg)
        mailbox["sell"] = self.nameForContract(sellLeg)
        mailbox["contract"] = bag
        mailbox["contractBuy"] = buyLeg
        mailbox["contractSell"] = sellLeg

    def nameForContract(self, contract: Contract) -> str:
        return nameForContract(contract, self.conIdCache)

    @property
    def positionsDB(self):
        return self.portfolio.positionsDB

    def quantityForContract(self, contract):
        return self.portfolio.quantityForContract(contract)

    def averagePriceForContract(self, contract):
        return self.portfolio.averagePriceForContract(contract)

    def multiplier(self, contract):
        return self.portfolio.multiplier(contract)

    def quantityForAmount(self, contract, amount, limitPrice):
        return self.portfolio.quantityForAmount(contract, amount, limitPrice)

    def orderPriceForSpread(self, contracts: Sequence[Contract], positionSize: int):
        return self.placer.orderPriceForSpread(contracts, positionSize)

    def orderPriceForContract(self, contract: Contract, positionSize: float | int):
        return self.placer.orderPriceForContract(contract, positionSize)

    def currentQuote(self, sym, show=True) -> tuple[float | None, float | None]:
        return self.quotemanager.currentQuote(sym, show)

    async def loadExecutions(self) -> None:
        return await self.placer.loadExecutions()

    def updateOrder(self, trade: Trade):
        return self.events.updateOrder(trade)

    def errorHandler(self, reqId, errorCode, errorString, contract):
        return self.events.errorHandler(reqId, errorCode, errorString, contract)

    def cancelHandler(self, err):
        return self.events.cancelHandler(err)

    def commissionHandler(self, trade, fill, report):
        return self.events.commissionHandler(trade, fill, report)

    def newsBHandler(self, news: NewsBulletin):
        return self.events.newsBHandler(news)

    def newsTHandler(self, news: NewsTick):
        return self.events.newsTHandler(news)

    async def orderExecuteHandler(self, trade, fill):
        return await self.events.orderExecuteHandler(trade, fill)

    def positionEventHandler(self, position: Position):
        return self.events.positionEventHandler(position)

    async def positionActiveLifecycleDoctrine(self, contract, target, upTick=0.25, downTick=0.25):
        return await self.events.positionActiveLifecycleDoctrine(contract, target, upTick, downTick)

    async def predicateSetup(self, prepredicate: ifthen.CheckableRuntime):
        """Attach data extractors and custom functions to all predicates inside 'prepredicate'.

        The ifthen predicate language only _describes_ what to check, but we must provide the predicate(s)
        with the actual data sources and custom functions so the predicate(s) can execute their checks
        with live data on every update.
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
                foundsym, c = await self.positionalQuoteRepopulate(symbol)

                logger.info("[{}] Tracking contract: {}", pid, c)

                # subscribe if not subscribed (no-op if already subscribed, but returns symkey either way)
                try:
                    symkey = self.addQuoteFromContract(c)
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
                    extractor.actual = self.nameForContract(iticker.contract)

                    logger.info(
                        "[{}] Assigning field extractor: {} ({}) @ {} {}",
                        pid,
                        extractor.symbol,
                        extractor.actual,
                        datafield,
                        timeframe or "",
                    )

                    assert datafield
                    fetcher = self.dataExtractorForTicker(iticker, datafield, timeframe or 0)

                    extractor.datafetcher = fetcher

            # now do the same for functions (if any)
            fnfetcher: (
                Callable[[dict[str, Any], str, float, float], Coroutine[Any, Any, Any]]
                | Callable[[str], Any]
            )

            for fn in predicate.functions():
                match fn.datafield.lower():
                    case "verticalput" | "vp":
                        # generate a vertical put (long) spread near requested start strike extending by point distance
                        # verticalPut(strike price for long leg, distance to short leg)
                        # Note: use negative distance to go DOWN to a lower priced short put leg
                        fnfetcher = self.ifthenExtensionVerticalSpreadPut
                    case "verticalcall" | "vc":
                        # generate a vertical put (long) spread near requested start strike extending by point distance
                        # verticalCall(strike price for long leg, distance to short leg)
                        # Note: use positive distance to go UP to a lower priced short call leg
                        fnfetcher = self.ifthenExtensionVerticalSpreadCall
                    case "position" | "pos" | "p":
                        fnfetcher = self.ifthenQuantityForContract
                    case "abs":
                        fnfetcher = self.ifthenAbs

                fn.scheduler = functools.partial(
                    self.task_create, f"[{pid}] predicate executor for {fn.datafield}"
                )
                fn.datafetcher = fnfetcher

    def dataExtractorForTicker(self, iticker: ITicker, field: str, timeframe: int):
        """Return a zero-argument function querying the live 'iticker' for 'field' and potentially 'timeframe' updates."""
        fetcher = None

        # a dot in the field means we HAVE AN ALGO! ALGO ALERT! ALGO ALERT!
        # TODO: maybe move this to an algo: prefix instead of just any dots?
        if "." in field:
            if not self.algobindertask:
                self.algobinderStart()

            assert self.algobinder

            # Note: it's up to the user ensuring a 100% correct algo field description for the full 3, 5, 8+ level depth they expect...
            return lambda *args: self.algobinder.read(field)

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
                fetcher = lambda *args: self.nameForContract(iticker.ticker.contract)  # type: ignore
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
                accountReader = self.ib.wrapper.portfolio[self.accountId]
                fetcher = lambda *args: (
                    accountReader[contractId].averageCost
                    / math.copysign(accountReader[contractId].position, mul)
                )
            case "qty":
                # fetch live qty for position as reported by portfolio reporting
                assert iticker.ticker.contract

                contractId = iticker.ticker.contract.conId
                accountReader = self.ib.wrapper.portfolio[self.accountId]
                fetcher = lambda *args: accountReader[contractId].position
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

    def tickersUpdate(self, tickr):
        return self.events.tickersUpdate(tickr)

    def updateSummary(self, v):
        return self.events.updateSummary(v)

    def updateURPLPercentages(self):
        return self.events.updateURPLPercentages()

    def updatePNL(self, v):
        return self.events.updatePNL(v)

    def updatePNLSingle(self, v):
        return self.events.updatePNLSingle(v)

    def updateAgentAccountStatus(self, category, update):
        return self.events.updateAgentAccountStatus(category, update)

    def bottomToolbar(self):
        return self.toolbar.render()

    async def qask(self, terms) -> dict[str, Any] | None:
        """Ask a questionary survey using integrated existing toolbar showing"""
        result = dict()
        extraArgs = dict(
            bottom_toolbar=self.bottomToolbar,
            refresh_interval=self.toolbarUpdateInterval,
            style=self.toolbarStyle,
        )
        for t in terms:
            if not t:
                continue

            try:
                got = await t.ask(**extraArgs)
            except EOFError:
                # if user hits CTRL-D in an input box, we get an exception which is just an input error
                got = None

            # if user canceled, give up
            # See: https://questionary.readthedocs.io/en/stable/pages/advanced.html#keyboard-interrupts
            if got is None:
                return None

            result[t.name] = got

        return result

    def levelName(self):
        if self.isSandbox is None:
            return "undecided"

        if self.isSandbox:
            return "paper"

        return "live"

    def addQuoteFromContract(self, contract):
        return self.quotemanager.addQuoteFromContract(contract)

    @property
    def quoteStateSorted(self):
        return self.quotemanager.quoteStateSorted

    def quoteExists(self, contract):
        return self.quotemanager.quoteExists(contract)

    def scanStringReplacePositionsWithSymbols(self, query):
        return self.quotemanager.scanStringReplacePositionsWithSymbols(query)

    async def positionalQuoteRepopulate(self, sym, exchange="SMART"):
        return await self.quotemanager.positionalQuoteRepopulate(sym, exchange)

    async def addQuotes(self, symbols):
        return await self.quotemanager.addQuotes(symbols)

    def complyITickersSharedState(self):
        return self.quotemanager.complyITickersSharedState()

    async def runCollective(self, concurrentCmds):
        """Given a list of commands and arguments, run them all concurrently."""

        # Run all our concurrent tasks NOW
        cmds = "; ".join([x[2] for x in concurrentCmds])
        with Timer(cmds):
            try:
                await asyncio.gather(
                    *[
                        self.dispatch.runop(
                            collectiveCmd,
                            collectiveRest[0] if collectiveRest else None,
                            self.opstate,
                        )
                        for collectiveCmd, collectiveRest, _originalFullCommand in concurrentCmds
                    ]
                )
            except:
                logger.exception("[{}] Collective command running failed?", cmds)

    def _printHelpWithDescriptions(self):
        """Print all commands grouped by category with their docstrings."""
        d = self.dispatch.d

        # Build a flat lookup from full command name -> class using the original
        # opcodes (not totalOps, which drops ambiguous prefixes like 'report').
        fullname_to_cls: dict[str, type] = {}
        for val in d.opcodes.values():
            if isinstance(val, dict):
                fullname_to_cls.update(val)

        def get_doc(name: str) -> str:
            cls = fullname_to_cls.get(name) or d.totalOps.get(name)
            if cls and cls.__doc__ and "state: Any = None" not in cls.__doc__:
                return cls.__doc__.strip().split("\n")[0]
            return ""

        uncategorized = []
        for row in d.cmdsByGroup:
            if isinstance(row, dict):
                for category, commands in row.items():
                    print(f"\n{category}:")
                    for name in sorted(commands):
                        print(f"  {name:20s} {get_doc(name)}")
            else:
                uncategorized.append(row)

        if uncategorized:
            print("\nOther:")
            for name in sorted(uncategorized):
                print(f"  {name:20s} {get_doc(name)}")

    async def runSingleCommand(self, cmd, rest):
        import time

        _t0 = time.perf_counter()
        try:
            try:
                await self.dispatch.runop(cmd, rest[0] if rest else None, self.opstate)
            except Exception as e:
                if self.localvars.get("bigerror"):
                    err = logger.exception
                else:
                    logger.warning(
                        "Using small exception printer. 'set bigerror yes' to enable full stack trace messages."
                    )
                    err = logger.error

                # NOTE: during FULL MARKET HOURS, printing these execptions now cause a 25 second terminal pause
                #       because loguru obtains a global Lock() against all logging when the full exception prints,
                #       and when we are doing 20,000 events per second, lots of things get delayed.
                # So, basically, set 'bigerror' if you care about full exceptions, else trust the smaller exceptions.
                se = str(e)
                if "token" in se or "terminal" in se:
                    # don't show a 100 line stack trace for mistyped inputs.
                    # Just tell the user it needs to be corrected.
                    err("[{}] Error parsing your input: {}", [cmd] + rest or [], se)
                else:
                    err("[{}] Error with command: {}", [cmd] + rest or [], se)
        finally:
            logger.debug("[{}] Duration: {:,.4f}", cmd, time.perf_counter() - _t0)

    def buildRunnablesFromCommandRequest(self, text1):
        # Attempt to run the command(s) submitted into the prompt.
        #
        # Commands can be:
        # Regular single-line commands:
        #  > COMMAND
        #
        # Multiple commands on a single line with semicolons splitting them:
        #  > COMMAND1; COMMAND2
        #
        # Multiple commands across multiple lines (easy for pasting from other scripts generating commands)
        #  > COMMAND1
        #    COMMAND2
        #
        # Commands can have end of line comments which *do* get saved to history, but *DO NOT* get sent to the command
        # > COMMAND # Comment about command
        #
        # Commands can also be run in groups all at once concurrently.
        # Concurrent commands requested back-to-back all run at the same time and non-concurrent commands between concurrent groups block as expected.
        #
        # This will run (1, 2) concurrently, then 3, then 4, then (5, 6) concurrently again.
        # > COMMAND1&; COMMAND2&; COMMAND3; COMMAND4; COMMAND5&; COMMAND6&
        #
        # Command processing process is:
        #  - Detect end-of-line comment and remove it (comments are PER FULL INPUT so "CMD1; CMD2; # CMD3; CMD4; CMD5" only runs "CMD1; CMD2")
        #  - Split input text on newlines and semicolons
        #  - Remove leading/trailing whitespace from each split command
        #  - Check if command is a concurrent command request (add to concurrent group if necessary)
        #  - Check if command is regular (add to regular runner if necessary)
        #  - Run collected concurrent and sequential command(s) in submitted group order.
        #
        # Originally we didn't have concurrency groups, so we processed commands in a simple O(N) loop,
        # but now we pre-process (concurrent, sequential) commands first, then we run commands after we
        # accumulate them, so we have ~O(2N) processing, but our N is almost always less than 10.
        #
        # (This command processing logic went from "just parse 1 command per run" to our
        #  current implementation of handling multi-commands and comments and concurrent commands,
        #  so command parsing has increased in complexity, but hopefully the increased running logic is
        #  useful to enable more efficient order entry/exit management.)
        #
        # These nice helpers require some extra input processing work, but our
        # basic benchmark shows cleaning up these commands only requires an
        # extra 30 us at the worst case, so it still allows over 30,000 command
        # parsing events per second (and we always end up blocked by the IBKR
        # gateway latency anyway which takes 100 ms to 300 ms for replies to the API)

        runnables: list[Awaitable[None]] = []

        # 'collective' holds the current accumulating concurrency group
        collective = []

        # Note: comments (if any) must have a leading space so we don't wipe out things like setting color hex codes with fg:#dfdfdf etc

        # We needed this more complex command runner because we can't just "split on semicolons" since if the semicolon designators
        # are *inside* quotes, we must not split commands *inside* a quote group because anything inside quotes must remain
        # untouched because it should be passed as-is as parameters for further processing.
        ccmds = split_commands(text1)
        # logger.info("ccmds: {}", ccmds)

        for ccmd in ccmds:
            ccmd = ccmd.strip()

            # if the split generated empty entries (like running ;;;;), just skip the command
            if not ccmd:
                continue

            # custom usability hack: we can detect math ops and not need to prefix 'math' to them manually
            if ccmd[0] == "(":
                ccmd = f"math {ccmd}"
            elif ccmd[0] == "!":
                ccmd = f"debug {ccmd[1:].lstrip()}"
            elif ccmd.startswith("if ") or ccmd.startswith("while "):
                # also allow 'if' statements directly then auto-prepend 'ifthen' to them.
                ccmd = f"ifthen {ccmd}"

            # Check if this command is a background command then clean it up
            isBackgroundCmd = ccmd[-1] == "&"
            if isBackgroundCmd:
                # remove ampersand from background request and re-strip command...
                ccmd = ccmd[:-1].rstrip()

            # split into command dispatch lookup and arguments to command
            cmd, *rest = ccmd.split(" ", 1)
            # logger.info("cmd: {}, rest: {}", cmd, rest)

            # Intercept bare '?' to show enhanced help with descriptions
            if cmd == "?":
                self._printHelpWithDescriptions()
                continue

            # If background command, add to our background concurrency group for this block
            if isBackgroundCmd:
                # now fixup background command...
                collective.append((cmd, rest, ccmd))

                # this 'run group' count is BEFORE the runnable is added
                logger.info(
                    "[{} :: concurrent] Added command to run group {}",
                    ccmd,
                    len(runnables),
                )
                continue

            # if we have previously saved concurrent tasks and this task is NOT concurrent, add all concurrent tasks,
            # THEN add this task.
            if collective and not isBackgroundCmd:
                runnables.append(self.runCollective(collective.copy()))

                # now since we added everything, remove the pending tasks so we don't schedule them again.
                collective.clear()

            # now schedule SINGLE command since we know the collective is properly handled already
            runnables.append(self.runSingleCommand(cmd, rest))

            if len(runnables) and len(ccmds) > 1:
                # this 'run group' count is AFTER the runnable is added (so we subtract one to get the actual order number)
                logger.info(
                    "[{} :: sequential] Added command to run group {}",
                    ccmd,
                    len(runnables) - 1,
                )

        # extra catch: if our commands END with a collective command, we need to now add them here too
        # (because the prior condition only checks if we went collective->single; but if we are ALL collective,
        #  we never trigger the "is single, cut previously collective into a full group" condition)
        if collective:
            runnables.append(self.runCollective(collective.copy()))

        return runnables

    @staticmethod
    async def _checkIbAsyncRemote() -> tuple[str, str, int] | None:
        """Fetch latest ib_async info from GitHub.

        Returns (remote_version, head_sha, commits_ahead) or None on failure.
        commits_ahead is the number of commits on main after the installed version's
        release commit (0 means up-to-date, >0 means behind).
        """
        import httpx

        try:
            async with httpx.AsyncClient(timeout=3) as client:
                headers = {"Accept": "application/vnd.github.v3+json"}

                # Fetch latest commits on main
                r = await client.get(
                    "https://api.github.com/repos/ib-api-reloaded/ib_async/commits",
                    params={"per_page": 50},
                    headers=headers,
                )
                if r.status_code != 200:
                    return None

                commits = r.json()
                head_sha = commits[0]["sha"]

                # Fetch remote version from pyproject.toml
                r2 = await client.get(
                    "https://raw.githubusercontent.com/ib-api-reloaded/ib_async/main/pyproject.toml",
                )
                remote_version = ""
                if r2.status_code == 200:
                    for line in r2.text.splitlines():
                        if line.strip().startswith("version"):
                            remote_version = line.split("=", 1)[1].strip().strip('"')
                            break

                # Count how many commits HEAD is ahead of the installed version's
                # release commit (search for "New release: <version>" pattern)
                local_version = ib_async.version.__version__
                commits_ahead = 0
                for i, c in enumerate(commits):
                    msg = c["commit"]["message"]
                    if local_version in msg:
                        commits_ahead = i
                        break
                else:
                    # Release commit not found in recent history — probably far behind
                    commits_ahead = len(commits)

                return (remote_version, head_sha, commits_ahead)
        except Exception:
            return None

    async def runall(self):
        local_version = ib_async.version.__version__
        remote = await self._checkIbAsyncRemote()

        if remote:
            remote_version, head_sha, commits_ahead = remote
            if commits_ahead == 0:
                logger.info(
                    "Using ib_async version: {} (up to date @ {})",
                    local_version,
                    head_sha[:12],
                )
            else:
                logger.warning(
                    "Using ib_async version: {} ({} commit{} behind remote {} @ {})",
                    local_version,
                    commits_ahead if commits_ahead < 50 else "50+",
                    "s" if commits_ahead != 1 else "",
                    remote_version,
                    head_sha[:12],
                )
        else:
            logger.info("Using ib_async version: {}", local_version)

        await self.prepare()
        await self.speak.say(say=f"Starting Client {self.clientId}!")

        while not self.exiting:
            try:
                await self.dorepl()
            except:
                logger.exception("Uncaught exception in repl? Restarting...")
                continue

    async def _resolveAccount(self) -> None:
        """Discover and select account ID if not already set."""
        if self.accountId:
            return

        accounts = self.ib.managedAccounts()
        if not accounts:
            logger.error("No managed accounts from gateway. Check login.")
            sys.exit(1)

        if len(accounts) == 1:
            self.accountId = accounts[0]
            logger.info("Auto-selected account: {}", self.accountId)
        else:
            import questionary

            chosen = await questionary.select(
                "Select IBKR account:",
                choices=accounts,
            ).ask_async()
            if not chosen:
                logger.error("No account selected. Exiting.")
                sys.exit(1)
            self.accountId = chosen
            logger.info("Selected account: {}", self.accountId)

    async def prepare(self):
        # Setup...

        # restore colors (if exists)
        await self.dispatch.runop("colorsload", "", self.opstate)

        # flip to enable/disable verbose ib_insync library logging
        if False:
            import logging

            ib_async.util.logToConsole(logging.INFO)

        # (default is 60 seconds which is too long if connections drop out a lot)
        # NOTE: this doesn't actually do anything except fire a 'timeoutEvent' event
        #       if there is no gateway network traffic for N seconds. It also only fires once,
        #       so we shoould reset setTimeout in the event handler if we are checking for such things.
        self.ib.setTimeout(5)

        # Attach IB events *outside* of the reconnect loop because we don't want to
        # add duplicate event handlers on every reconnect!
        # Note: these are equivalent to the pattern:
        #           lambda row: self.updateSummary(row)
        self.ib.accountSummaryEvent += self.updateSummary
        self.ib.pnlEvent += self.updatePNL
        self.ib.orderStatusEvent += self.updateOrder
        self.ib.errorEvent += self.errorHandler
        self.ib.cancelOrderEvent += self.cancelHandler
        self.ib.commissionReportEvent += self.commissionHandler
        self.ib.newsBulletinEvent += self.newsBHandler
        self.ib.tickNewsEvent += self.newsTHandler

        # We don't use these event types because ib_insync keeps
        # the objects "live updated" in the background, so everytime
        # we read them on a refresh, the values are still valid.
        # self.ib.pnlSingleEvent += self.updatePNLSingle

        # we calculate some live statistics here, and this gets called potentially
        # 5 Hz to 10 Hz because quotes are updated every 250 ms.
        # This event handler also includes a utility for writing the quotes to disk
        # for later backtest handling.
        self.ib.pendingTickersEvent += self.tickersUpdate

        # openOrderEvent is noisy and randomly just re-submits
        # already static order details as new events.
        # self.ib.openOrderEvent += self.orderOpenHandler
        self.ib.execDetailsEvent += self.orderExecuteHandler
        self.ib.positionEvent += self.positionEventHandler

        async def requestMarketData():
            logger.info("Requesting market data...")

            # We used to think this needed to be called before each new market data request, but
            # apparently it works fine now only set once up front?
            # Tell IBKR API to return "last known good quote" if outside
            # of regular market hours instead of giving us bad data.
            self.ib.reqMarketDataType(2)

            # resubscribe to active quotes
            # remove all quotes and re-subscribe to the current quote state
            logger.info("[quotes] Restoring quote state...")
            self.quoteState.clear()

            # Note: always restore snapshot state FIRST so the commands further down don't overwrite
            #       our state with only startup entries.
            with Timer("[quotes :: snapshot] Restored quote state"):
                # restore CLIENT ONLY symbols
                # run the snapshot restore by itself because it hits IBKR rate limits if run with the other restores
                loadedClientDefaultQuotes = await self.dispatch.runop(
                    "qloadsnapshot", "", self.opstate
                )

            # Only load shared quotes if we don't have a local snapshot to restore.
            # (otherwise, we end up loading the global state over our per-client state, so if a client
            #  removes a default symbol, it would _always_ get added back on restart unless we exclude these...
            #  which also make us wonder if we even need the "global" quote namespace anymore ("global on load"
            #  was from before we had per-client saved quote states)).
            # TODO: make this a callable command to "Restore defaults" if we ended up with a busted quote state.
            if not loadedClientDefaultQuotes:
                contracts: list[Stock | Future | Index] = [
                    Stock(sym, "SMART", "USD") for sym in stocks
                ]
                contracts += futures
                contracts += idxs

                with Timer("[quotes :: global] Restored quote state"):
                    # run restore and local contracts qualification concurrently
                    # logger.info("pre=qualified: {}", contracts)

                    # Only attempt global quote restore if the group was previously saved.
                    # On first run (or fresh cache) no global group exists — this is normal,
                    # not an error, so we skip rather than trigger qrestore's error log.
                    if ("quotes", "global") in self.cache:
                        (
                            loadedClientDefaultQuotes,
                            contractsQualified,
                        ) = await asyncio.gather(
                            # restore SHARED global symbols
                            self.dispatch.runop("qrestore", "global", self.opstate),
                            # prepare to restore COMMON symbols
                            self.qualify(*contracts),
                        )
                    else:
                        contractsQualified = await self.qualify(*contracts)
                    # logger.info("post=qualified: {}", contractsQualified)

                with Timer("[quotes :: common] Restored quote state"):
                    for contract in contractsQualified:
                        try:
                            # logger.info("Adding quote for: {} via {}", contract, contracts)
                            self.addQuoteFromContract(contract)
                        except Exception as e:
                            logger.error("Failed to add on startup: {} ({})", contract, e)

            # also, re-attach predicate data readers since any previous live data sources
            # the predicates were attached to no longer exist after the reconnect().
            await asyncio.gather(
                *[
                    self.predicateSetup(prepredicate)
                    for prepredicate in self.ifthenRuntime.predicates.values()
                ]
            )

        async def reconnect():
            # don't reconnect if an exit is requested
            if self.exiting:
                return

            # TODO: we should really find a better way of running this on startup because currently, if the
            #       IBKR gateway/API is down or unreachable, icli will never actually start since we just
            #       get stuck in this "while not connected, attempt to connect" pre-launch condition forever.
            logger.info("Connecting to IBKR API...")
            while True:
                self.connected = False

                logger.info(
                    "Total Updates: {}; Updates since last connect: {}",
                    self.updates,
                    self.updatesReconnect,
                )

                self.updatesReconnect = 0

                try:
                    # NOTE: Client ID *MUST* be 0 to allow modification of
                    #       existing orders (which get "re-bound" with a new
                    #       order id when client 0 connects—but it *only* works
                    #       for client 0)
                    # If you are using the IBKR API, it's best to *never* create
                    # orders outside of the API (TWS, web interface, mobile) because
                    # the API treats non-API-created orders differently.

                    # reset cached states on reconnect so we don't show stale data
                    self.summary.clear()
                    self.pnlSingle.clear()

                    await self.ib.connectAsync(
                        self.host,
                        self.port,
                        clientId=self.clientId,
                        readonly=False,
                        account=self.accountId,
                        fetchFields=ib_async.StartupFetchALL & ~ib_async.StartupFetch.EXECUTIONS,
                    )

                    logger.info(
                        "Connected! Current Request ID for Client {}: {} :: Current Server Version: {}",
                        self.clientId,
                        self.ib.client._reqIdSeq,
                        self.ib.client.serverVersion(),
                    )

                    self.connected = True

                    # Resolve account ID from gateway if not provided at startup.
                    # On reconnects self.accountId is already set, so this is a no-op.
                    await self._resolveAccount()

                    self.ib.reqNewsBulletins(True)

                    # we load executions fully async after the connection happens because
                    # the fetching during connection causes an extra delay we don't need.
                    self.task_create("load executions", self.loadExecutions())

                    # also load market data async for quicker non-blocking startup
                    self.task_create("req mkt data", requestMarketData())

                    # request live updates (well, once per second) of account and position values
                    self.ib.reqPnL(self.accountId)

                    # Subscribe to realtime PnL updates for all positions in account
                    # Note: these are updated once per second per position! nice.
                    # TODO: add this to the account order/filling notifications too.
                    for p in self.ib.portfolio():
                        self.pnlSingle[p.contract.conId] = self.ib.reqPnLSingle(
                            self.accountId, "", p.contract.conId
                        )

                    # run some startup accounting subscriptions concurrently
                    await asyncio.gather(
                        self.ib.reqAccountSummaryAsync(),  # self.ib.reqPnLAsync()
                    )

                    break
                except (
                    TimeoutError,
                    ConnectionRefusedError,
                    ConnectionResetError,
                    OSError,
                    asyncio.CancelledError,
                ) as e:
                    # Don't print full network exceptions for just connection errors
                    logger.error(
                        "[{}] Failed to connect to IB Gateway, trying again... (also check this client id ({}) isn't already connected)",
                        str(e),
                        self.clientId,
                    )
                except:
                    # Do print exception for any unhandled or unexpected errors while connecting.
                    logger.exception("why?")

                try:
                    await asyncio.sleep(3)
                except:
                    logger.warning("Exit requested during sleep. Goodbye.")
                    sys.exit(0)

        try:
            # Run the initial connect in the background so it still starts up at least even if there's
            # no active server running.
            # Note: we need to run the initial connect BLOCKING because the initial connect tell us things like "is this live or sandbox,"
            #       which we use for configuring some other systems (like the history cache and process name and prompt prefix), though
            #       we could just use the same history file for all runs anyway and the prompt would fix itself.
            # asyncio.xreate_task(reconnect())
            await reconnect()
        except SystemExit:
            # do not pass go, do not continue, throw the exit upward
            sys.exit(0)

        customName = ""
        if self.customName:
            customName = f" — {self.customName}"

        set_title(f"{self.levelName().title()} Trader ({self.clientId}){customName}")
        self.ib.disconnectedEvent += lambda: self.task_create("reconnect", reconnect())

    async def buildAndRun(self, text1):
        # 'runnables' is the list of all commands to run after we collect them
        runnables = self.buildRunnablesFromCommandRequest(text1)

        # if no commands, just draw the prompt again
        if not runnables:
            return

        if len(runnables) == 1:
            # if only one command, don't run with an extra Timer() report like we do below
            # with multiple commands (individual commands always report their individual timing)
            return await runnables[0]
        else:
            # only show the "All commands" timer if we have multiple commands to run
            with Timer("All commands"):
                for run in runnables:
                    try:
                        # run a COLLECTIVE COMMAND GROUP we previously created
                        await run
                    except:
                        logger.exception("[{}] Runnable failed?", run)

    async def dorepl(self):
        completer = CommandCompleter(self)
        session: PromptSession = PromptSession(
            history=ThreadedHistory(
                FileHistory(os.path.expanduser(f"~/.tplatcli_ibkr_history.{self.levelName()}"))
            ),
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer,
        )

        app = session.app
        loop = asyncio.get_event_loop()

        async def updateToolbar():
            """Update account balances"""
            try:
                app.invalidate()
            except:
                # network error, don't update anything
                pass

            loop.call_later(
                self.toolbarUpdateInterval, lambda: asyncio.create_task(updateToolbar())
            )

        loop.create_task(updateToolbar())

        # The Command Processing REPL
        while True:
            try:
                # read input from Prompt Toolkit
                text1 = await session.prompt_async(
                    f"{self.levelName()}> ",
                    enable_history_search=True,
                    bottom_toolbar=self.bottomToolbar,
                    # NOTE: refresh interval is handled manually by "call_later(timeout, fn)" at the end of each toolbar update
                    # refresh_interval=3,
                    # mouse_support=True,
                    complete_in_thread=True,
                    complete_while_typing=True,
                    search_ignore_case=True,
                    style=self.toolbarStyle,
                    reserve_space_for_menu=4,
                )

                # log user input to our active logfile(s)
                logger.trace("{}> {}", self.levelName(), text1)

                await self.buildAndRun(text1)
            except KeyboardInterrupt:
                # Control-C pressed. Try again.
                continue
            except EOFError:
                # Control-D pressed
                logger.error("Exiting...")
                self.exiting = True
                break
            except BlockingIOError:
                # this is noisy macOS problem if using a non-fixed
                # uvloop and we don't care, but it will truncate or
                # duplicate your output.
                # solution: don't use uvloop or use a working uvloop
                try:
                    logger.error("FINAL\n")
                except:
                    pass
            except Exception:
                while True:
                    try:
                        logger.exception("Trying...")
                        break
                    except Exception:
                        await asyncio.sleep(1)
                        pass

    def task_create(self, name, coroutine, *args, **kwargs):
        # provide a default US/Eastern timezone to the scheduler unless user provides their own scheduler
        if "scheduler" not in kwargs:
            kwargs = dict(schedule=BGSchedule(tz=USEastern)) | kwargs

        return self.tasks.create(name, coroutine, *args, **kwargs)

    def task_stop(self, task):
        return self.tasks.stop(task)

    def task_stop_id(self, taskId):
        return self.tasks.stopId(taskId)

    def task_report(self):
        return self.tasks.report()

    def stop(self):
        self.exiting = True
        self.ib.disconnect()

    async def setup(self):
        pass
