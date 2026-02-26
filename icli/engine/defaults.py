"""Default quote subscriptions loaded on first run (no saved client state).

These are the instruments that get subscribed automatically when no
per-client quote snapshot exists in the cache.  After first run, the
client's snapshot takes over and these are never consulted again unless
the user explicitly requests a "restore defaults."
"""

from ib_async import Future, Index, Stock

from icli.helpers import FUT_EXP

# ── Stocks ──────────────────────────────────────────────────────────
DEFAULT_STOCKS = ["QQQ", "SPY", "AAPL"]

# ── Futures ─────────────────────────────────────────────────────────
# Futures to exchange mappings:
# https://www.interactivebrokers.com/en/index.php?f=26662
# Note: Use ES and RTY and YM for quotes because higher volume
#       also curiously, MNQ has more volume than NQ?
# Volumes at: https://www.cmegroup.com/trading/equity-index/us-index.html
# ES :: MES
# RTY :: M2K
# YM :: MYM
# NQ :: MNQ
FUTURES_BY_EXCHANGE: dict[str, list[str]] = {
    "CME": ["ES", "RTY", "MNQ", "GBP"],  # "HE"],
    "CBOT": ["YM"],  # , "TN", "ZF"],
    #    "NYMEX": ["GC", "QM"],
}

# ── Indexes ─────────────────────────────────────────────────────────
# Discovered via mainly: https://www.linnsoft.com/support/symbol-guide-ib
# The DJI / DOW / INDU quotes don't work.
# The NDX / COMP quotes require differen't data not included in default packages.
#    Index("COMP", "NASDAQ"),
DEFAULT_INDEXES = [
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


def default_contracts() -> list[Stock | Future | Index]:
    """Build the full list of default contracts to subscribe on first run.

    Note: ContFuture is only for historical data; it can't quote or trade.
    So, all trades must use a manual contract month (quarterly).
    TODO: we should be consuming a better expiration date system because some
          futures expire end-of-month (interest rates), others quarterly (indexes), etc.
    """
    contracts: list[Stock | Future | Index] = [
        Stock(sym, "SMART", "USD") for sym in DEFAULT_STOCKS
    ]

    contracts += [
        Future(
            symbol=sym,
            lastTradeDateOrContractMonth=FUT_EXP or "",
            exchange=x,
            currency="USD",
        )
        for x, syms in FUTURES_BY_EXCHANGE.items()
        for sym in syms
    ]

    contracts += DEFAULT_INDEXES

    return contracts
