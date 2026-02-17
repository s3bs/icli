"""Contract creation, parsing, and utility functions for ib_async contracts."""

from __future__ import annotations

import asyncio
import datetime
import os

import dateutil.parser
import httpx
import ib_async
from cachetools import cached
from dataclasses import dataclass, field
from decimal import Decimal
from ib_async import (
    Bag,
    Bond,
    CFD,
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
    Order,
    Stock,
    Trade,
    Warrant,
)
from loguru import logger

from icli.engine.exchanges import FUTS_EXCHANGE
from icli.engine.primitives import FUTS_MONTH_MAPPING, convert_futures_code

# Module-level config — must be set before contractForName() is called.
# Typically set during app startup from helpers.FUT_EXP.
FUT_EXP: str = ""


@dataclass(slots=True, frozen=True)
class TradeOrder:
    """Just holde a trade/order combination pair for results reporting."""

    trade: Trade
    order: Order


@dataclass(slots=True, frozen=True)
class FullOrderPlacementRecord:
    limit: TradeOrder
    profit: TradeOrder | None = None
    loss: TradeOrder | None = None


def nameForContract(contract: Contract, cdb: dict[int, Contract] | None = None) -> str:
    """Generate a text description for a contract we can re-parse into a contract again.

    The goal here is to provide a more user-readable contract description for logs or metadata details
    than just printing the underlying contract ids everywhere.
    """

    if isinstance(contract, Option) or contract.secType == "OPT":
        return contract.localSymbol.replace(" ", "")

    if contract.secType in {"FOP", "EC"}:
        # Note: this could also technically be just "/" + localSymbol.replace(" ", "") becase we can read futures option syntax now too
        return f"/{contract.symbol}{contract.lastTradeDateOrContractMonth[2:]}{contract.right}{int(contract.strike * 1000):08}-{contract.tradingClass}"

    if isinstance(contract, Bag):
        result = []

        # TODO: if we want more descriptive names (instead of contract ids) this needs to be attached to a place where we can read cached contracts...
        if cdb:
            # if we have the contract db available, we can generate more meaningful details
            for leg in contract.comboLegs:
                foundLeg = cdb.get(leg.conId)
                result.extend(
                    [
                        leg.action,
                        str(leg.ratio),
                        nameForContract(foundLeg) if foundLeg else str(leg.conId),
                    ]
                )
        else:
            for leg in contract.comboLegs:
                result.extend([leg.action, str(leg.ratio), str(leg.conId)])

        return " ".join(result)

    if isinstance(contract, Future):
        return f"/{contract.localSymbol}"

    if isinstance(contract, Stock):
        return contract.localSymbol

    if isinstance(contract, Index):
        return f"I:{contract.localSymbol}"

    if isinstance(contract, Forex):
        return f"F:{contract.localSymbol}"

    if isinstance(contract, Crypto):
        return f"C:{contract.localSymbol}"

    assert None, f"Unexpected contract for name creation? {contract=}"


def contractForName(sym, exchange="SMART", currency="USD") -> Contract:
    """Convert a single text symbol data format into an ib_insync contract."""

    sym = sym.upper()

    contract: Contract

    # TODO: how to specify warrants/equity options/future options/spreads/bonds/tbills/etc?
    if sym.isnumeric() and len(sym) > 3:
        # Allow exact contract IDs to generate an abstract contract we then qualify to a more concrete type.
        # (also only accept numbers longer than 3 digits so a typo of positions like :13 as 13 doesn't become a contract id by mistake)
        contract = Contract(conId=int(sym))
    elif sym.startswith("/") or sym.startswith(","):
        # (in some places we use ',' as a futures prefix instead of '/' if we had to serialize out a symbol name as a file...)
        sym = sym[1:]

        # Check if this symbol is directly in our futures lookup map (or two minus the end of the end is a month/year indicator)...
        inFutureMap = sym in FUTS_EXCHANGE or (
            len(sym) >= 4 and sym[:-2] in FUTS_EXCHANGE
        )

        # first check if this is a CME syntax for FOP at lengths 9, 10, 13 (/EWZ4P4000) (/E4AU4C4700) or EC (/ECESZ431P4000)
        # Also consider: /GCZ4C2665 or /GCZ4C2665-EC or /ECGCZ4C2665 or /ECBTCZ4C63500 or /BTCZ4C63500-EC
        if (not inFutureMap) and (9 <= len(sym) <= 13):
            if "-" in sym:
                sympart, tradingClass = sym.split("-")
            else:
                sympart = sym
                tradingClass = None

            halfsearch = len(sympart) // 2 - 1
            try:
                mid = sympart.index("C", halfsearch)
            except:
                mid = sympart.index("P", halfsearch)

            strike = sympart[mid:]
            symbol = sympart[: -len(sympart) + mid]

            if not tradingClass:
                tradingClass = symbol[:-2]

            localSymbol = f"{symbol} {strike}"

            if symbol.startswith("EC"):
                # remove EC prefix and date code
                exchangeSymbol = symbol[2:-2]
            else:
                exchangeSymbol = tradingClass

            fxchg = FUTS_EXCHANGE.get(exchangeSymbol)

            contract = FuturesOption(
                exchange=fxchg.exchange if fxchg else "CME",
                localSymbol=localSymbol,
                tradingClass=tradingClass,
            )
        elif len(sym) > 15:
            # else, use our custom hack format allowing CME futures options to look like OCC options syntax
            if "-" in sym:
                sym, tradingClass = sym.split("-")
            else:
                tradingClass = ""

            # Is Future Option! FOP!
            symbol = sym[:-15]

            body = sym[-15:]
            date = "20" + body[:6]
            right = body[-9]  # 'C'

            if right not in {"C", "P"}:
                raise Exception(f"Invalid option format right: {right} in {sym}")

            price = int(body[-8:])  # 320000 (leading 0s get trimmed automatically)
            strike = price / 1000  # 320.0

            # fix up if has date code embedded in the symbol
            if symbol[-1].isdigit():
                symbol = symbol[:-2]

            fxchg = FUTS_EXCHANGE[symbol]
            contract = FuturesOption(
                currency=currency,
                symbol=fxchg.symbol,
                exchange=fxchg.exchange,
                strike=strike,
                right=right,
                lastTradeDateOrContractMonth=date,
                tradingClass=tradingClass,
            )
        else:
            # else, is regular future (or quote-like thing we treat as a future)

            # our symbol lookup table is the unqualified contract name like "ES" but
            # when trading, the month and year gets added like "ESZ3", so if we have
            # a symbol ending in a digit here, remove the "expiration/year" designation
            # from the string to lookup the actual name.
            dateForContract = FUT_EXP
            if sym[-1].isdigit() and sym[-2] in FUTS_MONTH_MAPPING:
                fullsym = sym
                sym = sym[:-2]

                # if we have an EXACT date syntax requested, populate it instead of the default "current next main future expiration quarter"
                dateForContract = convert_futures_code(fullsym[-2:])

            try:
                fxchg = FUTS_EXCHANGE[sym]
            except:
                logger.error("[{}] Symbol not in our futures database mapping!", sym)
                raise ValueError(f"Unknown future mapping requested: {sym}")

            if dateForContract == FUT_EXP and fxchg.name.endswith("Yield"):
                # "Yield" products expire MONTHLY and not quarterly, so do big end-of-month smash here
                # (if you want a *specific* forward month (usually only listed current and next 2 months at once), you
                # can use the more common futures codes like /10YN4 to mean July 2024 etc. By default you'll get THIS MONTH expiry.
                # TODO: technically the "is Yield type" should be a property of the futures mapping instead of this
                #       more hacky "if name ends in Yield, it's a yield quote, so use monthly expirations..."
                now = datetime.datetime.now().date()
                dateForContract = f"{now.year}{now.month:02}"

            isweeklyvix = sym[:2] == "VX"

            # We need some extra consideration for populating weekly vix contracts because
            # we reference them by local symbol for trading like /VX17, but they are still "symbol VIX" and
            # the _trading class_ is the input symbol requested (VX17).
            # Also, to run this properly, you must manually override the date spec with futures month/year symbols
            # so you bind the correct month/year to the weekly expiration like /VX17J5 (could be more automated, but isn't yet).
            # See the "Contract Expirations" header here for more details: https://www.cboe.com/tradable_products/vix/vix_futures/specifications/
            if isweeklyvix:
                contract = Future(
                    currency=currency,
                    symbol=fxchg.symbol,
                    exchange=fxchg.exchange,
                    lastTradeDateOrContractMonth=dateForContract,
                    tradingClass=sym,
                )
            else:
                contract = Future(
                    currency=currency,
                    symbol=fxchg.symbol,
                    exchange=fxchg.exchange,
                    # if it looks like our symbol ends in a futures date code, convert the futures
                    # date code to IBKR date format. else, use our default continuous next-expiry futures calculation.
                    lastTradeDateOrContractMonth=dateForContract,
                    tradingClass="",
                )
    elif len(sym) > 15:
        # looks like: COIN210430C00320000
        symbol = sym[:-15]  # COIN
        body = sym[-15:]  # 210430C00320000

        # Note: Not year 2100+ compliant!
        # 20 + YY + MM + DD
        date = "20" + body[:6]

        right = body[-9]  # 'C'

        if right not in {"C", "P"}:
            raise Exception(f"Invalid option format right: {right} in {sym}")

        price = int(body[-8:])  # 320000 (leading 0s get trimmed automatically)
        strike = price / 1000  # 320.0

        contract = Option(
            symbol=symbol,
            lastTradeDateOrContractMonth=date,
            strike=strike,
            right=right,
            exchange=exchange,
            currency=currency,
            # also pass in tradingClass so options like SPXW220915P03850000 work
            # directly instead of having IBKR guess if we want SPX or SPXW.
            # for all other symbols the underlying trading class doesn't alter
            # behavior (and we don't allow users to specify extra classes yet
            # like if you want to trade on fragemented options chains after
            # reverse splits, etc).
            tradingClass=symbol,
        )
    else:
        # if symbol has a : we are namespacing by type:
        #   - W: - warrant
        #   - C: - crypto
        #   - F: - forex
        #   - B: - bond
        #   - S: - stock (or no contract namespace)
        #   - I: - an index value (VIX, VIN, SPX, etc)
        #   - K: - a direct IBKR contract id to populate into a contract
        # Note: futures and future options are prefixed with /
        #       equity options are the full OCC symbol with no prefix
        namespaceParts = sym.split(":")
        if len(namespaceParts) > 1:
            contractNamespace, symbol = namespaceParts
            if contractNamespace == "W":
                # TODO: needs option-like strike, right, multiplier, contract date spec too
                contract = Warrant(
                    symbol=symbol, exchange=exchange, currency=currency, right="C"
                )
                # Requires all the details like:
                # contract = Warrant(conId=504262528, symbol='BGRY', lastTradeDateOrContractMonth='20261210', strike=11.5, right='C', multiplier='1', primaryExchange='NASDAQ', currency='USD', localSymbol='BGRYW', tradingClass='BGRY')
            elif contractNamespace == "C":
                contract = Crypto(symbol=symbol, exchange="PAXOS", currency=currency)
            elif contractNamespace == "B":
                contract = Bond(symbol=symbol, exchange=exchange, currency=currency)
            elif contractNamespace == "S":
                contract = Stock(symbol, exchange, currency)
            elif contractNamespace == "I":
                # this appears to work fine without specifying the full Index(symbol, exchange) format
                contract = Index(symbol)
            elif contractNamespace == "F":
                # things like F:GBPUSD F:EURUSD (but not F:JPYUSD or F:RMBUSD shrug)
                # also remember C: is CRYPTO not "CURRENCY," so currencies are F for FOREX
                contract = Forex(symbol)
            elif contractNamespace == "CFD":
                # things like CFD:XAUUSD
                contract = CFD(symbol)
            elif contractNamespace == "K":
                contract = Contract(conId=int(symbol))
            else:
                logger.error("Invalid contract type: {}", contractNamespace)
                raise ValueError(
                    f"Invalid contract type requested: {contractNamespace}"
                )
        else:
            # TODO: warrants, bonds, bills, etc
            contract = Stock(sym, exchange, currency)

    return contract


def contractToSymbolDescriptor(contract) -> str:
    """Extracts the class name of a contract to return className-Symbol globally unique string"""

    # We need the input contract request to generate a strong enough primary key where it doesn't conflict
    # with other contracts. So we can't just do "Class-Symbol" because every option would be e.g. "Option-MSFT".
    # NOTE: just remember to only include "user lookup fields" in the primary key. The user isn't populating things like
    #       'tradingClass' so we don't want to use it in the primary key even though it does get populated after the qualify.
    # The cache storage is looked up using UNQUALIFIED contracts then stored using the QUALIFIED contracts.
    # Also note: we use 'contract.secType' instead of 'contract.__class__.__name__' because some contracts don't have exact
    #            class types, but we end up populating similar class types with different secTypes underneath.
    parts = (
        contract.secType,
        contract.localSymbol,
        contract.symbol,
        contract.lastTradeDateOrContractMonth or "NoDate",
        contract.right or "NoRight",
        str(float(contract.strike or 0) or "NoStrike"),
        contract.tradingClass or "NoTradingClass",
    )
    return "-".join(parts)


def contractFromTypeId(contractType: str, conId: int) -> Contract:
    """Consume a previously extract contract class name and conId to generate a new proper concrete subclass of Contract"""
    match contractType:
        case "Bag":
            return Bag(conId=conId)
        case "Bond":
            return Bond(conId=conId)
        case "CFD":
            return CFD(conId=conId)
        case "Commodity":
            return Commodity(conId=conId)
        case "ContFuture":
            return ContFuture(conId=conId)
        case "Crypto":
            return Crypto(conId=conId)
        case "Forex":
            return Forex(conId=conId)
        case "Future":
            return Future(conId=conId)
        case "FuturesOption":
            return FuturesOption(conId=conId)
        case "Index":
            return Index(conId=conId)
        case "MutualFund":
            return MutualFund(conId=conId)
        case "Option":
            return Option(conId=conId)
        case "Stock":
            return Stock(conId=conId)
        case "Warrant":
            return Warrant(conId=conId)
        case _:
            raise ValueError(f"Unsupported contract type: {contractType}")


def contractFromSymbolDescriptor(contractType: str, symbol: str):
    match contractType:
        case "Bag":
            return Bag(symbol=symbol)
        case "Bond":
            return Bond(symbol=symbol)
        case "CFD":
            return CFD(symbol=symbol)
        case "Commodity":
            return Commodity(symbol=symbol)
        case "ContFuture":
            return ContFuture(symbol=symbol)
        case "Crypto":
            return Crypto(symbol=symbol)
        case "Forex":
            return Forex(symbol=symbol)
        case "Future":
            return Future(symbol=symbol)
        case "FuturesOption":
            return FuturesOption(symbol=symbol)
        case "Index":
            return Index(symbol=symbol)
        case "MutualFund":
            return MutualFund(symbol=symbol)
        case "Option":
            return Option(symbol=symbol)
        case "Stock":
            return Stock(symbol=symbol)
        case "Warrant":
            return Warrant(symbol=symbol)
        case _:
            raise ValueError(f"Unsupported contract type: {contractType}")


def tickFieldsForContract(contract) -> str:
    # Available fields from:
    # https://interactivebrokers.github.io/tws-api/tick_types.html
    # NOTE: the number to use here is the 'Generic tick required' number and NOT the 'Tick id' number.
    extraFields = [
        # start with VWAP and volume data requested everywhere
        233,
        # also add "volume per minute"
        295,
        # also add "trades per minute"
        294,
    ]

    # There is also "fundamentals" as tick 258 but it returns things like this which isn't useful for us because
    # it's just reporting on historical financial reports:
    #     fundamentalRatios=FundamentalRatios(TTMNPMGN=35.20226, NLOW=274.38, TTMPRCFPS=18.29654, TTMGROSMGN=81.49056, TTMCFSHR=25.00352, QCURRATIO=2.83036, TTMREV=149783, TTMINVTURN=nan, TTMOPMGN=39.25479, TTMPR2REV=8.03502, AEPSNORM=16.18888, TTMNIPEREM=783264.3, EPSCHNGYR=82.25327, TTMPRFCFPS=25.55114, TTMRECTURN=11.08847, TTMPTMGN=40.1354, QCSHPS=22.92933, TTMFCF=47102, LATESTADATE='2024-06-30', APTMGNPCT=35.15737, AEBTNORM=50880, TTMNIAC=52727, NetDebt_I=-39691, PRYTDPCTR=23.60872, TTMEBITD=73664, AFEEPSNTM=22.401, PR2TANBK=8.90664, EPSTRENDGR=14.79464, QTOTD2EQ=11.73045, TTMFCFSHR=17.9044, QBVPS=61.88827, NPRICE=475.73, YLD5YAVG=nan, PR13WKPCT=2.15813, PR52WKPCT=53.10076, REVTRENDGR=19.29375, AROAPCT=19.10341, TTMEPSXCLX=20.0446, QTANBVPS=53.34583, PRICE2BK=7.68692, MKTCAP=1203510, TTMPAYRAT=4.84951, TTMINTCOV=nan, TTMREVCHG=24.27752, TTMROAPCT=24.13544, TTMROEPCT=36.26391, TTMREVPERE=2225040, APENORM=29.38622, TTMROIPCT=27.75098, REVCHNGYR=22.10069, CURRENCY='USD', DIVGRPCT=nan, TTMEPSCHG=136.651, PEEXCLXOR=23.73357, QQUICKRATI=nan, TTMREVPS=56.93547, BETA=1.17755, TTMEBT=60116, ADIV5YAVG=nan, ANIACNORM=42560.56, PR1WKPCT=2.15155, QLTD2EQ=11.73045, NHIG=542.81, PR4WKPCT=-10.12431)

    if isinstance(contract, Stock):
        # 104:
        # "The 30-day historical volatility (currently for stocks)."
        # 106:
        # "The IB 30-day volatility is the at-market volatility estimated
        #  for a maturity thirty calendar days forward of the current trading
        #  day, and is based on option prices from two consecutive expiration
        #  months."
        # 236:
        # "Number of shares available to short"
        # "Shortable: < 1.5, not availabe
        #             > 1.5, if shares can be located
        #             > 2.5, enough shares are available (>= 1k)"
        # 595: Stock volume averaged over 3 minutes, 5 minutes, 10 minutes.
        extraFields += [104, 106, 236, 595]

    # yeah, the API wants a CSV for the tick list. sigh.
    tickFields = ",".join([str(x) for x in extraFields])

    # logger.info("[{}] Sending fields: {}", contract, tickFields)
    return tickFields


def parseContractOptionFields(contract, d):
    # logger.info("contract is: {}", o.contract)
    if isinstance(contract, (Warrant, Option, FuturesOption)):
        try:
            d["date"] = dateutil.parser.parse(
                contract.lastTradeDateOrContractMonth
            ).date()  # type: ignore
        except:
            logger.error("Row didn't have a good date? {}", contract)
            return

        d["strike"] = contract.strike
        d["PC"] = contract.right
    else:
        # populate columns for non-contracts/warrants too so the final
        # column-order generator still works.
        d["date"] = None
        d["strike"] = None
        d["PC"] = None


def strFromPositionRow(o):
    """Return string describing an order (for quick review when canceling orders).

    As always, 'averageCost' field is for some reason the cost-per-contract field while
    'marketPrice' is the cost per share, so we manually convert it into the expected
    cost-per-share average cost for display."""

    useAvgCost = o.averageCost / float(o.contract.multiplier or 1)
    digits = 2  # TODO: move this so we can read digits

    return f"{o.contract.localSymbol} :: {o.contract.secType} {o.position:,.{digits}f} MKT:{o.marketPrice:,.{digits}f} CB:{useAvgCost:,.{digits}f} :: {o.contract.conId}"


def isset(x: float | Decimal | None) -> bool:
    """Sadly, IBKR/ib_insync API uses FLOAT_MAX to mean "number is unset" instead of
    letting numeric fields be Optional[float] where we could just check for None.

    So we have to directly compare against another value to see if a returned float
    is a _set_ value or just a placeholder for the default value. le sigh."""

    # the round hack is because sometimes we convert the floats to 2 digits which makes them rather... smaller
    return (
        (x is not None)
        and (x != ib_async.util.UNSET_DOUBLE)
        and (x != round(ib_async.util.UNSET_DOUBLE, 2))
    )


# Note: we use a custom key here instead of just letting the
#       contract hash itself (hash(contract)) because the contract is hashed only
#       by id, but sometimes we have un-populated contracts like Contract(id=X) which
#       then generates an invalid lookup key result because the name is missing.
#       So we want to cache contracts based on their full details so we return different results
#       for fully qualified contract details versus partial contract details.
# TODO: though, if we make Contract types immutable, then we could just do id(contract) as a key.
@cached(cache={}, key=lambda x: x)  # contractToSymbolDescriptor(x))
def lookupKey(contract):
    """Given a contract, return something we can use as a lookup key.

    Needs some tricks here because spreads don't have a built-in
    one dimensional representation."""

    # if this is a spread, there's no single symbol to use as an identifier, so generate a synthetic description instead
    if isinstance(contract, Bag):
        # Generate a custom tuple representation we can use as immutable dict keys.
        # Only the ratio, side/action, and contract id matters when defining collective spread definitions.
        return tuple(
            [
                (b.ratio, b.action, b.conId)
                for b in sorted(
                    contract.comboLegs, key=lambda x: (x.ratio, x.action, x.conId)
                )
            ]
        )

    # else, is not a spread so we can use regular in-contract symbols
    if contract.localSymbol:
        return contract.localSymbol.replace(" ", "")

    # else, if a regular symbol but DOESN'T have a .localSymbol (means
    # we added the quote from a contract without first qualifying it,
    # which works, it's just missing extra details)
    if contract.symbol:
        return contract.symbol

    logger.error("Your contract doesn't have a symbol? Bad contract: {}", contract)

    return None


@dataclass(slots=True)
class CompleteTradeNotification:
    """A wrapper to hold a Trade object and a notifier we can attach to.

    Used by our automated order placement logic to get updates when an _entire_ order
    completes so we can continue scaling in or out of the next steps.
    """

    trade: Trade | None = None
    event: asyncio.Event = field(default_factory=asyncio.Event)

    async def orderComplete(self):
        """Wait for the event to trigger then clear it sicne we woke up."""
        await self.event.wait()
        self.event.clear()

    def set(self):
        self.event.set()


async def getExpirationsFromTradier(symbol: str):
    """Fetch option chain expirations and strikes from Tradier.

    I'm tired of IBKR data causing minutes of blocking delays during operations, so let's just use other providers instead.
    """

    # TODO: should we make `token` a parameter instead?
    token = os.getenv("TRADIER_KEY")
    if not token:
        raise Exception("Tradier Token Needed for Tradier Data Fetching!")

    # https://documentation.tradier.com/brokerage-api/markets/get-options-expirations
    async with httpx.AsyncClient() as client:
        got = await client.get(
            "https://api.tradier.com/v1/markets/options/expirations",
            params={
                "symbol": symbol,
                "includeAllRoots": "true",
                "strikes": "true",
                "contractSize": "true",
                "expirationType": "true",
            },
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )

    if not got:
        return None

    found = got.json()

    # API returns map of {expiration: strikes}
    #               e.g. {"20240816": [100, 101, 102, 103, 104, ...], ...}
    result = {}

    # convert returned tradier JSON in to a format compat with how we store IBKR data
    expirations = found["expirations"]

    if not expirations:
        return None

    for date in expirations["expiration"]:
        strikes = date["strikes"]["strike"]

        # fix awful tradier data formats where if only one element exists, it is a scalar and not a list
        # like all the other fields.
        if not isinstance(strikes, list):
            strikes = [strikes]

        # tradier sometimes has bad data where they list only 1-2 strikes for an expiration date?
        # kinda odd, so just skip those dates entirely.
        if len(strikes) < 5:
            continue

        # convert date to IBKR format with no dashes as YYYYMMDD
        result[date["date"].replace("-", "")] = strikes

    return result
