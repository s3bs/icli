"""Command: qquote

Category: Live Market Quotes
"""

import asyncio
import math

from dataclasses import dataclass, field

import pandas as pd
from ib_async import Stock, Index
from loguru import logger
from mutil.dispatch import DArg
from mutil.frame import printFrame

from icli.cmds.base import IOp, command
from icli.engine.contracts import contractForName, tickFieldsForContract


def is_ticker_ready(ticker, contract) -> tuple[bool, str]:
    """Check if a ticker has sufficient data for its contract type.

    Uses NaN-aware checks — ib_async initializes all numeric Ticker fields
    to float('nan'), and bool(nan) is True, so we must use hasBidAsk() or
    math.isnan() rather than truthiness.

    Returns:
        (is_ready, status_message)
    """
    has_bid_ask = ticker.hasBidAsk()

    if isinstance(contract, Stock):
        # Stocks get ticks 104 (histVol), 106 (impliedVol), 236 (shortable)
        # from tickFieldsForContract. Wait for all of them.
        missing = []
        if not has_bid_ask:
            missing.append("bid/ask")
        if math.isnan(ticker.impliedVolatility):
            missing.append("impliedVolatility")
        if math.isnan(ticker.histVolatility):
            missing.append("histVolatility")
        if math.isnan(ticker.shortable):
            missing.append("shortable")

        if missing:
            return (False, f"missing: {', '.join(missing)}")
        return (True, "ready")

    elif isinstance(contract, Index):
        # Indexes: prefer bid/ask (VIX/VIN/VIF have CBOE-published bid/ask).
        # Fallback: calculation indexes (TICK-NYSE, ADD-NYSE) may only have
        # a computed last or close value with no tradeable bid/ask.
        if has_bid_ask:
            return (True, "ready")
        if not math.isnan(ticker.last) or not math.isnan(ticker.close):
            return (True, "ready (last/close only)")
        return (False, "waiting for data")

    else:
        # Futures, Options, FuturesOptions, Forex, Crypto, CFD, Bond,
        # Warrant, Bag — bid/ask is sufficient. Volatility ticks are not
        # requested for these types and will never arrive.
        if has_bid_ask:
            return (True, "ready")
        return (False, "waiting for bid/ask")


@command(names=["qquote"])
@dataclass
class IOpQQuote(IOp):
    """Quick Quote: Run a temporary quote request then print results when data arrives."""

    symbols: list[str] = field(init=False)

    def argmap(self) -> list[DArg]:
        return [DArg("*symbols")]

    async def run(self):
        if not self.symbols:
            logger.error("No symbols requested?")
            return

        contracts = [contractForName(sym) for sym in self.symbols]
        contracts = await self.state.qualify(*contracts)

        if not all(c.conId for c in contracts):
            logger.error("Not all contracts reported successful lookup!")
            logger.error(contracts)
            return

        # IBKR populates each quote data field async, so even after we
        # "request market data," it can take 5-10 seconds for all the fields
        # to become populated (if they even populate at all).
        tickers = []
        logger.info(
            "Requesting tickers for {}",
            ", ".join([c.localSymbol.replace(" ", "") or c.symbol for c in contracts]),
        )

        # TODO: check if we are subscribed to live quotes already and use live quotes
        #       instead of re-subscribing (also note to _not_ unsubscribe from already-existing
        #       live quotes if we merge them into the tickers check here too).
        for contract in contracts:
            # Request quotes with metadata fields populated
            # (note: metadata is only populated using "live" endpoints,
            #  so we can't use the self-canceling "11 second snapshot" parameter)
            tf = tickFieldsForContract(contract)
            # logger.info("[{}] Tick Fields: {}", contract, tf)
            tickers.append(self.ib.reqMktData(contract, tf))

        ATTEMPT_LIMIT = 10
        for i in range(ATTEMPT_LIMIT):
            statuses = [
                is_ticker_ready(ticker, contract)
                for ticker, contract in zip(tickers, contracts)
            ]

            if all(is_ready for is_ready, _ in statuses):
                break

            pending_info = [
                f"{contract.symbol} ({contract.secType}): {status_msg}"
                for contract, (is_ready, status_msg) in zip(contracts, statuses)
                if not is_ready
            ]

            logger.warning(
                "Waiting for data to arrive... (attempt {} of {})\n  Pending: {}",
                i,
                ATTEMPT_LIMIT,
                " | ".join(pending_info),
            )
            await asyncio.sleep(1.33)
        else:
            incomplete = [
                (contract.symbol, contract.secType)
                for contract, (is_ready, _) in zip(contracts, statuses)
                if not is_ready
            ]
            logger.warning(
                "Partial data for {} contract(s) after {} attempts: {}",
                len(incomplete),
                ATTEMPT_LIMIT,
                ", ".join([f"{sym} ({typ})" for sym, typ in incomplete]),
            )

        # logger.info("Got tickers: {}", pp.pformat(tickers))

        df = pd.DataFrame(tickers)

        # extract contract data from nested object pandas would otherwise
        # just convert to a blob of json text.
        contractframe = pd.DataFrame([t.contract for t in tickers])
        contractseries = contractframe["symbol secType conId".split()]

        # NB: 'halted' statuses are:
        # -1 Halted status not available.
        # 0 Not halted.
        # 1 General halt. regulatory reasons.
        # 2 Volatility halt.
        dfSlice = df[
            """bid bidSize
               ask askSize
               last lastSize
               volume open high low close vwap
               halted shortable shortableShares
               histVolatility impliedVolatility""".split()
        ]

        # attach inner name data to data rows since it's a nested field thing
        # this 'concat' works because the row index ids match across the contracts
        # and the regular ticks we extracted.
        dfConcat = pd.concat([contractseries, dfSlice], axis=1)

        printFrame(dfConcat)

        # all done!
        for contract in contracts:
            self.ib.cancelMktData(contract)
