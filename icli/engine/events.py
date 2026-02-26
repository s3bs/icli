"""IBKR event handler methods extracted from IBKRCmdlineApp (cli.py).

Handles routing of ib_async callbacks to their internal state updates:
order status, error handling, commissions, news, executions, positions,
account summary, P&L, and ticker updates.
"""
from __future__ import annotations

import asyncio
import datetime
import math
from decimal import Decimal
from typing import TYPE_CHECKING, Final

from loguru import logger
from ib_async import Bag, Contract, NewsBulletin, NewsTick, Position, Trade

from icli.engine.contracts import lookupKey
from icli.engine.primitives import FillReport, fmtmoney
from icli.engine.calendar import readableHTML
import icli.engine.orders as orders

import tradeapis.ifthen as ifthen

if TYPE_CHECKING:
    from tradeapis.ordermgr import Trade as OrderMgrTrade


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


class IBEventRouter:
    """Routes ib_async event callbacks to their respective state-mutation handlers.

    Extracted from IBKRCmdlineApp so these handlers can be tested without a full
    app instance.

    Parameters
    ----------
    ib:
        Active ib_async IB connection.
    quoteState:
        Shared dict mapping symbol key -> ITicker.
    summary:
        Shared dict for account summary tag/value pairs.
    accountStatus:
        Shared dict for numeric account status fields.
    pnlSingle:
        Shared dict mapping conId -> PnLSingle subscription object.
    iposition:
        Shared dict mapping contract -> IPosition.
    fillers:
        defaultdict(Event) used for order-fill notification.
    ordermgr:
        OrderMgr instance tracking executed trades.
    speak:
        Text-to-speech helper (has `.url` attribute and `.say()` coroutine).
    duplicateMessageHandler:
        Object with `.handle_message(message, log_func)` for deduplicating errors.
    ifthenRuntime:
        ifthen runtime object with `.check(quotekey)` method.
    conIdCache:
        Contract id lookup cache.
    contractIdsToQuoteKeysMappings:
        Shared dict mapping conId -> quote key string.
    app:
        Back-reference to IBKRCmdlineApp for cross-component calls.
    """

    def __init__(
        self,
        ib,
        quoteState,
        summary,
        accountStatus,
        pnlSingle,
        iposition,
        fillers,
        ordermgr,
        speak,
        duplicateMessageHandler,
        ifthenRuntime,
        conIdCache,
        contractIdsToQuoteKeysMappings,
        app=None,
    ):
        self.ib = ib
        self.quoteState = quoteState
        self.summary = summary
        self.accountStatus = accountStatus
        self.pnlSingle = pnlSingle
        self.iposition = iposition
        self.fillers = fillers
        self.ordermgr = ordermgr
        self.speak = speak
        self.duplicateMessageHandler = duplicateMessageHandler
        self.ifthenRuntime = ifthenRuntime
        self.conIdCache = conIdCache
        self.contractIdsToQuoteKeysMappings = contractIdsToQuoteKeysMappings
        self._app = app  # back-reference for cross-component calls

    # ------------------------------------------------------------------
    # Order lifecycle handlers
    # ------------------------------------------------------------------

    def updateOrder(self, trade: Trade):
        # Only print update if this is regular runtime and not
        # the "load all trades on startup" cycle
        if not self._app.connected:
            return

        logger.warning(
            "[{} :: {} :: {}] Order update: {}",
            trade.orderStatus.orderId,
            trade.orderStatus.status,
            trade.contract.localSymbol,
            trade,
        )

        # notify any watchers only when order is 100% filled
        if trade.log[-1].status == "Filled" and trade.orderStatus.remaining == 0:
            f = self.fillers[trade.contract]
            f.trade = trade
            f.set()

    def errorHandler(self, reqId, errorCode, errorString, contract):
        # Official error code list:
        # https://interactivebrokers.github.io/tws-api/message_codes.html
        # (note: not all message codes are listed in their API docs, of course)
        if errorCode in {1102, 2104, 2108, 2106, 2107, 2119, 2152, 2158}:
            # non-error status codes on startup or informational messages during running.
            # we ignore reqId here because it is either always -1 or a data request id (but never an order id)
            logger.info(
                "API Status [code {}]: {}",
                errorCode,
                errorString,
            )
        else:
            # Instead of printing these errors directly, we pass them through a deduplication
            # filter because sometimes we get unlimited repeated error messages (which aren't
            # actually errors) and we want to suppress them to only one repeated error update
            # every 30 seconds instead of 1 error per second continuously.
            msg = "{} [code {}]: {}{}".format(
                f"Order Error [orderId {reqId}]"
                if (reqId > 0 and errorCode not in {321, 366})
                else "API Error",
                errorCode,
                readableHTML(errorString),
                f" for {contract}" if contract else "",
            )

            self.duplicateMessageHandler.handle_message(
                message=msg, log_func=logger.error
            )

    def cancelHandler(self, err):
        logger.warning("Order canceled: {}", err)

    def commissionHandler(self, trade, fill, report):
        # Only report commissions if not bulk loading them as a refresh
        # (the bulk load API causes the event handler to fire for each historical fill)
        if self._app.loadingCommissions:
            logger.warning(
                "Ignoring commission because bulk loading history: [{:>2} :: {} {:>7.2f} of {:>7.2f} :: {}]",
                fill.execution.clientId,
                fill.execution.side,
                fill.execution.shares,
                fill.execution.cumQty,
                fill.contract.localSymbol,
            )
            return

        # TODO: different sounds if PNL is a loss?
        #       different sounds for big wins vs. big losses?
        #       different sounds for commission credit vs. large commission fee?
        # TODO: disable audio for algo trades?

        if self.speak.url:
            # using "BOT" and "SLD" as real words because the text-to-speech was pronouncing "SLD" as individual letters "S-L-D"
            side = "bought" if fill.execution.side == "BOT" else "sold"

            fillQty = f"{fill.contract.localSymbol} ({side} {int(fill.execution.shares)} (for {int(fill.execution.cumQty)} of {int(trade.order.totalQuantity)}))"

            #  This triggers on a successful close of a position (TODO: need to fill out more details)
            if fill.commissionReport.realizedPNL:
                PorL = "profit" if fill.commissionReport.realizedPNL >= 0 else "loss"

                content = f"CLOSED: {trade.orderStatus.status} FOR {fillQty} ({PorL} ${round(fill.commissionReport.realizedPNL, 2):,})"
            else:
                # We notify about orders HERE instead of in 'orderExecuteHandler()' because HERE we have details about filled/canceled for
                # the status, where 'orderExecuteHandler()' always just has status of "Submitted" when an execution happens (also with no price details) which isn't as useful.
                content = f"OPENED: {trade.orderStatus.status} FOR {fillQty} (commission {fmtmoney(fill.commissionReport.commission)})"

            self._app.task_create(content, self.speak.say(say=content))

        logger.warning(
            "[{} :: {} :: {}] Order {} commission: {} {} {} at ${:,.2f} (total {} of {}) (commission {} ({} each)){}",
            trade.orderStatus.orderId,
            trade.orderStatus.status,
            trade.contract.localSymbol,
            fill.execution.orderId,
            fill.execution.side,
            fill.execution.shares,
            fill.contract.localSymbol,
            fill.execution.price,
            fill.execution.cumQty,
            trade.order.totalQuantity,
            fmtmoney(fill.commissionReport.commission),
            fmtmoney(fill.commissionReport.commission / fill.execution.shares),
            f" (pnl {fill.commissionReport.realizedPNL:,.2f})"
            if fill.commissionReport.realizedPNL
            else "",
        )

        self.updateAgentAccountStatus(
            "commission",
            FillReport(
                orderId=trade.orderStatus.orderId,
                conId=trade.contract.conId,
                sym=trade.contract.localSymbol.replace(" ", ""),
                side=fill.execution.side,
                shares=fill.execution.shares,
                price=fill.execution.price,
                pnl=fill.commissionReport.realizedPNL,
                commission=fill.commissionReport.commission,
                when=fill.execution.time,
            ),
        )

    # ------------------------------------------------------------------
    # News handlers
    # ------------------------------------------------------------------

    def newsBHandler(self, news: NewsBulletin):
        logger.warning("News Bulletin: {}", readableHTML(news.message))

    def newsTHandler(self, news: NewsTick):
        logger.warning("News Tick: {}", news)

    # ------------------------------------------------------------------
    # Execution / position handlers
    # ------------------------------------------------------------------

    async def orderExecuteHandler(self, trade, fill):
        from icli.helpers import IPosition

        isBag: Final = isinstance(trade.contract, Bag)
        logger.warning(
            "[{} :: {}] Trade executed for {}",
            trade.orderStatus.orderId,
            trade.orderStatus.status,
            self._app.nameForContract(trade.contract),
        )

        if isBag:
            # if order executed as a bag, attach each contract id to the same spread
            # (but only create if it doesn't already exist; we don't want partial fills replacing exsiting positions)
            newPosition = IPosition(trade.contract)
            for leg in trade.contract.comboLegs:
                if leg.conId not in self.iposition:
                    self.iposition[leg.conId] = newPosition
        else:
            conId = trade.contract.conId
            if conId not in self.iposition:
                self.iposition[conId] = IPosition(trade.contract)

    def positionEventHandler(self, position: Position):
        """Update position against our local metadata.

        Note: positions are always SINGLE contracts (i.e. you will never get a Bag contract here).
        """

        # TODO: re-evaluate if we actually need this? It doesn't work on startup since we moved
        #       this here instead of in the order notification system. Maybe these _are_ subscribed
        #       on startup by default?
        if False:
            conId = position.contract.conId
            if conId not in self.pnlSingle:
                self.pnlSingle[conId] = self.ib.reqPnLSingle(self._app.accountId, "", conId)
            else:
                self.iposition[position.contract].update(
                    position.contract, position.position, position.avgCost
                )

                # if quantity is gone, stop listening for updates and remove.
                if position.position == 0 and conId in self.pnlSingle:
                    self.ib.cancelPnLSingle(self._app.accountId, "", conId)
                    del self.pnlSingle[conId]

    async def positionActiveLifecycleDoctrine(
        self,
        contract: Contract,
        target,
        upTick: float | Decimal = 0.25,
        downTick: float | Decimal = 0.25,
    ):
        """Begin an acquisition or distribution run for 'position' up to 'target' quantity or total spend.

        'contract' is the instrument for this order.
        'target' is the price and bracket description for this order.
        'upTick' is the next scale-in offset from the previously executed price (higher).
        'downTick' is the next scale-in offset from the previously executed price (lower).

        This is essentialy a meta-wrapper around the 'buy' automation because:
            - 'buy' already loops until it purchases the target quantity (or price)
            - 'buy' already auto-adjusts price (but only to be MORE towards a live price)

        Here we just want to puppet 'buy' a little looser to:
            - buy immediately
            - wait for price to move up or down
            - buy more
            - wait for price to move again, continue buying, until total quantity reached
            - then camp at a stop loss or take profit

        Basically, we want to avoid doing a single large purchase when price is already floating in a ± 10% range sometimes.
        We would rather buy at X, X+N, X-N, X, X+N+K, X-N+K etc for a more realistic time-average cost basis. Also, if the price
        _temporarily_ goes against us, we can acquire at a better cost basis, but if it _continues_ going agianst us, we can stop out.

        Essentially: we want to optimize our chances of staying in a +EV position without being afraid of buying and having something
        retrace to an extent we feel we need to bounce out, only to have it reverse up again.

        Also, if we are in a clear trend direction, it can be safer to scale in at N, N+K, N+K+Z, N+K+Z+Y at worse and worse cost basis
        because prices are working for us. I would rather have purchases between $4 and $10 for something going to $30 than being too afraid
        of buying all at $4 because it could reverse or all at $5 because it could reverse, or buying all at $10 and then having it actually reverse, etc.
        """
        from icli.helpers import IPosition
        from tradeapis.orderlang import DecimalCash

        ...

        # create our position tracker from the input contract.
        # Details of live executions will be updated in this position object by the execution event handlers.
        position: Final = IPosition(contract)
        self.iposition[contract] = position

        f: Final = self.fillers[contract]
        name: Final = self._app.nameForContract(contract)

        isPreview: Final = target.preview

        assert target.qty

        # yeah, a dreaded while-true..... sorry.
        # Basically, our initial condition _can_ be None, but by None we mean "please wait and try again" not "STOP WE ARE DONE,"
        # so we need a way to capture "None" and retry again instead of doing "while not complete..." because complete can be None
        # as a valid indicator too.
        prevCompletePct = None
        while True:
            while (completePct := position.percentComplete(target)) is None:
                # if percentage report is None, it means the IPosition is in the middle of an async update,
                # so we need to WAIT FOR MORE DATA before we have a valid reading from the percentage output.
                logger.warning(
                    "[{}] Waiting for IPosition to complete its updates...",
                    contract.localSymbol,
                )
                await asyncio.sleep(0.001)

            # we need to verify the number actually changed (maybe we are reading too soon after an order completion
            # and the position update callbacks haven't fired to change the position values yet, but we need _new_ details
            # to continue properly).
            if prevCompletePct is not None:
                # if previous is the same as current, no changes were found, so we need to loop again...
                if prevCompletePct == completePct:
                    await asyncio.sleep(0.001)
                    continue

            prevCompletePct = completePct

            # since we got a non-None result as completePct, the remaining 'position' members should be populated

            # if percentage of completness is reported as 100% or larger, we are done here.
            if completePct >= 1:
                logger.info(
                    "[{}] Goal is reached. No more active trade management for purchasing!",
                    contract.localSymbol,
                )
                break

            # if we have ZERO percentage complete so far: buy now and try again at next bounds
            if completePct == 0:
                # TODO: adaptive (or parameter) start quantity
                startQty = 2
                algo = "AF"
                cmd = f"buy '{name}' {startQty} {algo}"

                if isPreview:
                    logger.info("[preview :: {}] Would have run: {}", name, cmd)
                    break
                else:
                    logger.info("[{}] Running: {}", name, cmd)
                    await self._app.buildAndRun(cmd)

                    await f.orderComplete()
                    continue

            # remaining percentage to acquire...
            # Currently, we are targeting a maximum of 5% purchase per buy (or just the remaining quantity, whichever is less)
            remainingPct = min(0.05, 1 - completePct)

            # get STRING representation of quantity to buy (e.g. short is '-1', short cash is '-$100', etc)
            buyVal: str = target.qtyPercent(remainingPct).nice()

            # If remaining percent is _more than_ the remaining quantity, just add the missing quantity.
            if isinstance(target.qty, DecimalCash):
                # if quantity is cash, we want to use cash-based differences of current spend
                # (either: buy buyVal or buy REMAINING quantity if remaining is less than buy val)
                remainingQty = min(
                    target.qty - Decimal(str(position.totalSpend)), buyVal
                )
            else:
                # else, is shares, so use direct quantity instead of price data
                remainingQty = min(target.qty - Decimal(str(position.totalQty)), buyVal)

            # TODO: create dynamic metric for deciding how much to order based on:
            #   - volatility
            #   - "safety net" of average cost versus current market price
            #   - aggressiveness score?
            #   - time of day
            #   - active current and historical IV of contract?

            # else, acquisition has started but is not complete, so we need to continue scheduling order placement triggers.
            # GOAL: buy LOW, buy HIGH up to QTY, when QTY complete, enact STOP or PROFIT conditions.

            assert f.trade
            lastPrice = f.trade.orderStatus.avgFillPrice
            assert lastPrice

            # buy a SMALLER LEVEL or HIGHER LEVEL
            # TODO: evaluate other metric guards for entry besides price ticks? prevscores? delta growth?
            # TODO: more easily adjustable price ticks (use scale system of order intent?)
            # TODO: how to cancel/abnadon predicates once generated if we want to cancel this completely?
            dualside = f"if ('{name}' mid <= {lastPrice} - {downTick}) or ('{name}' mid >= {lastPrice} + {upTick}): buy '{name}' {str(remainingQty)} AF"

            # submit ifthen predicate then WAIT FOR IT TO FILL
            logger.info(
                "[{}] Building predicate for next acquisition: {}", name, dualside
            )

            if not isPreview:
                predicateId = await self._app.buildAndRun(dualside)

                # TODO: how to error check here if we run the buy, but the buy errors out? We would just wait forever here because the fill will never happen.
                #       Do we also have to wait on the _status_ of the buy command after the predicate is executes somehow? We would need "predicate executed" trigger
                #       to start a countdown/timeout waiting for the buy to fill..."

                await f.orderComplete()

        # TODO: integrate this with the OrderIntent scale system?
        # TODO: time-of-day blocking for these (follow overnight volstops?)
        # TODO: trigger on algo/volstop change to true?
        # WHEN ALL ACQUIRED, ENACT STOP LOSS AND TAKE PROFIT EXIT CONDITIONS
        closeQty = position.closeQty
        assert closeQty

        # TODO: we should be assembling these as a proper Peers/OCA object so we can
        #       easily pick between using one or both.
        if target.bracketLoss:
            if target.isBracketLossPercent:
                stopLoss = position.closePercent(-float(target.bracketLoss) / 100)
            else:
                stopLoss = position.closeCash(-float(target.bracketLoss))

        if target.bracketProfit:
            if target.isBracketProfitPercent:
                takeProfit = position.closePercent(float(target.bracketProfit) / 100)
            else:
                takeProfit = position.closeCash(float(target.bracketProfit))

        # TODO: for closing out, we could do the buy automation *or* we could run THIS ENTIRE PROCESS AGAIN but
        #       with a close target instead of the open target... (then obviously don't loop again at the end).
        # TODO: should also adjust 'mid' to be different sides for long/short stop/profit.
        close = f"if '{name}' {{ mid <= {stopLoss} or mid >= {takeProfit} }}: buy '{name}' {closeQty} AF"
        logger.info("[{}] Building predicate for exit: {}", name, close)

        if not isPreview:
            predicateId = await self._app.buildAndRun(close)

    # ------------------------------------------------------------------
    # Ticker / quote update handler (HOT PATH — runs 4 Hz per symbol)
    # ------------------------------------------------------------------

    @logger.catch
    def tickersUpdate(self, tickr):
        """This runs on EVERY quote update which happens 4 times per second per subsubscribed symbol.

        We don't technically need this to receive ticker updates since tickers are "live updated" in their
        own classes for reading, but we _do_ use this to calculate live metadata, reporting, or quote-based
        algo triggers (though, we could also run our own timer-based system to update once per second instead
        of running once per tick... TODO: do that instead so we run one fix-up loop maybe every 500ms to 750ms
        instead of running this callback function 200 times per second across all our symbols).

        This method should always be clean and fast because it runs up to 100+ times per second depending on how
        many tickers you are subscribed to in your client.

        Also note: because this is an ib_insync event handler, any errors or exceptions in this method are NOT
                   reported to the main program. You should attach @logger.catch to this method if you think it
                   isn't working correctly because then you can see the errors/exceptions (if any).
        """
        # logger.info("Ticker update: {}", tickr)

        for ticker in tickr:
            c = ticker.contract
            quotekey = lookupKey(c)

            try:
                # Note: we run the processTickerUpdate() before the "no bid or ask" check because some
                #       symbols like VIF/VIX/VIN and TICK-NYSE and TRIN-NYSE have 'last' values but no bid/ask on them,
                #       but we still want to process their 'last' price updates for EMA trending and alerting.
                iticker = self.quoteState[quotekey]
            except:
                # Often when we unsubscribe from a symbol, we still receive delayed ticker updates even
                # though the symbol is now deleted. Don't alert on getting updates for non-existing
                # symbols unless we need it for debugging.
                # logger.warning("Ticker update for non-existing quote: {}", quotekey)
                continue

            iticker.processTickerUpdate()

            for successCmd in self.ifthenRuntime.check(quotekey):
                match successCmd:
                    case ifthen.IfThenRuntimeSuccess(
                        pid=predicateId, cmd=cmd, predicate=p
                    ):
                        # we have a COMMAND TO RUN so SCHEDULE TO RUN A COMMAND at the next event loop wakeup
                        logger.info("Predicate Complete: {}", p)
                        logger.info(
                            "[{}] Predicate scheduling command: {}", predicateId, cmd
                        )
                        self._app.task_create(
                            f"[{predicateId}] predicate command execution",
                            self._app.buildAndRun(cmd),
                        )
                    case ifthen.IfThenRuntimeError(pid=predicateId, err=e):
                        logger.warning(
                            "[{} :: [predicateId {}]] Check failed for predicate: {}",
                            quotekey,
                            predicateId,
                            str(e),
                        )

            if ticker.bid is None or ticker.ask is None:
                continue

            # only run subscription checks every 2 seconds, but run them on all symbols
            # TODO: okay, so this is now backwards. we just need a global list of things to check once per second
            #       because if we aren't triggering updates PER SYMBOL UPDATE, we don't need to track subscribers per symbol
            #       (except for stopping predicates if we remove a ticker and it has live subscribers...)
            # if time.time() - self.lastSubscriberUpdate >= 2:
            #     for iticker in self.quoteState.values():
            #         complete = set()
            #        for subscriber in iticker.subscribers:

            #   self.lastSubscriberUpdate = time.time()

            # Calculate our live metadata per update.
            # We maintain:
            #    - composite bag greeks for spreads (from each underlying leg)
            #    - EMAs for each symbol
            #    - encapsulated operations for local data details (current price vs. HOD/LOD/close percentages, etc)
            # iquote = self.iquote[quotekey]

            # this is going to be a no-op for most symbols/contracts, but when we have
            # active grabbers for this subscribed contract id, we need to run it.
            # An empty dict.get() is less than 40 ns, so we could run over 20 million of these
            # per second just for the empty dict check and it's okay.
            # The grabber check itself must also be fast and if the grabber decides it needs
            # to grab more quantity, it launches via asyncio.xreate_task() since this ticker
            # update method isn't a coroutine itself...
            # logger.info("[{}] Checking grabbers: {}", ticker.contract.conId, grabbers)
            # (this doesn't work for Bag because Bag has underlying symbols but spread prices so we can't compare "name vs. price")
            name = (c.localSymbol or c.symbol).replace(" ", "")

        # TODO: we could also run volume crossover calculations too...

        # TODO: we should also do some algo checks here based on the live quote price updates...

        # maybe also store all prices into a historical dict indexed by timestamp rounded to nearest 5, 30, 90, 300 seconds and start of day and high and low of day?
        # so we can alert on if price moving up/down by each time slot or high/low of day?
        # auto-alert on positions moving quickly
        # if ticker.contract.localSymbol in self.positions:
        #       price =
        #       if is_long:
        #           we are LONG, so we close by the BID side
        #           price = ticker.bid
        #       else:
        #          # else, we are short, so we close by the ASK side
        #           price = ticker.ask

        #   need to track previous bid/ask to determine when prices are moving for/against us

        # Check module-level flag for quote dumping (import at top to get the value;
        # note: ICLI_DUMP_QUOTES is a module-level bool in cli.py — we re-evaluate it here
        # by importing it lazily so we stay decoupled from the cli module at import time)
        try:
            import icli.cli as _cli_module
            _dump = _cli_module.ICLI_DUMP_QUOTES
        except Exception:
            _dump = False

        if _dump:
            with open(
                f"tickers-{datetime.datetime.now().date()}-{self._app.clientId}.json", "ab"
            ) as tj:
                for ticker in tickr:
                    from icli.helpers import ourjson
                    tj.write(
                        ourjson.dumps(
                            dict(
                                symbol=name,
                                time=str(ticker.time),
                                bid=ticker.bid,
                                bidSize=ticker.bidSize,
                                ask=ticker.ask,
                                askSize=ticker.askSize,
                                volume=ticker.volume,
                            )
                        )
                    )
                    tj.write(b"\n")

    # ------------------------------------------------------------------
    # Account summary / P&L handlers
    # ------------------------------------------------------------------

    def updateSummary(self, v):
        """Each row is populated after connection then continually
        updated via subscription while the connection remains active."""
        # logger.info("Updating sumary... {}", v)
        self.summary[v.tag] = v.value

        # regular accounts are U...; sanbox accounts are DU... (apparently)
        # Some fields are for "All" accounts under this login, which don't help us here.
        # TODO: find a place to set this once instead of checking every update?
        if self._app.isSandbox is None and v.account != "All":
            self._app.isSandbox = v.account[0] == "D"

        # TODO: we also want to maintain "Fake ITicker" for each account value so we can track it over time and use the ITicker values in ifthen statements.
        #       e.g: if :UPL > 10_000: evict *
        #       But, currently, the `ifthen` system only uses symbols and positional symbol aliases (:N) for deriving values for checking.
        # We would have to:
        #   - Create a fake/synthetic ITicker system with lookups mapping from :[AccountDetailName] or :[AccountDetailShorthand] to a fake ITicker object
        #   - Run the `.processTickerUpdate()` on the synthetic ITicker object of each new value being updated so it would trigger any `ifthen` predicate checks

        # collect updates into a single update dict so we can re-broadcast this update
        # to external agent listeners too all at once.
        update = {}
        if v.tag in STATUS_FIELDS_PROCESS:
            try:
                match v.tag:
                    case "FullMaintMarginReq":
                        update["FullMaintMarginReq"] = float(v.value)
                    case "BuyingPower":
                        # regular 25% margin for boring symbols
                        update["BuyingPower4"] = float(v.value)

                        # 30% margin for "regular" symbols
                        update["BuyingPower3"] = float(v.value) / 1.3333333333

                        # 50% margin for overnight or "really exciting" symbols
                        update["BuyingPower2"] = float(v.value) / 2
                    case "NetLiquidation":
                        update[v.tag] = float(v.value)
                    case _:
                        update[v.tag] = float(v.value)
            except:
                # don't care, just keep going
                pass
            finally:
                self.accountStatus |= update

                if v.tag == "NetLiquidation":
                    self.updateURPLPercentages()

                # TODO: resume doing this apparently
                # self.updateAgentAccountStatus("summary", update)

    def updateURPLPercentages(self):
        """Update account percentages.

        We refactored this out to its own method because there are TWO places where we need to
        calculate the URPL percentages:
          - When a new DailyPnL value is updated (during live trades once per second)
          - When a new NetLiquidation value is generated (every couple minutes)
          - When our live RealizedPnL value changes (when any trade completes an execution)

        TODO: I guess we could refactor this out to only calculate realizedpnl% on realized updates
              and unrealizedpnl% on live "dailypnl" or "unrealizedpnl" updates? Thogugh, it would
              always have to re-calculate "totalpnl%" so we only save one row of math for contiional
              single-purpose updates instead of updating all 3 of these values on each trigger.

        The "DailyPnL" stops updating after trades execute, so we need to use the "NetLiquidation" trigger
        as a fallback "flush the current result live" backup so we don't get stuck on pre-closed-trades
        percentages showing."""
        # Update {un,}realizedpnl with DailyPnL updates because DailyPnL updates is refreshed
        # once per second, while previously we were using NetLiquidation as the trigger, but
        # NetLiquidation only updates once every minute or two sometimes (so our (un)realizedpnl
        # percentages were often delayed by an annoying amount of time).
        nl = self.accountStatus.get("NetLiquidation", 1)
        upl = self.accountStatus.get("UnrealizedPnL", 0)
        rpl = self.accountStatus.get("RealizedPnL", 0)

        # Hold updates we refresh into accountStatus all at once.
        update = {}

        # Also generate some synthetic data about percentage gains we made.
        # Is this accurate enough? Should we be doing the math differently or basing it off AvailableFunds or BuyingPower instead???
        # We subtract the PnL values from the account NetLiquidation because the PnL contribution is *already* accounted for
        # in the NetLiquidation value.
        # (the updates are *here* because this runs on every NetLiq val update instead of ONLY on P&L updates)
        update["RealizedPnL%"] = (rpl / (nl - rpl)) * 100
        update["UnrealizedPnL%"] = (upl / (nl - upl)) * 100

        # Also combine realized+unrealized to show the current daily total PnL percentage because
        # maybe we have 12% realized profit but -12% unrealized and we're actually flat...
        update["TotalPnL%"] = update["RealizedPnL%"] + update["UnrealizedPnL%"]

        self.accountStatus |= update

    def updatePNL(self, v):
        """Kinda like summary, except account PNL values aren't summary events,
        they are independent PnL events. shrug.

        Also note: we merge these into our summary dict instead of maintaining
        an indepdent PnL structure.

        Also note: thse don't always get cleared automatically after a day resets,
        so if your client is open for multiple days, sometimes the previous PnL values
        still show up."""

        # TODO: keep moving average of daily PNL and trigger sounds/events
        #       if it spikes higher/lower.
        # logger.info("Updating PNL... {}", v)
        self.summary["UnrealizedPnL"] = v.unrealizedPnL
        self.summary["RealizedPnL"] = v.realizedPnL
        self.summary["DailyPnL"] = v.dailyPnL

        update = {}
        try:
            update["UnrealizedPnL"] = float(v.unrealizedPnL)
            update["RealizedPnL"] = float(v.realizedPnL)
            update["DailyPnL"] = float(v.dailyPnL)
        except:
            # don't care, just keep going
            # (maybe some of these keys don't exist yet, but they will get populated quickly as
            #  the post-connect-async-data-population finishes sending us data for all the fields)
            pass
        finally:
            self.accountStatus |= update
            self.updateURPLPercentages()
            # ignore agent pnl update for now since it is probably in the summary updates anyway?
            # self.updateAgentAccountStatus("pnl", update)

    def updatePNLSingle(self, v):
        """Streaming individual position PnL updates.

        Must be requested per-position.

        The reqPnLSingle method is the only way to get
        live 'dailyPnL' updates per position (updated once per second!)."""

        # logger.info("Updating PNL... {}", v)
        # These are kept "live updated" too, so just save the
        # return value after the subscription.
        self.pnlSingle[v.conId] = v

    def updateAgentAccountStatus(
        self, category: str, update: "FillReport | dict[str, float | int]"
    ):
        """Update internal account data (and maybe external accounting) when trade details get updated.

        Send the `update` dict to current agent server so agent can know about our portfolio for making decisions.

        Note: this method isn't async because it's called from the ib_insync update callbacks, which themselves aren't async,
              but we can just create tasks for updating instead.
        """
        from tradeapis.ordermgr import Trade as OrderMgrTrade

        match category:
            case "commission":
                # New trade event occurred, so let's record it in our local trade-order-position-quantity-stop tracker.
                assert isinstance(update, FillReport)

                # TODO: we also need a "generic sync" mechanism to update _current_ positions into the state if we don't have
                #       executions against them. Basically: list all positions, and if a contractId isn't in our ordermgr, just
                #       add it. We also need additional ordermgr management tools for gluing and ungluing positions possibly.
                self.ordermgr.add_trade(
                    # using just raw contract id as the identifier for now... we can look it up every time we want it resolved I guess.
                    update.conId,
                    OrderMgrTrade(
                        # We scope order ids _per client_ since IBKR request IDs are only per-client.
                        # This also assumes you never "reset request IDs" in your gateway, but this
                        # feature is mainly for tracking trades occurring near in time together.
                        orderid=(self._app.clientId, update.orderId),
                        price=update.price,
                        qty=update.qty,
                        timestamp=update.when,
                        commission=update.commission,
                    ),
                )
            case "summary":
                pass

