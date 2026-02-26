"""Order placement and pricing logic extracted from IBKRCmdlineApp.

This module contains the OrderPlacer class, which handles:
- Tick compliance / rounding (comply, complyNear, complyUp, complyDown)
- Safe order modification (safeModify)
- Bracket order creation (createBracketAttachParent)
- Core order placement (placeOrderForContract)
- Order preview report generation (generatePreviewReport)
- Exit price discovery (orderPriceForSpread, orderPriceForContract)
- Historical execution loading (loadExecutions)
"""
from __future__ import annotations

import asyncio
import dataclasses
import math
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Mapping, Sequence

import prettyprinter as pp
from loguru import logger
from mutil.timer import Timer

from ib_async import Bag, Contract, Future, Order, Stock, Trade

from icli import instrumentdb
from icli.engine import IOrder, CLIOrderType
import icli.engine.orders as orders
from icli.engine.contracts import (
    FullOrderPlacementRecord,
    TradeOrder,
    isset,
    lookupKey,
    nameForContract,
    contractToSymbolDescriptor,
)
from icli.engine.primitives import (
    Bracket,
    BuySell,
    PriceOrQuantity,
    fmtmoney,
)
from icli.helpers import Ladder

# These are imported at runtime to avoid circular imports, but we name them here for type hints.
if TYPE_CHECKING:
    pass

# Import ib_async contract types needed in isinstance checks.
from ib_async import FuturesOption, Option, Crypto  # type: ignore[attr-defined]


class OrderPlacer:
    """Handles order placement, pricing, and tick compliance for IBKR orders.

    Dependencies are injected at construction so this module has no direct
    coupling to IBKRCmdlineApp.
    """

    def __init__(
        self,
        ib,
        conIdCache,
        idb,
        *,
        qualifier,
        portfolio,
        quotes,
        quoteState: dict,
        accountStatus: dict,
    ):
        self.ib = ib
        self.conIdCache = conIdCache
        self.idb = idb
        self._qualifier = qualifier
        self._portfolio = portfolio
        self._quotes = quotes
        self._quoteState = quoteState
        self._accountStatus = accountStatus
        self.loadingCommissions: bool = False

    # ------------------------------------------------------------------
    # Tick compliance / rounding
    # ------------------------------------------------------------------

    async def tickIncrement(self, contract: Contract) -> Decimal | None:
        """Dynamically calculate the tick increment for 'contract' by assuming we want the lowest price above zero it can be."""
        return await self.complyUp(contract, Decimal("0.00001"))

    async def comply(
        self, contract: Contract, price: Decimal | float, direction: instrumentdb.ROUND
    ) -> Decimal | None:
        """Given a contract and an estimated price, round the price to a value appropriate for the instrument."""
        return await self.idb.round(contract, price, direction)

    async def complyNear(
        self, contract: Contract, price: Decimal | float
    ) -> Decimal | None:
        """Given a contract and an estimated price, round price to NEAREST value appropriate for the instrument."""
        return await self.idb.round(contract, price, instrumentdb.ROUND.NEAR)

    async def complyUp(
        self, contract: Contract, price: Decimal | float
    ) -> Decimal | None:
        """Given a contract and an estimated price, round price to equal OR HIGHER value appropriate for the instrument."""
        return await self.idb.round(contract, price, instrumentdb.ROUND.UP)

    async def complyDown(
        self, contract: Contract, price: Decimal | float
    ) -> Decimal | None:
        """Given a contract and an estimated price, round price to equal OR LOWER value appropriate for the instrument."""
        return await self.idb.round(contract, price, instrumentdb.ROUND.DOWN)

    # ------------------------------------------------------------------
    # Safe order modification
    # ------------------------------------------------------------------

    async def safeModify(self, contract, order, **kwargs) -> Order:
        """Given a current order, generate a new order we can _safely_ use to submit for modification.

        This is needed because we can't just re-use an existing trade.order object. The IBKR API dynamically
        back-populates live metadata into the cached trade order object, and we can't send all those auto-populated
        fields back as order modification updates.

        So here we have a centralized way of "cleaning up" an existing trade record to generate a new order which won't
        generate API errors when submitted (hopefully)."""

        # fmt: off
        updatedOrder = dataclasses.replace(
            order,

            # IBKR rejects updates if we submit "IBALGO" as the order type (but IBKR also back-populates the order type to 'IBALGO' itself, so we have to _remove_ it every time)
            orderType="LMT" if order.orderType == "IBALGO" else order.orderType,

            # now add user-requested updates
            **kwargs
        )
        # fmt: on

        # IBKR populates the 'volatility' field on some orders, but it also rejects order updates
        # if the 'volatility' field has a value when the order type is not a VOLATILITY order itself.
        # This covers orders named: VOL and these others below.
        # There are also other VOL orders, but it's unclear if they use the volatility parameter or not:
        # PEGMIDVOL,PEGMKTVOL,PEGPRMVOL,PEGSRFVOL,VOLAT
        if "VOL" not in updatedOrder.orderType:
            updatedOrder.volatility = None

        # also, order updates cannot have parentIds even if they were originally submitted with them (as far as we can tell based on error messages)
        if updatedOrder.parentId:
            updatedOrder.parentId = 0

        # Cached original orders from brackets sometimes have the executing order as Transmit=False, but once the order is live, Transmit must be True always.
        updatedOrder.transmit = True

        # IBKR auto-popualtes fields in live orders we want to remove in future updates if we aren't actively providing them
        removeIfNotRequested = (
            "adjustedOrderType",
            "clearingIntent",
            "deltaNeutralOrderType",
            "displaySize",
            "dontUseAutoPriceForHedge",
            "trailStopPrice",
            "volatilityType",
        )

        for key in removeIfNotRequested:
            if key not in kwargs:
                setattr(updatedOrder, key, None)

        if isset(updatedOrder.lmtPrice):
            fixprice = await self.comply(
                contract,
                updatedOrder.lmtPrice,
                instrumentdb.ROUND.UP
                if updatedOrder.action == "BUY"
                else instrumentdb.ROUND.DOWN,
            )

            if fixprice != updatedOrder.lmtPrice:
                logger.warning(
                    "Updated limit price to comply with order mintick from {} to {}",
                    updatedOrder.lmtPrice,
                    fixprice,
                )
                updatedOrder.lmtPrice = fixprice

        if isset(updatedOrder.auxPrice):
            fixprice = await self.comply(
                contract,
                updatedOrder.auxPrice,
                instrumentdb.ROUND.UP
                if updatedOrder.action == "BUY"
                else instrumentdb.ROUND.DOWN,
            )

            if fixprice != updatedOrder.auxPrice:
                logger.warning(
                    "Updated aux price to comply with order mintick from {} to {}",
                    updatedOrder.auxPrice,
                    fixprice,
                )
                updatedOrder.auxPrice = fixprice

        return updatedOrder

    # ------------------------------------------------------------------
    # Bracket order creation
    # ------------------------------------------------------------------

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
        """Given a starting order, generate bracket take profit/stop loss orders attached to the starting order.

        Return value is (profitOrder, lossOrder) but either or both orders may be None if parameters are not defined.

        Yes, these parameters are a bit of a mess, but it works for now since we want to use this logic in more than one place.
        Technically this could consume a full OrderIntent object and just do the correct thing itself along with a parent order a time params.

        Note: only run this if you have AT LEAST ONE profit or loss order to attach.
        """
        # When creating attached orders, we need manual order IDs because by default they only
        # get generated during the order placement phase.
        if not order.orderId:
            order.orderId = self.ib.client.getReqId()

        order.transmit = False

        profitOrder = None
        lossOrder = None
        if profitLimit is not None:
            profitOrder = orders.IOrder(
                sideClose,
                qty,
                profitLimit,
                outsiderth=outsideRth,
                tif=tif,
                config=config,
            ).order(orderTypeProfit)
            assert profitOrder

            profitOrder.orderId = self.ib.client.getReqId()
            profitOrder.parentId = order.orderId
            profitOrder.transmit = False

        if lossLimit is not None:
            assert lossStopPrice
            lossOrder = orders.IOrder(
                sideClose,
                qty,
                lossLimit,
                aux=lossStopPrice,
                outsiderth=outsideRth,
                tif=tif,
                config=config,
            ).order(orderTypeLoss)
            assert lossOrder

            lossOrder.orderId = self.ib.client.getReqId()
            lossOrder.parentId = order.orderId
            lossOrder.transmit = False

        # if loss order exists, it ALWAYS transmits last
        if lossOrder:
            lossOrder.transmit = True
        elif profitOrder:
            # else, only PROFIT ORDER exists, so send it (profit order is ignored)
            profitOrder.transmit = True

        return profitOrder, lossOrder

    # ------------------------------------------------------------------
    # Core order placement
    # ------------------------------------------------------------------

    async def placeOrderForContract(
        self,
        sym: str,
        isLong: bool,
        contract: Contract,
        qty: PriceOrQuantity,
        limit: Decimal | None,
        orderType: str,
        preview: bool = False,
        # This may be getting out of hand, but for the priority is:
        #  - If 'bracket' and 'ladder' exists, it's an error.
        #  - Always use 'config' for orders.
        #  - If 'ladder' exists, generate a chain of parent orders for each step
        #    ending in a final average price stop loss.
        bracket: Bracket | None = None,
        config: Mapping[str, Decimal | float] | None = None,
        ladder: Ladder | None = None,
    ) -> FullOrderPlacementRecord | None:
        """Place a BUY (isLong) or SELL (not isLong) for qualified 'contract' at 'qty' for 'limit' (if applicable).

        The 'qty' parameter allows switching between price amounts and share/contract/quantity amounts directly.
        """

        # if bracket *AND* ladder are provided, we _remove_ the bracket because
        # the bracket is _included_ in the ladder, but we _do not_ generate a bracket
        # for the initial order itself here, the bracket is _only_ for the average
        # price to stop out or take profit on the ladder if it gets that far.
        if bool(bracket) and bool(ladder):
            bracket = None

        # if we have an exchange override, use it here. If not, the contract exchange is not altered.
        if qty.exchange:
            # Note: if this contract has NOT had a detail request run against it, we can get a false exchange listing the first request,
            #       (like, "SMART ONLY" even though it has a dozen other exchanges), but it _will_ populate in the background for a future
            #       run to likely work properly again.
            xs = self.idb.exchanges(contract)

            # TODO: this exchange validator may need to be a top-level helper method for exchange confirmation in other places too.
            if qty.exchange not in xs:
                logger.error(
                    "[{} :: {}] Requested exchange not found for instrument! Requested exchange: {}\nAvailable exchanges: {}",
                    contract.secType,
                    sym,
                    qty.exchange,
                    pp.pformat(sorted(xs)),
                )

                return None

            contract.exchange = qty.exchange

        logger.info(
            "[{} :: {}] Using exchange: {}",
            contract.symbol,
            contract.localSymbol,
            contract.exchange,
        )

        # Immediately ask to add quote to live quotes for this trade positioning...
        # turn option contract lookup into non-spaced version
        sym = contract.localSymbol.replace(" ", "") or sym.replace(" ", "")

        if limit is not None:
            if isLong:
                comply = await self.complyUp(contract, limit)
            else:
                comply = await self.complyDown(contract, limit)

            assert comply is not None
            limit = comply

        if qty.is_quantity and not limit:
            logger.info("[{}] Request to order qty {} at current prices", sym, qty)
        else:
            digits = self._portfolio.decimals(contract) or 2
            logger.info(
                "[{}] Request to order at dynamic qty {} @ price {:,.{}f}",
                sym,
                qty,
                limit,
                digits,
            )

        quotesym = sym

        # if quote isn't live, add it so we can check against bid/ask details
        self._quotes.addQuoteFromContract(contract)

        if not contract.conId:
            # spead contracts don't have IDs, so only reject if NOT a spread here.
            if not isinstance(contract, Bag):
                logger.error(
                    "[{} :: {}] Not submitting order because contract not qualified!",
                    sym,
                    quotesym,
                )
                return None

        if isinstance(contract, Bag):
            # steal multiplier of first thing in contract. we assume it's okay? This would be wrong for buy-write bags and is only valid for spreads.
            (innerContract,) = await self._qualifier.qualify(
                Contract(conId=contract.comboLegs[0].conId)
            )
        else:
            innerContract = contract

        multiplier = self._portfolio.multiplier(contract)

        # REL and LMT/MKT/MOO/MOC orders can be outside RTH, but futures trade without RTH designation all the time
        # Futures have no "RTH" so they always execute if markets are open.
        if isinstance(innerContract, (FuturesOption, Future)):
            outsideRth = False
        else:
            outsideRth = True

        if isinstance(contract, Option):
            # Purpose: don't trigger warning about "RTH option has no effect" with options...
            if contract.localSymbol[0:3] not in {"SPX", "VIX"}:
                # Currently only SPX and VIX options trade outside (extended) RTH, but other things don't,
                # so turn the flag off so the IBKR Order system doesn't generate a warning
                # considered "outside RTH."
                # For SPY, QQQ, IWM, SMH, and other ETFs, RTH is considered to end at 1615.
                outsideRth = False

        # Note: don't make this an 'else if' to the previous check because this needs to also run again
        # for all option types.
        # TODO: make this list more exhaustive for what only works during RTH liquid hours. Often the IBKR
        #       order system will just say "well, this flag is wrong and I'm ignoring it to execute the order anyway..."
        if orderType in {
            "MIDPRICE",
            "MKT + ADAPTIVE + FAST",
            "MKT + ADAPTIVE + SLOW",
            "LMT + ADAPTIVE + FAST",
            "LMT + ADAPTIVE + SLOW",
        }:
            # as a usability helper, if we are trying to AF or AS on futures, just LMT instead because
            # IBKR rejects attempts to use adaptive algo orders on CME exchanges apparently.
            if isinstance(innerContract, FuturesOption):
                orderType = "LMT"
            else:
                # TODO: cleanup, also verify how we want to run FAST or EVICT outside RTH?
                # Algos can only operate RTH:
                outsideRth = False

                logger.warning(
                    "[{}] ALGO NOT SUPPORTED FOR ALL HOURS. ORDER RESTRICTED TO RTH ONLY!",
                    orderType,
                )

        from typing import Literal
        tif: Literal["Minutes", "DAY", "GTC"]
        if isinstance(contract, Crypto) and isLong:
            # Crypto can only use IOC or Minutes for tif BUY
            # (but for SELL, can use IOC, Minutes, Day, GTC)
            tif = "Minutes"
        elif contract.exchange in {"OVERNIGHT", "IBEOS"}:
            # Overnight requests can't persist past the 20:00-03:50 session (vampire orders!)
            tif = "DAY"
        else:
            # TODO: Add default TIF capability also to global var setting? Or let it be configured in the "limit" menu too?
            tif = "GTC"

        determinedQty: Decimal | float | int = 0

        # if input is quantity, use quantity directly
        # TODO: also allow quantity trades to submit their own limit price like 100@3.33???
        # Maybe even "100@3.33+" to start with _our_ limit price, but then run our price-follow-tracking algo
        # if the initial offer doesn't execute after a couple seconds?
        if qty.is_quantity:
            determinedQty = qty.qty

        bid: None | float = None
        ask: None | float = None
        # Also, this loop does quote lookup to generate the 'limit' price if none exists.
        # Conditions:
        #  - if quantity is a dollar amount, we need to calculate quantity based on current quote.
        #  - also, if this is a preview (with or without a limit price), we calculate a price for margin calculations.
        #  - basically: guard against quantity orders attempting to lookup prices when they aren't needed.
        #    (market orders also imply quantity is NOT money because a market order with no quantity doesn't make sense)
        if ((not limit) and ("MKT" not in orderType)) or preview:
            # only use our automatic-midpoint if we don't already have a limit price
            quoteKey = lookupKey(contract)

            # if this is a new quote just requested, we may need to wait
            # for the system to populate it...
            loopFor = 10

            # only show this quote loop if: LIVE REQUEST or REQUESTING DYNAMIC LIMIT PRICE
            while not (
                currentQuote := self._quotes.currentQuote(
                    quoteKey, show=(not (preview or limit))
                )
            ):
                logger.warning(
                    "[{} :: {}] Waiting for quote to populate...", quoteKey, loopFor
                )
                try:
                    await asyncio.sleep(0.033)
                except:
                    logger.warning("Cancelled waiting for quote...")
                    return None

                if (loopFor := loopFor - 1) == 0:
                    # if we exhausted the loop, we didn't get a usable quote so we can't
                    # do the requested price-based position sizing.
                    logger.error("[{}] No live quote available?", quoteKey)

                    # if we have a limit price, use it as the synthetic quote if a quote isn't available
                    if limit is not None:
                        currentQuote = (float(limit), float(limit))
                        break

                    # no price and no quote, so we can't do anything else here
                    return None

            assert currentQuote
            bid, ask = currentQuote

            if bid is None and ask is None:
                # just make mypy happy with float/Decimal potential differences
                assert limit is not None
                baL = float(limit)
                bid, ask = baL, baL
            elif limit is None:
                assert ask is not None

                if bid is None:
                    logger.warning(
                        "[{}] WARNING: No bid price, so just using ASK directly for buying!",
                        quoteKey,
                    )
                    bid = ask

                # TODO: need customizable aggressiveness levels
                #   - midpoint (default)
                #   - ask + X% for aggressive time sensitive buys
                #   - bid - X% for aggressive time sensitive sells
                # TODO: need to create active management system to track growing/shrinking
                #       midpoint for buys (+price, -qty) or sell (-price) targeting.
                #       See: lang: "buy" for price tracking after order logic.

                # calculate current midpoint of spread rounded to 2 decimals.
                # FAKE THE MIDPOINT WITH A BETTER MARKET BUFFER
                # If we do *exact* midpoint and prices are rapidly rising or falling, we constantly miss
                # the fills. So give it a generous buffer for quicker filling.
                # (could aso just do MKT or MKT PRT orders too in some circumstances)
                # (LONG means allow HIGHER prices for buying (worse entries the higher it goes);
                #  SHORT means allow LOWER prices for selling (worse entries the lower it goes)).
                # We expect the market NBBO to be our actual bounds here, but we're adjusting the
                # overall price for quicker fills.

                # Note: this logic is different than the direct 'evict' logic where we place wider limit
                #       bounds in an attempt to get out as soon as possible. This is more "at market, best effort,
                #       and follow the price if we don't get it the first time" attempts.
                if isinstance(contract, Option):
                    # Options retain "regular" midpoint behavior because spreads can be wide and hopefully
                    # quotes are fairly slow/stable.
                    mid = (bid + ask) / 2
                else:
                    # equity, futures, etc get the wider margins
                    # NOTE: this looks backwards because for us to ACQUIRE a position we must be BETTER than the market
                    #       on limit prices, so here we have BUY HIGH and SELL LOW just to get the position at first.
                    # TODO: these offsets need to be more adaptable to recent ATR-like conditions per symbol,
                    #       but the goal here is immediate fills at market-adjusted prices anyway.
                    # TODO: compare against automaticLimitBuffer() for setting values here???
                    mid = ((bid + ask) / 2) * (1.005 if isLong else 0.995)

                # we checked 'limit is None' in this branch, so we are safe to set/overwrite limit here.
                limit = await self.complyNear(contract, mid)
                assert limit is not None

        # only update qty if this is a money-ask because we also use this limit discovery
        # for quantity-only orders, where we don't want to alter the quantity, obviously.
        if qty.is_money:
            amt = qty.qty

            # calculate order quantity for spend budget at current estimated price
            logger.info("[{}] Trying to order ${:,.2f} worth at {}...", sym, amt, qty)

            assert limit is not None
            determinedQty = self._portfolio.quantityForAmount(contract, amt, limit)

            if not determinedQty:
                logger.error(
                    "[{}] Zero quantity calculated for: {} {} {}!",
                    sym,
                    contract,
                    amt,
                    limit,
                )
                return None

            assert determinedQty > 0

            logger.info(
                "Ordering {:,} {} at ${:,.2f} for ${:,.2f}",
                determinedQty,
                sym,
                limit,
                Decimal(determinedQty) * limit * Decimal(multiplier),
            )

        # declare default values so we can check against them later...
        profitOrder = None
        lossOrder = None

        try:
            sideOpen: BuySell = "BUY" if isLong else "SELL"
            sideClose: BuySell = "SELL" if isLong else "BUY"

            # add instrument-specific digits only to price data (not qty or multipler data)
            digits = self._portfolio.decimals(contract) or 2
            logger.info(
                "[{} :: {}] {:,.2f} @ ${:,.{}f} x {:,.0f} ({}) ALL_HOURS={} TIF={}",
                orderType,
                sideOpen,
                determinedQty,
                limit,
                digits,
                multiplier,
                fmtmoney(float(determinedQty) * float(limit or 0) * multiplier),
                outsideRth,
                tif,
            )

            order = orders.IOrder(
                sideOpen,
                float(determinedQty),
                limit,
                outsiderth=outsideRth,
                tif=tif,
                config=config,
            ).order(orderType)
            assert order

            if bracket:
                profitOrder, lossOrder = self.createBracketAttachParent(
                    order,
                    sideClose,
                    float(determinedQty),
                    bracket.profitLimit,
                    bracket.lossLimit,
                    bracket.lossStop,
                    outsideRth,
                    tif,
                    bracket.orderProfit,
                    bracket.orderLoss,
                )
        except:
            logger.exception("ORDER GENERATION FAILED. CANNOT PLACE ORDER!")
            return None

        # if ladder requested, create a tier of orders each parented to the previous.
        steps: list[tuple[str, Order]] = []
        if ladder:
            prevId = None
            prevOrder = None
            for i, step in enumerate(ladder):
                # generate a new order ID on each order step
                currentId = self.ib.client.getReqId()

                # get current price and quantity of step
                p = step.limit
                q = step.qty
                stepname = f"STEP {i}"

                # Create order for step
                steporder = orders.IOrder(
                    sideOpen,
                    q,
                    p,
                    outsiderth=outsideRth,
                    tif=tif,
                    config=config,
                ).order(orderType)
                assert steporder

                steporder.orderId = currentId
                steporder.transmit = False

                # attach future orders to be children of the previous order
                if prevOrder:
                    steporder.parentId = prevOrder.orderId

                prevOrder = steporder

                steps.append((stepname, steporder))

            profitprice = ladder.profit
            stopprice = ladder.stop

            # if we have at least one of take profit or stop loss, engage the bracket logic.
            if stopprice or profitprice:
                profitprice = (
                    await self.comply(
                        innerContract,
                        profitprice,
                        instrumentdb.ROUND.DOWN if isLong else instrumentdb.ROUND.UP,
                    )
                    if profitprice
                    else None
                )
                stopprice = (
                    await self.comply(
                        innerContract,
                        stopprice,
                        instrumentdb.ROUND.DOWN if isLong else instrumentdb.ROUND.UP,
                    )
                    if stopprice
                    else None
                )

                profitOrder, lossOrder = self.createBracketAttachParent(
                    prevOrder,
                    sideClose,
                    ladder.qty,
                    profitprice,
                    stopprice,
                    stopprice,
                    outsideRth,
                    tif,
                    # TODO: make these also adjutable... we probably want a "STOP LIMIT" here instead of just STP since we have multiple qty
                    ladder.profitAlgo,
                    ladder.stopAlgo,
                    config,
                )

                # Add profit take and loss reducer to orders for placement.
                steps.extend(
                    [
                        (n, o)
                        for n, o in [("PROFIT", profitOrder), ("LOSS", lossOrder)]
                        if o is not None
                    ]
                )
            else:
                steps[-1][-1].transmit = True

            # else, if we do not have ANY profit or loss requested, we must update the final order to transmit=True so the entire chain executes.

        if order.orderType == "PEG MID":
            if isinstance(contract, Option):
                logger.warning(
                    "[{}] Routing order to IBUSOPT for PEG MID",
                    contract.localSymbol or contract.symbol,
                )
                contract.exchange = "IBUSOPT"
            elif isinstance(contract, Stock):
                logger.warning(
                    "[{}] Routing order to IBKRATS for PEG MID",
                    contract.localSymbol or contract.symbol,
                )
                contract.exchange = "IBKRATS"
            else:
                logger.error("Peg-to-Midpoint is only valid for Stocks and Options!")
                return None

        name = contract.localSymbol or contract.symbol
        desc = f"{name} :: QTY {order.totalQuantity:,}"

        # convert quantity to integer (away from float) if it is the same for nicer formatting
        if order.totalQuantity == (itq := int(order.totalQuantity)):
            order.totalQuantity = itq

        ordpairs: tuple[tuple[str, Order | None], ...]
        if preview:
            # generate input format to preview report (tuple of tuples of (name, order))
            if steps:
                # if we have steps created, the 'steps' are already in ordpairs format.
                # (except, we want the FIRST order to be LAST, so we reverse all of these here)
                ordpairs = tuple(reversed(steps))
            else:
                # default profit/loss/order
                ordpairs = (
                    ("PROFIT", profitOrder),
                    ("LOSS", lossOrder),
                    ("TRADE", order),
                )
            runOrders = tuple(
                [
                    (ordname, ordord)
                    for ordname, ordord in ordpairs
                    if ordord is not None
                ]
            )

            # logger.info("Orders are: {}", pp.pformat(runOrders))

            await self.generatePreviewReport(contract, bid, ask, runOrders, multiplier)

            # preview request complete! Nothing remaining to do here.
            return None

        logger.info("[{}] Ordering {} via {}", desc, contract, order)

        profitTrade = None
        lossTrade = None

        # placeOrder() returns a "live updating" Trade object with live position execution detail updates

        if isinstance(contract, Bag):
            await self._qualifier.addNonGuaranteeTagsIfRequired(
                contract, order, profitOrder, lossOrder
            )

        trade = self.ib.placeOrder(
            await self._qualifier.contractForOrderSide(order, contract), order
        )

        limitRecord = TradeOrder(trade, order)

        profitTrade = None
        profitRecord = None
        if profitOrder:
            profitTrade = self.ib.placeOrder(
                await self._qualifier.contractForOrderSide(profitOrder, contract), profitOrder
            )

            profitRecord = TradeOrder(profitTrade, profitOrder)

        lossTrade = None
        lossRecord = None
        if lossOrder:
            lossTrade = self.ib.placeOrder(
                await self._qualifier.contractForOrderSide(lossOrder, contract), lossOrder
            )

            lossRecord = TradeOrder(lossTrade, lossOrder)

        assert trade
        logger.info(
            "[{} :: {} :: {}] Placed: {}",
            trade.orderStatus.orderId,
            trade.orderStatus.status,
            name,
            pp.pformat(trade),
        )

        if profitOrder:
            assert profitTrade
            logger.info(
                "[{} :: {} :: {}] Profit Order Placed: {}",
                profitTrade.orderStatus.orderId,
                profitTrade.orderStatus.status,
                name,
                pp.pformat(profitTrade),
            )

        if lossOrder:
            assert lossTrade
            logger.info(
                "[{} :: {} :: {}] Loss Order Placed: {}",
                lossTrade.orderStatus.orderId,
                lossTrade.orderStatus.status,
                name,
                pp.pformat(lossTrade),
            )

        # create event for clients to listen for in-progress execution updates until
        # the entire quantity of this order is filled.
        return FullOrderPlacementRecord(
            limitRecord, profit=profitRecord, loss=lossRecord
        )

    # ------------------------------------------------------------------
    # Preview report
    # ------------------------------------------------------------------

    async def generatePreviewReport(
        self,
        contract: Contract,
        bid: float | None,
        ask: float | None,
        orders: Sequence[tuple[str, Order]],
        multiplier: float = 1,
    ) -> None:
        # we assume the final order in the list is the ACTUAL INITIAL TRADE ORDER
        # Note: order previews don't respect the OCA or OTOCO or parentId system, so they all just get run independently,
        #       but we use the price of the *last* element in the order list to be the initial live order we are attempting.
        order = orders[-1][-1]
        previewPrice = order.lmtPrice
        assert previewPrice is not None

        # Note: we require space removal here because we use 'no-space-symbols' for quote lookups later.
        symname = (contract.localSymbol or contract.symbol).replace(" ", "")
        desc = f"{symname} :: QTY {order.totalQuantity:,}"

        def whatIfPrepare(ordName, o):
            # preview orders ignore brackets and always transmit by default (the IBKR API won't preview transmit=False orders)
            # Also, the IBKR order preview system doesn't know what to do with parent-tracked since it doesn't maintain an order submission queue.
            logger.info(
                "[{} :: {:>6} :: {}] For preview query, converting: [transmit {}] to False; [parentId {}] to None",
                desc,
                ordName,
                o.orderId,
                o.transmit,
                o.parentId,
            )
            o.transmit = True
            o.parentId = None

            # logger.info("Sending order: {}", pp.pformat(o))
            return o

        if isinstance(contract, Bag):
            await self._qualifier.addNonGuaranteeTagsIfRequired(contract, *[o[1] for o in orders])

        try:
            # if this is a multi-order bracket, run all the whatIf requests concurrently (a 2x-3x speedup over sequential operations)
            orderStatusResults = await asyncio.wait_for(
                asyncio.gather(
                    *[
                        self.ib.whatIfOrderAsync(contract, whatIfPrepare(name, check))
                        for name, check in orders
                    ]
                ),
                timeout=2,
            )

            assert orderStatusResults
            whatIfResults = tuple(zip(orders, orderStatusResults))
        except:
            logger.error(
                "Timeout while trying to run order preview (sometimes IBKR is slow or the order preview API could be offline)"
            )
            return

        assert whatIfResults

        # If order has negative price to desginate a short, flip it back to negative quanity for our reporting metrics to work.
        # Note: this is safe because preview orders are never processed as 'live' orders, so modifying the orders in-place is okay.
        for name, o in orders:
            if o.lmtPrice is not None and o.lmtPrice < 0:
                o.totalQuantity = -o.totalQuantity
                o.lmtPrice = -o.lmtPrice

        # preview EACH part of the potential 3-way backet order.
        # Note: we process 'TRADE' last, leaving 'status' as the final trade status for the final preview math after this loop.
        for (ordName, order), statusStrs in whatIfResults:  # type: ignore
            if not statusStrs:
                logger.warning(
                    "No preview status generated? Can't process preview request!"
                )
                return

            # request all fields as numeric types instead of default strings (so we can do math on the results easier)
            status: OrderStateNumeric = statusStrs.numeric(digits=2)

            logger.info(
                "[{} :: {}] PREVIEW REQUEST {} via {}",
                desc,
                ordName,
                contract,
                pp.pformat(order),
            )

            logger.info(
                "[{} :: {}] PREVIEW RESULT: {}",
                desc,
                ordName,
                pp.pformat(status.formatted()),
            )

        isContract = isinstance(contract, (Bag, Option, Future, FuturesOption))

        # We currently assume only two kinds of things exist. We could add more.
        nameOfThing = "CONTRACT" if isContract else "SHARE"

        # set 100% margin defaults so our return value has something populated even if margin isn't relevant (options, etc)
        margPctInit = 100.0
        margPctMaint = 100.0

        digits = self._portfolio.decimals(contract)

        # fix up math issues if totalQuantity became a Decimal() along the way
        order.totalQuantity = float(order.totalQuantity)

        # TODO: we still need to fix ib_async to return None for unset fields of Order() object, but it's more defaults-rewrite work.
        if isset(previewPrice):
            logger.info(
                "[{}] PREVIEW LIMIT PRICE PER {}: ${:,.{}f} (actual @ {}x: ${:,.{}f})",
                desc,
                nameOfThing,
                previewPrice,
                digits,
                multiplier,
                float(previewPrice or 0) * multiplier,
                digits,
            )

        # for options or other conditions, there's no margin change to report.
        # also, if there is a "warning" on the trade, the numbers aren't valid.
        if (
            (not status.warningText)
            and (status.initMarginChange > 0)
            and previewPrice is not None
        ):
            assert order
            baseTotal = order.totalQuantity * float(previewPrice) * multiplier
            margPctInit = (
                status.initMarginChange / (baseTotal or status.initMarginChange)
            ) * 100
            margPctMaint = (
                status.maintMarginChange / (baseTotal or status.maintMarginChange)
            ) * 100

            # if this order is for a CREDIT spread, our margin calculations don't apply because the margin
            # is _reserved_ on our account instead of "paid forward" as with equity symbols.
            # Basically, only show margin requirements if account is debited for this transaction.
            logger.info(
                "[{}] PREVIEW MARGIN REQUIREMENT INIT: {:,.2f} % ({})",
                desc,
                margPctInit,
                fmtmoney(status.initMarginChange),
            )

            # "MAIN" for "MAINTENANCE" to match the length of "INIT" above for alignment.
            # FIX WITH PRT:
            # 2023-11-17 06:55:20.473 | INFO     | icli.cli:placeOrderForContract:834 - [RTYZ3 :: QTY 6] PREVIEW MARGIN REQUIREMENT INIT: 0.00 %
            # 2023-11-17 06:55:20.474 | INFO     | icli.cli:placeOrderForContract:841 - [RTYZ3 :: QTY 6] PREVIEW MARGIN REQUIREMENT MAIN: 0.00 % (IBKR is loaning 100.00 %)
            logger.info(
                "[{}] PREVIEW MARGIN REQUIREMENT MAIN: {:,.2f} % (IBKR is loaning {:,.2f} %)",
                desc,
                margPctMaint,
                100 - margPctMaint,
            )

            if margPctInit and status.initMarginChange >= status.maintMarginChange:
                logger.info(
                    "[{}] PREVIEW MARGIN REQUIREMENT DRAWDOWN ALLOWED: {:,.2f} %",
                    desc,
                    100 * (1 - margPctMaint / margPctInit),
                )

                logger.info(
                    "[{}] PREVIEW MARGIN REQUIREMENT DRAWDOWN LEVERAGE POINTS: {:,.{}f}",
                    desc,
                    (status.initMarginChange - status.maintMarginChange)
                    / multiplier
                    / order.totalQuantity,
                    digits,
                )

                logger.info(
                    "[{}] PREVIEW MAINT MARGIN PER {}: {}",
                    desc,
                    nameOfThing,
                    fmtmoney(status.maintMarginChange / order.totalQuantity),
                )

            logger.info(
                "[{}] PREVIEW INIT MARGIN PER {}: {}",
                desc,
                nameOfThing,
                fmtmoney(status.initMarginChange / order.totalQuantity),
            )

        leverageKind = "CONTRACT" if isContract else "STOCK"
        assert order

        # estimate gains if the obtained quantity moves by specific price increment amounts.
        # Note: only report leverage if order is taking on risk (e.g. don't report for closing transactions).
        if status.equityWithLoanChange <= 0 or status.initMarginChange > 0:
            for amt in (0.20, 0.75, 1, 3, 5):
                logger.info(
                    "[{}] PREVIEW LEVERAGE ({:,} x {}): ${:,.2f} {} MOVE LEVERAGE is ${:,.2f}",
                    desc,
                    order.totalQuantity,
                    multiplier,
                    amt,
                    leverageKind,
                    amt * multiplier * order.totalQuantity,
                )

        # also print a delta-adjusted leverage for the underlying if delta is less than 1
        symkey = lookupKey(contract)
        highCommission = status.maxCommission or status.commission

        if highCommission and (ticker := self._quoteState.get(symkey)):
            if ticker.modelGreeks and abs(ticker.modelGreeks.delta) < 1:
                for amt in (1, 3, 9):
                    # NOTE: we report both the value move and the exit-at profit AFTER COMISSIONS assuming the 'highComission' commission is 2x (open+close)
                    # Also note: this is an estimate, because we are not adjusting for delta growing as price moves in our favor.
                    # Also also note: this move leverage can be misleadig for volatility straddles because we expect vega thus gamma to overwhelm underlying movements directly.
                    move = (
                        amt
                        * multiplier
                        * abs(order.totalQuantity)
                        * abs(ticker.modelGreeks.delta)
                    )
                    logger.info(
                        "[{}] ${:,.2f} UNDERLYING MOVE LEVERAGE is {} (exit: {:>6})",
                        desc,
                        amt,
                        fmtmoney(move),
                        fmtmoney(move - (highCommission * 2)),
                    )

        # "MAIN" for "MAINTENANCE" to match the length of "INIT" above for alignment.

        # if we have any commission, estimate how many leveraged points we need to "Earn it out" for a round trip.
        if status.minCommission or status.commission:
            low = status.minCommission or status.commission
            high = status.maxCommission or status.commission

            # only print number of commissions reported (some products have "flat" commissions and the price is always the same,
            # so for those cases we never have a low/high to report, it's just one consistent nubmer)
            l = 2 * low
            h = 2 * high
            l2 = (2 * low) / multiplier / abs(order.totalQuantity)
            h2 = (2 * high) / multiplier / abs(order.totalQuantity)

            if low == high:
                # if low commission == high commission, then we only have one commission to report.
                points = f"(${l:,.{digits}f}): ${l2:,.{digits}f}"
            else:
                points = f"(${l:,.{digits}f} to ${h:,.{digits}f}): ${l2:,.{digits}f} to ${h2:,.{digits}f}"

            logger.info(
                "[{}] CONTRACT POINTS TO PAY ROUNDTRIP COMMISSION {}",
                desc,
                points,
            )

        if status.minCommission:
            # options and stocks have a range of commissions
            logger.info(
                "[{}] PREVIEW COMMISSION PER {}: ${:.4f} to ${:.4f}",
                desc,
                nameOfThing,
                status.minCommission / order.totalQuantity,
                status.maxCommission / order.totalQuantity,
            )

            if multiplier > 1:
                # (Basically: how much must the underlying change in price for you to pay off the commission for this order.
                tcmin = status.minCommission / order.totalQuantity / multiplier
                tcmax = status.maxCommission / order.totalQuantity / multiplier
                logger.info(
                    "[{}] PREVIEW COMMISSION PER UNDERLYING: ${:.4f} to ${:.4f} (2x: ${:.4f} to ${:.4f})",
                    desc,
                    tcmin,
                    tcmax,
                    2 * tcmin,
                    2 * tcmax,
                )
        elif status.commission:
            # futures contracts and index options contracts have fixed priced commissions so
            # they don't provide a min/max range, it's just one guaranteed value.
            legComm = ""
            if isinstance(contract, Bag):
                legComm = f" (per ech leg ({len(contract.comboLegs)}): ${status.commission / (order.totalQuantity * len(contract.comboLegs)):.4f})"

            logger.info(
                "[{}] PREVIEW COMMISSION PER CONTRACT: ${:.4f}{}",
                desc,
                (status.commission) / order.totalQuantity,
                legComm,
            )

            tc = status.commission / order.totalQuantity / multiplier
            if multiplier > 1:
                logger.info(
                    "[{}] PREVIEW COMMISSION PER UNDERLYING: ${:.4f} (2x: ${:.4f})",
                    desc,
                    tc,
                    2 * tc,
                )

        # we allow bid to be none if this is a 'buy on unlikely' scenario
        # assert bid is not None

        if bid is None:
            bid = 0

        assert ask is not None

        # calculate percentage width of the spread just to note if we are trading difficult to close positions
        # Note: if bid is '0' (for some spreads) then we just use "whole quantity" for the divisor percentage math here.
        spreadDiff = ((ask - bid) / max(1, bid)) * 100
        logger.info(
            "[{}] BID/ASK SPREAD IS {:,.2f} % WIDE ({} spread @ {} total)",
            desc,
            spreadDiff,
            fmtmoney(ask - bid),
            fmtmoney(
                order.totalQuantity
                * float(previewPrice)
                * (float(spreadDiff) / 100)
                * multiplier
            ),
        )

        if spreadDiff > 5:
            logger.warning(
                "[{}] WARNING: BID/ASK SPREAD ({:,.2f} %) MAY CAUSE NOTICEABLE LOSS/SLIPPAGE ON EXIT",
                desc,
                spreadDiff,
            )

        # TODO: make this delta range configurable? config file? env? global setting?
        if isinstance(contract, (Option, FuturesOption)):
            mg = self._quoteState[symname].modelGreeks
            if mg:
                delta = mg.delta
                if not delta:
                    logger.warning("[{}] WARNING: OPTION DELTA NOT POPULATED YET", desc)
                elif abs(delta) <= 0.15:
                    logger.warning(
                        "[{}] WARNING: OPTION DELTA IS LOW ({:.2f}) — THIS MAY NOT WORK FOR SHORT TERM TRADING",
                        desc,
                        delta,
                    )
            else:
                logger.warning(
                    "[{}] WARNING: OPTION GREEKS NOT YET POPULATED FOR DELTA CHECKING",
                    desc,
                )

        # if this trade INCREASES our equity, let's see if there's a risk involved
        # Note: this is primarily useful for credit spreads where we receive a credit for a fixed margin risk.
        # TODO: also run R:R calculations when buying debit spreads versus the spread filling to 100%
        # TODO: test this against regular equity shorts to see what it reports too.
        # TODO: this doesn't run properly if there's zero margin impact (due to other offsets) but it's a new short position.
        # TODO: maybe only run this if the contract is a Bag instead of checking equity loan change as the trigger?
        # This needs to account for reg-t shorts (initial == maint then credit received as EWLC) and also
        # account for SPAN shorts (initial > maint and credit received not reflected in EWLC).
        # For Reg-T shorts, the account holds the full short block amount as initial margin because
        # the credit can't relieve your own margin (so if you short sell $20k on a $20k risk, you still have $20k margin).
        # For SPAN shorts, the exact risk is _only_ the margin change (so if you
        # short sell $20k on a $20k risk, you have $0 margin chnage since the credit cancels out the risk).
        if status.initMarginChange > 0 and order.action == "SELL":
            ewlc = status.equityWithLoanChange
            if status.initMarginChange > 0:
                # If equity is increasing, then this is a short (receiving credit) with margin risk.
                # Our risk is (total increase in stop-out margin call requirement).
                # Maint Margin is always less than or equal to initial margin, so it will stop-out the trade at an
                # equal or _sooner_ level than the initial margin requirement.
                risk = status.maintMarginChange - ewlc
                riskPct = risk / (baseTotal or risk)

                # the more decimals the more extreme the ratio generated, so instead of 69:100 at 2 decimals, show 7:10 at 1 decimal.
                fracMin = Fraction(round(riskPct, 1)).limit_denominator()
                riskMin, rewardMin = fracMin.numerator, fracMin.denominator

                # also provide an even lower resoultion R:R single digit ratio
                fracMin2 = Fraction(round(riskPct)).limit_denominator()
                riskMin2, rewardMin2 = fracMin2.numerator, fracMin2.denominator
                logger.warning(
                    "[{}] RISKING MARGIN: ${:,.2f} (received ${:,.2f} credit; risking {:,.2f}x == {}:{} Risk:Reward ratio{})",
                    desc,
                    risk,
                    baseTotal,
                    riskPct,
                    riskMin,
                    rewardMin,
                    # provide a lower resolution (but easier to read) ratio if the first attempt isn't 1:N already:
                    f" ({riskMin2}:{rewardMin2})"
                    if riskMin > 10 and (riskMin, rewardMin) != (riskMin2, rewardMin2)
                    else "",
                )

        # (if trade isn't valid, trade is an empty list, so only print valid objects...)
        if status:
            if not (status.commission or status.minCommission or status.maxCommission):
                logger.error(
                    "[{}] TRADE NOT VIABLE DUE TO MISSING COMMISSION ESTIMATES",
                    desc,
                )

            # excess = self.accountStatus["ExcessLiquidity"] - status.initMarginChange
            excess = status.equityWithLoanAfter - status.initMarginAfter
            if excess < 0:
                logger.warning(
                    "[{}] TRADE NOT VIABLE. MISSING EQUITY: {}",
                    desc,
                    fmtmoney(excess),
                )
            else:
                # show rough estimate of how much we're spending.
                # for equity instruments with margin, we use the margin buy requirement as the cost estimate.
                # for non-equity (options) without margin, we use the absolute value of the buying power drawdown for the purchase.
                # TODO: this value is somewhere between wrong or excessive if there's already marginable positions engaged since
                #       the calculation here is assuming a new position request is the only position in the account.

                # amount required to hold trade - amount extraced from trade
                # options are: 0 - (-premium)
                # short spreads are: full margin - (received credit)
                # longs are: (margin holding) - (almost nothing (basically just commission offsets))
                # Note: assuming standard reg-t margin, your short spreads are NOT returned to you as buying power,
                #       so we DO NOT add credits from spreads back as a risk minimization here.
                #       e.g. if you have a $5,000 margin spread with a +$3,000 credit, you still have $5,000 reduced
                #            buying power instead of (trade.initMarginChange - trade.equityWithLoanChange = $5,000 - $3,000) reduced buying power
                # UPDATE: for credit events, we assume the initial margin is our cost impact.
                #         for debit events, we assume there is no margin change, but the EWL is negative (cost of position) so we flip it as our risk
                risk = status.maintMarginChange or -status.equityWithLoanChange

                # there is also the account status 'InitMarginReq' field we could potentially use as well.
                logger.info(
                    "[{}] PREVIEW REMAINING INIT CASH AFTER TRADE ({}): {}",
                    desc,
                    fmtmoney(-status.initMarginChange),
                    fmtmoney(status.equityWithLoanAfter - status.initMarginAfter),
                )

                logger.info(
                    "[{}] PREVIEW REMAINING MAINT CASH AFTER TRADE ({}): {}",
                    desc,
                    fmtmoney(-status.maintMarginChange),
                    fmtmoney(status.equityWithLoanAfter - status.maintMarginAfter),
                )

                fundsDiff = (risk / self._accountStatus["AvailableFunds"]) * 100

                if fundsDiff < 0:
                    # your account value is GROWING, this is a funds increase
                    logger.info(
                        "[{}] PREVIEW TRADE PERCENTAGE OF FUNDS ADDED: {:,.2f} %",
                        desc,
                        -fundsDiff,
                    )
                else:
                    # else, account value is LOWERING so this is a funds reduction
                    logger.info(
                        "[{}] PREVIEW TRADE PERCENTAGE OF FUNDS USED: {:,.2f} %",
                        desc,
                        fundsDiff,
                    )

    # ------------------------------------------------------------------
    # Exit price discovery
    # ------------------------------------------------------------------

    def orderPriceForSpread(self, contracts: Sequence[Contract], positionSize: int):
        """Given a set of contracts, attempt to find the closing order."""
        ot = self.ib.openTrades()

        contractIds = set([c.conId for c in contracts])
        # Use a list so we can collect multiple exit points for the same position.
        ts = []
        for t in ot:
            if not isinstance(t.contract, Bag):
                continue

            legIds = set([c.conId for c in t.contract.comboLegs])
            if legIds == contractIds:
                qty, price = t.orderStatus.remaining, t.order.lmtPrice
                ts.append((qty, price))

        # if only one and it's the full position, return without formatting
        if len(ts) == 1:
            if abs(int(positionSize)) == abs(ts[0][0]):
                return ts[0][1]

        # else, break out by order size, sorted from smallest to largest exit prices
        return sorted(ts, key=lambda x: abs(x[1]))

    def orderPriceForContract(self, contract: Contract, positionSize: float | int):
        """Attempt to match an active closing order to an open position.

        Works for both total quantity closing and partial scale-out closing."""
        ot = self.ib.openTrades()

        # Use a list so we can collect multiple exit points for the same position.
        ts = []
        for t in ot:
            # t.order.action is "BUY" or "SELL"
            opposite = "SELL" if positionSize > 0 else "BUY"
            if (
                t.order.action == opposite
                and t.contract.localSymbol == contract.localSymbol
            ):
                # Closing price is opposite sign of the holding quantity.
                # (i.e. LONG positions are closed for a CREDIT (-) and
                #       SHORT positions are closed for a DEBIT (+))
                assert (
                    t.order.lmtPrice is not None
                ), "How is the order limit price None here?"

                ts.append(
                    (
                        int(t.orderStatus.remaining),
                        float(
                            math.copysign(1, positionSize)
                            * -1
                            * float(t.order.lmtPrice)
                        ),
                    )
                )

        # if only one and it's the full position, return without formatting
        if len(ts) == 1:
            if abs(int(positionSize)) == abs(ts[0][0]):
                return ts[0][1]

        # else, break out by order size, sorted from smallest to largest exit prices
        return sorted(ts, key=lambda x: abs(x[1]))

    # ------------------------------------------------------------------
    # Historical executions
    # ------------------------------------------------------------------

    async def loadExecutions(self) -> None:
        """Manually fetch all executions from the gateway.

        The IBKR API only sends live push updates for executions on the _current_ client,
        so to see executions from either _all_ clients or executions before this client started,
        we need to ask for them all again.
        """

        logger.info("Fetching full execution history...")
        try:
            # manually flag "we are loading historical commissions, so don't run the event handler"
            self.loadingCommissions = True

            with Timer("Fetched execution history"):
                try:
                    await asyncio.wait_for(self.ib.reqExecutionsAsync(), 7)
                except:
                    logger.error("Executions failed to load before the timeout period.")
        finally:
            # allow the commission report event handler to run again
            self.loadingCommissions = False
