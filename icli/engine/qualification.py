"""Contract qualification methods extracted from IBKRCmdlineApp."""
from __future__ import annotations

import asyncio
import copy
import fnmatch
import math
import os
from collections import defaultdict
from typing import TYPE_CHECKING

import prettyprinter as pp
from loguru import logger

import ib_async
from ib_async import Bag, ComboLeg, Contract, Future, Index, Option, Stock
import tradeapis.buylang as buylang

import icli.engine.orders as orders
from icli.engine.contracts import contractForName, contractToSymbolDescriptor, getExpirationsFromTradier

if TYPE_CHECKING:
    from ib_async import IB


class ContractQualifier:
    """Contract qualification methods.

    Dependencies injected at construction:
    - ib: IB connection
    - conIdCache: contract ID cache
    - quoteState: live quote state dict
    - ol: order language parser (optional, only needed for bagForSpread)
    """

    def __init__(self, ib, conIdCache, quoteState, ol=None):
        self.ib = ib
        self.conIdCache = conIdCache
        self.quoteState = quoteState
        self.ol = ol

    async def qualify(self, *contracts, overwrite: bool = False) -> list[Contract]:
        """Qualify contracts against the IBKR allowed symbols.

        Mainly populates .localSymbol and .conId

        You can set 'overwrite' if you want to ALWAYS recache these lookups.

        We also cache the results for ease of re-use and for mapping
        contractIds back to names later."""

        # logger.info("Inbound request (overwrite: {}): {}", overwrite, contracts)
        # logger.warning("Current full cache: {}", [x for x in self.conIdCache])

        # Group contracts into cached and uncached so we can look up uncached contracts
        # all at once but still iterate them in expected order of results.
        cached_contracts = {}
        uncached_contracts = []

        # in order to retain the result in the same order as the input, we map the python id() value of
        # each contract object to the final contract result itself. Then, at the end, we just iterate
        # the input 'contracts' collection looking up them by id() for the final result order matching
        # the input order again.
        totalResult = {}

        def cachedContractCorrupt(cached_contract, contract) -> bool:
            if cached_contract and (not cached_contract.conId):
                logger.warning(
                    "BUG: Why doesn't cached contract have an ID? Looking up again. Request: {} vs Found: {}",
                    contract,
                    cached_contract,
                )

                return True

            # watch out for data bugs if we had a mis-placed cache element somehow
            if cached_contract and (
                (contract.strike and (contract.strike != cached_contract.strike))
                or (contract.secType and (contract.secType != cached_contract.secType))
                or (
                    contract.localSymbol
                    and (contract.localSymbol != cached_contract.localSymbol)
                )
            ):
                logger.warning(
                    "BUG: Cached contract doesn't match requested contract? Rejecting cache and looking up again. Request: {} vs. Found: {}",
                    contract,
                    cached_contract,
                )

                return True

            if type(cached_contract) == Contract:
                logger.warning(
                    "BUG: Cached contract is just 'Contract' which doesn't work. Rejecting cache and looking up again. Request: {} vs Found: {}",
                    contract,
                    cached_contract,
                )

                return True

            return False

        # Our cache operates on two tiers:
        #  - ideally, we look up contracts by id directly
        #  - alternatively, we look up contracts by name, but we also need to know the _type_ of the name we are looking up.
        # So, we check the requested contracts for:
        #  - if input contract already has a contract id, we look up the conId directly.
        #  - if input contract doesn't have an id, we generate a lookup key of Class-Symbol like "Future-ES" or "Index-SPX"
        #    so we can retrieve the correct instrument class combined with the proper symbol from a cached contract.
        for contract in contracts:
            cached_contract = None

            assert isinstance(
                contract, Contract
            ), f"Why didn't you send a contract here? Note: the input is *contracts, and _not_ a list of contracts! Got: {contract}"

            key = None
            try:
                # only attempt to look up using ID if ID exists, else attempt to lookup by name
                if contract.conId:
                    # key = contractToSymbolDescriptor(contract)
                    # logger.info("by ID Looking up: {} :: {}", contract, key)
                    # Attempt first lookup using direct ID, but if ID isn't found try to use the Class-Symbol key format...
                    cached_contract = self.conIdCache.get(contract.conId)  # type: ignore
                else:
                    key = contractToSymbolDescriptor(contract)
                    # logger.info("by KEY Looking up: {} :: {}", contract, key)
                    cached_contract = self.conIdCache.get(key)  # type: ignore

                # logger.info("Using cached contract for {}: [cached {}] :: [requested {}]", contract.conId, cached_contract, contract)
            except ModuleNotFoundError:
                # the pickled contract is from another library we don't have loaded in this environment anymore,
                # so we need to drop the existing bad pickle and re-save it
                try:
                    del self.conIdCache[contract.conId]
                    del self.conIdCache[contractToSymbolDescriptor(contract)]
                except:
                    pass

            # if a manual refresh, never use a cached contract to force a new lookup
            if overwrite:
                cached_contract = None

            if cachedContractCorrupt(cached_contract, contract):
                cached_contract = None

            # if we _found_ a contract (and the contract has an id (just defensively preventing invalid contracts in the cache)),
            # then we don't look it up again.
            if cached_contract and cached_contract.conId:
                # logger.info("Found in cache: {} for {}", cached_contract, contract)
                cached_contracts[cached_contract.conId] = cached_contract
                totalResult[id(contract)] = cached_contract
            else:
                # else, we need to look up this contract before returning.
                # logger.info("Not found in cache: {}", contract)
                # Also, save the ORIGINAL LOOK UP KEY along with the uncached contract so we can
                # _correctly_ map the lookup key to the resolved contract (the resolved contract
                # can (and _will_ have more details than the lookup contract, but we only want to
                # generate the lookup key using the INPUT DETAILS and not the FULL DETAILS because
                # we are going from VAGUE DETAILS -> SPECIFIC DETAILS, and if we use the specific
                # details as the cache key, we can't look it up again because we only have more
                # vague details to start (like: lookup future with YM expiration, but qualify
                # converts YM into YMD, but we didn't look up YMD at first, so we must not cache
                # by using YMD details in the key, etc).

                # always populate unresolved contract for safety in case it can't be resolved
                # we just return it directly as originally provided
                totalResult[id(contract)] = contract

                # also, don't look up Bag contracts because they don't qualify (the legs *inside* the bag qualify instead)
                if not isinstance(contract, Bag):
                    uncached_contracts.append((key, contract))

        # logger.info("CACHED: {} :: UNCACHED: {}", cached_contracts, uncached_contracts)

        # if we have NO UNCACHED CONTRACTS, then we found all input requests in the cache,
        # so we can just return everything we already recorded as a "result contract" (_including_ unqualified bags).
        if not uncached_contracts:
            return [totalResult[id(c)] for c in contracts]

        # For uncached, fetch them from the IBKR lookup system
        got = []
        try:
            # logger.info("Looking up uncached contracts: {}", uncached_contracts)

            # iterate requests in smaller blocks if we have a large input request
            CHUNK = 50

            # Note: "Bag" contracts can NEVER be qualified, so don't ever try them (avoid a timeout wait if bags are attempted)
            for block in range(0, len(uncached_contracts), CHUNK):
                # logger.info("CHECKING: {}", pp.pformat(uncached_contracts))
                logger.info(
                    "Qualifying {} contracts from {} to {}...",
                    len(uncached_contracts),
                    block,
                    block + CHUNK,
                )
                got.extend(
                    await asyncio.wait_for(
                        self.ib.qualifyContractsAsync(
                            *[
                                c
                                for (k, c) in uncached_contracts[block : block + CHUNK]
                            ],
                            returnAll=True,
                        ),
                        timeout=min(
                            6, 2 * len(uncached_contracts[block : block + CHUNK])
                        ),
                    )
                )

            # logger.info("Got: {}", got)
        except Exception as e:
            logger.error(
                "Timeout while trying to qualify {} contracts (sometimes IBKR is slow or the API is offline during nightly restarts) :: {}",
                len(uncached_contracts),
                str(e),
            )

        assert (
            len(got) == len(uncached_contracts)
        ), f"We can't continue caching if we didn't lookup all the contracts! {len(got)=} vs {len(uncached_contracts)=}"

        # iterate resolved contracts and cache them by multiple lookup keys
        for (originalContractKey, requestContract), contract in zip(
            uncached_contracts, got
        ):
            # don't cache continuous futures contracts because those are only for quotes and not trading
            # if contract.secType == "CONTFUT":
            #    continue

            if isinstance(contract, list):
                logger.error(
                    "[{}] contract request returned multiple matches! Can't determine which one of {} to cache: {}",
                    requestContract,
                    len(contract),
                    pp.pformat(contract),
                )
                continue

            # if this lookup was a failure, don't attempt to cache anything...
            if not contract:
                logger.error("No contract ID resolved for: {}", requestContract)
                continue

            if not contract.conId:
                logger.error("No contract ID resolved for: {}", contract)
                continue

            # sometimes we end up with a fully populated Contract() we want to make more specific,
            # so _only_ run this check if type is Contract and _not_ a sub-class of Contract
            # (which is why we must use `type(c) is Contract` and not isinstance(c, Contract))
            if type(contract) is Contract:
                contract = Contract.recreate(contract)

            # final verification check the fields match as expected
            if cachedContractCorrupt(contract, requestContract):
                logger.error("Failed to qualify _actual_ Contract type?")
                continue

            # the `qualifyContractsAsync` modifies the contracts in-place, so we map their
            # id to itself since we replaced it directly.
            # (yes, we _always_ set this even if we didn't resolve a 'conId' because we need
            #  to return _all_ contracts back to the user in the order of their inputs, so
            #  we need every input contract to be in the 'totalResult' map regardless of its final
            #  success/fail resolution value)
            totalResult[id(contract)] = contract

            if type(contract) == Contract:
                # Convert generic 'Contract' to its actual underlying type for proper storage and future retrieval
                contract = Contract.create(**ib_async.util.dataclassAsDict(contract))

                # update key too
                originalContractKey = contractToSymbolDescriptor(contract)

            # Note: this is correct because we want to check for EXACT contract matches and not any subclass of Contract
            if type(contract) == Contract:
                logger.warning(
                    "Not caching because Contract isn't a specific type: {}", contract
                )
                continue

            # Only cache actually qualified contracts with a full IBKR contract ID
            if not contract.conId:
                continue

            # We added double layers of sanity checking here because we had some cache data anomalies where
            # the incorrect contract was cached into the wrong id. These should detect and prevent it from happening
            # again if our logic changes didn't catch all the edge cases.
            if requestContract.strike:
                if requestContract.strike != contract.strike:
                    logger.error(
                        "Why didn't resolved contract have the same strike as the input contract? [request {}] vs [qualified {}]",
                        requestContract,
                        contract,
                    )

                    continue

            cached_contracts[contract.conId] = contract

            # we want Futures contracts to refresh more often because they have
            # embedded expiration dates which may change over time if we are using
            # generic symbol names like "ES" for the next main contract.
            EXPIRATION_DAYS = 5 if isinstance(contract, Future) else 90

            # logger.info("Saving {} -> {}", contract.conId, contract)
            # logger.info("Saving {} -> {}", originalContractKey, contract)

            if False:
                logger.info("Setting {} -> {}", contract.conId, contract)
                logger.info("Setting {} -> {}", originalContractKey, contract)
                logger.info(
                    "Setting {} -> {}",
                    (contract.localSymbol, contract.symbol),
                    contract,
                )

            # cache by id
            assert contract.conId

            # TODO: make the expiration time more clever where it picks an expiration time at 5pm eastern after hours.
            self.conIdCache.set(
                contract.conId, contract, expire=86400 * EXPIRATION_DAYS
            )  # type: ignore

            # also set by Class-Symbol designation as key (e.g. "Index-SPX" or "Future-ES")
            self.conIdCache.set(
                originalContractKey,
                contract,
                expire=86400 * EXPIRATION_DAYS,
            )  # type: ignore

            # also cache the same thing by the most well defined symbol we have
            self.conIdCache.set(
                (contract.localSymbol, contract.symbol),
                contract,
                expire=86400 * EXPIRATION_DAYS,
            )  # type: ignore

        # Return in the same order as the input by combining cached and uncached results.
        # NOTE: we DO NOT MODIFY THE CACHED CONTRACT RESULTS IN-PLACE so you must assign the
        #       return value of this async call to be your new list of contracts.
        result = [totalResult[id(c)] for c in contracts]

        assert len(result) == len(
            contracts
        ), "Why is result length different than request length?"

        # logger.info("Returning contracts: {}", result)
        return result

    def contractsForPosition(
        self, sym, qty: float | None = None
    ) -> list[tuple[Contract, float, float]]:
        """Returns matching portfolio positions as list of (contract, size, marketPrice).

        Note: input 'sym' can be a glob pattern for symbol matching. '?' matches single character, '*' matches any characters.

        Looks up position by symbol name (allowing globs) and returns either provided quantity or total quantity.
        If no input quantity, return total position size.
        If input quantity larger than position size, returned size is capped to max position size.
        """
        portitems = self.ib.portfolio()
        # logger.debug("Current Portfolio is: {}", portitems)

        results = []
        for pi in portitems:
            # Note: using 'localSymbol' because for options, it includes
            # the full OCC-like format, while contract.symbol will just
            # be the underlying equity symbol.
            # Note note: using fnmatch.fnmatch() because we allow 'sym' to
            #            have glob characters for multiple lookups at once!
            # Note 3: options .localSymbols have the space padding, so remove for input compare.
            # TODO: fix temporary hack of OUR symbols being like /NQ but position values dont' have the slash...
            if fnmatch.fnmatch(
                pi.contract.localSymbol.replace(" ", ""), sym.replace("/", "")
            ):
                contract = None
                contract = pi.contract
                position = pi.position

                if qty is None:
                    # if no quantity requested, use entire position
                    foundqty = position
                elif abs(qty) >= abs(position):
                    # else, if qty is larger than position, truncate to position.
                    foundqty = position
                else:
                    # else, use requested quantity but with sign of position
                    foundqty = math.copysign(qty, position)

                # note: '.marketPrice' here is IBKR's "best effort" market price because it only
                #       updates maybe every 30-90 seconds? So (qty * .marketPrice * multiplier) may not represent the
                #       actual live value of the position.
                results.append((contract, foundqty, pi.marketPrice))

        return results

    async def contractForOrderRequest(
        self, oreq: buylang.OrderRequest
    ) -> Contract | None:
        """Return a valid qualified contract for any order request.

        If order request has multiple legs, returns a Bag contract representing the spread.
        If order request only has one symbol, returns a regular future/stock/option contract.

        If symbol(s) in order request are not valid, returns None."""

        if oreq.isSpread():
            return await self.bagForSpread(oreq)

        if oreq.isSingle():
            contract = contractForName(oreq.orders[0].symbol)
            # logger.info("Contracting: {}", contract)

            if contract:
                (contract,) = await self.qualify(contract)

                # only return success if the contract validated
                if contract.conId:
                    return contract

            return None

        # else, order request had no orders...
        return None

    async def bagForSpread(self, oreq: buylang.OrderRequest) -> Contract | Bag | None:
        """Given a multi-leg OrderRequest, return a qualified Bag contract.

        If legs do not validate, returns None and prints errors along the way."""

        # For IBKR spreads ("Bag" contracts), each leg of the spread is qualified
        # then placed in the final contract instead of the normal approach of qualifying
        # the final contract itself (because Bag contracts have Legs and each Leg is only
        # a contractId we have to look up via qualify() individually).
        contracts = [
            contractForName(
                s.symbol,
            )
            for s in oreq.orders
        ]
        contracts = await self.qualify(*contracts)

        # if the bag only has one contract, just return it directly and avoid actually creating a bag.
        if len(contracts) == 1:
            return contracts[0]

        if not all([c.conId for c in contracts]):
            logger.error("Not all contracts qualified! Got: {}", contracts)
            return None

        # trying to match logic described at https://interactivebrokers.github.io/tws-api/spread_contracts.html
        underlyings = ",".join(sorted({x.symbol for x in contracts}))

        # Iterate (in MATCHED PAIRS) the resolved contracts with their original order details
        legs = []

        # We want to order the purchase legs as:
        #   - Option or FuturesOption first (protection first)
        #   - anything else later (underlying later)
        # So, if security type is OPT or FOP, use a smaller value so they SORT FIRST (just negative contract integers so it's always lower!)
        # then regular contract ids for anything else...
        useExchange: str
        for c, o in sorted(
            zip(contracts, oreq.orders),
            key=lambda x: -x[0].conId if x[0].secType in {"OPT", "FOP"} else x[0].conId,
        ):
            useExchange = c.exchange
            leg = ComboLeg(
                conId=c.conId,
                ratio=o.multiplier,
                action="BUY" if o.isBuy() else "SELL",
                exchange=c.exchange,
            )

            legs.append(leg)

        bag = Bag(
            symbol=underlyings,
            comboLegs=legs,
            currency="USD",
        )

        # use SMART if mixing security types.
        bag.exchange = useExchange if (await self.isGuaranteedSpread(bag)) else "SMART"

        return bag

    async def fetchContractExpirations(
        self, contract: Contract, fetchDates: list[str] | None = None
    ):
        """Abstract a dual-use API preferring to use Tradier for strike fetching because IBKR data processing is awful.

        fetchDates is optional if using tradier fetching because tradier returns all expiration dates for a symbol, but
        IBKR requires per-date fetching per contract (and providing wider dates like YYYYMM returns an entire month of
        chains, but also invokes IBKR data pacing limitations because their data formats are big and they are slow)."""

        # only run for regular Options on Stock-like things (SPX counts as "Stock" here for our lookups too)
        if os.getenv("TRADIER_KEY") and isinstance(contract, Stock):
            if found := await getExpirationsFromTradier(contract.symbol):
                return found

        strikes: dict[str, list[float]] = defaultdict(list)
        allStrikes: dict[str, list[float]] = dict()

        assert fetchDates
        fetchDates = sorted(set(fetchDates))
        logger.info("[{}] Requested dates: {}", contract, fetchDates)

        if False:
            # TODO: use this as a fallback if the regualr lookups don't work.... mainly for indexes?
            # pre-lookup
            everything = await asyncio.wait_for(
                self.ib.reqSecDefOptParamsAsync(
                    contract.symbol, "CME", contract.secType, contract.conId
                ),
                timeout=10,
            )

            logger.info(
                "Also found: {}",
                pp.pformat(everything),
            )

            exchanges = sorted(set([x.exchange for x in everything]))
            logger.info("Valid exchanges: {}", pp.pformat(exchanges))

            if contract.secType == "IND":
                if exchanges:
                    contract.exchange = exchanges[0]

        for date in fetchDates:
            contract.lastTradeDateOrContractMonth = date
            chainsExact = await asyncio.wait_for(
                self.ib.reqContractDetailsAsync(contract), timeout=180
            )

            # group strike results by date
            logger.info(
                "[{}{}] Populating strikes...",
                contract.localSymbol,
                contract.lastTradeDateOrContractMonth,
            )

            for d in sorted(
                chainsExact,
                key=lambda k: k.contract.lastTradeDateOrContractMonth,  # type: ignore
            ):
                assert d.contract
                strikes[d.contract.lastTradeDateOrContractMonth].append(
                    d.contract.strike
                )

            # cleanup the results because they were received in an
            # arbitrary order, but we want them sorted for bisecting
            # and just nice viewing.
            allStrikes |= {k: sorted(set(v)) for k, v in strikes.items()}

        return allStrikes

    async def isGuaranteedSpread(self, bag: Contract) -> bool:
        # Note: only STK+OPT or OPT+OPT spreads are guaranteed. Other instrument bags are but not executed atomically and may partially execute.
        # Also note: if using 'SMART' due to conflicting instrument types, the Order for execution must be marked NonGuaranteed.

        # We need to fetch the contracts from the contract ids since this is just a bag with ids and we don't know contract types... thanks API.
        legs = await self.qualify(*[Contract(conId=x.conId) for x in bag.comboLegs])

        secTypes = set([x.secType for x in legs])

        # Single instrument routing is safe
        if len(secTypes) == 0:
            return True

        # a spread is guaranteed only for bags with stock and option combinations.
        # So, if we have more than one security type, but everything is STK and OPT, we are still Guaranteed.
        if len(secTypes) > 1 and len({"STK", "OPT"} - secTypes) > 0:
            return False

        # else, we just checked all legs are either STK or OPT, so we are guaranteed
        return True

    async def contractForOrderSide(self, order, contract: Contract) -> Contract:
        """If required, return a copy of 'contract' with legs in the direction of the order.

        This is only reuqired for non-guaranteed spreads where we need to execute protection before acquiring or selling underlyings.
        """

        if isinstance(contract, Bag):
            # for BUYING, we need the protective options FIRST
            # for SELLING, we need the protective options SECOND

            # return a COPY of the contract here because we need different contracts for different order sides.
            contract = copy.copy(contract)

            def optionScoreFirst(leg, secType):
                """If security type is an option, use a lower sort key, else use a higher sort key for non-options"""
                return -leg.conId if secType in {"OPT", "FOP"} else leg.conId

            async def sortLegsDirection(legs, optionsFirstMultiplier=1):
                # option legs are only contract ids, so we need to look up the actual contracts to figure out what each leg actually is
                cs = await self.qualify(*[Contract(conId=x.conId) for x in legs])

                # we need to sort the legs, but legs don't have details, so we combine the legs with the contracts, sort the pairs,
                # then we un-combine them and return only the legs (in correct order now)
                resorted = sorted(
                    zip(cs, legs),
                    # this basically just keeps options first if multiper == 1 or options last if multiplier == -1
                    key=lambda x: optionsFirstMultiplier
                    * optionScoreFirst(x[1], x[0].secType),
                )

                return list(zip(*resorted))[1]

            async def sortLegsOptionFirst(legs):
                # we need to look up ids to contracts to figure out what they are... thanks IBKR API
                return await sortLegsDirection(legs, 1)

            async def sortLegsOptionLast(legs):
                return await sortLegsDirection(legs, -1)

            match order.action:
                case "BUY":
                    # sort protection FIRST
                    contract.comboLegs = await sortLegsOptionFirst(contract.comboLegs)
                case "SELL":
                    # sort protection LAST
                    contract.comboLegs = await sortLegsOptionLast(contract.comboLegs)

        return contract

    async def addNonGuaranteeTagsIfRequired(self, contract, *reqorders):
        if isinstance(contract, Bag):
            # only check contract once up front if we need to actually use the 'not guaranteed' fields
            # (non-guaranteed means IBKR may execute the spread sequentially instead of as a single price-locked unit)
            isGuaranteed = await self.isGuaranteedSpread(contract)

            # if contract is guaranteed (default behavior), it executes as expected so we don't need to modify the orders
            if isGuaranteed:
                return

            # else, we need to add custom "non-guaranteed" config flags/tags to each order
            for order in reqorders:
                # only send orders if the order is populated (we get _all_ potential limit/profit/loss orders here, but the bracket may not be defined)
                if order:
                    orders.markOrderNotGuaranteed(order)
