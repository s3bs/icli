"""Portfolio position queries extracted from IBKRCmdlineApp."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ib_async import Bag, Contract, Crypto, FuturesOption, Option, Position
from loguru import logger

if TYPE_CHECKING:
    from ib_async import IB
    import diskcache


class PortfolioQueries:
    """Portfolio and position query methods.

    Dependencies injected at construction:
    - ib: IB connection
    - conIdCache: contract ID cache
    - idb: instrument database
    """

    def __init__(self, ib, accountId: str, conIdCache, idb):
        self.ib = ib
        self.accountId = accountId
        self.conIdCache = conIdCache
        self.idb = idb

    @property
    def positionsDB(self) -> dict[int, Position]:
        """Return a dict holding a mapping of contract ids to contracts.

        This currently uses some ib_async internals which we should have better insight to.
        """
        positionsDB = self.ib.wrapper.positions[self.accountId]
        return positionsDB

    def quantityForContract(self, contract: Contract) -> float:
        """Returns positive numbers for longs, negative counts for shorts, and 0 for no position."""

        if pos := self.positionsDB.get(contract.conId):
            pos.position

        return 0

    def averagePriceForContract(self, contract: Contract) -> float:
        """Using live portfolio data, return current average cost for position having contract ids.

        We accept multiple contract ids to allow for this to generate the combined cost basis for spreads too."""

        cost = 0.0

        # fetch all relevant contract ids for the contract provided
        # (e.g. spreads/bags have inner contract ids, but everything else is just a single top-level conId)
        # (also ignoring type here because we know each of these elements has .conId even though they are different types)
        src = contract.comboLegs if contract.comboLegs else [contract]  # type: ignore
        conIds = [x.conId for x in src]

        positionsDB = self.positionsDB
        # TODO: update ib_async API to allow direct retrieval of portfolio or positions by contract id instead of fetching all then iterating.
        for conId in conIds:
            # FIX: using internal API hack to get direct access to contract-level account row fetching
            row = positionsDB.get(conId)

            # missing rows is the same as a $0 cost basis because we don't hold the position.
            # (the math still works out; if going short new entry is < 0; if going long, new entry is > 0)
            if row is None:
                return 0

            # API TRICK: "position" objects have 'avgCost' while "portfolio" objects have 'averageCost'
            cost += row.avgCost

        return cost

    def multiplier(self, contract: Contract) -> float | int:
        """Abstraction to get the multiplier for any contract.

        Why do we need this?

        Equity symbols have no multipler, so we use 1.
        Options and Futures have a definied multiplier.
        Bags / Spreads have no multiplier defined at the contract leve, so we need to look _inside_ the bag to find the multiplier.
        """
        if isinstance(contract, (Option, FuturesOption)):
            mul = float(contract.multiplier or 1.0)
        elif isinstance(contract, Bag):
            # steal multiplier of first thing in contract. we assume it's okay? This would be wrong for buy-write bags and is only valid for spreads.
            innerContract = self.conIdCache.get(contract.comboLegs[0].conId)
            mul = float(innerContract.multiplier or 1.0)
        else:
            mul = float(contract.multiplier or 1.0)

        if mul == (imul := int(mul)):
            mul = imul

        return mul

    def quantityForAmount(
        self,
        contract: Contract,
        amount,
        limitPrice,
    ) -> int | float:
        """Return valid quantity for contract using total dollar amount 'amount'.

        Also compensates for limitPrice being a contract quantity.

        Also compensates for contracts allowing fractional quantities (Crypto)
        versus only integer quantities (everything else)."""

        # For options, the multipler is PART OF THE COST OF BUYING because a $0.15 option costs $15 to buy,
        # but for futures, the multiplier is NOT PART OF THE BUY COST because buying futures only costs
        # future margin which is much less than the quoted contract price (but the futures margin is
        # technically aorund 4% of the total value because a $4,000 MES contract has a 5 multipler so
        # your $4,000 MES contract is holding $20,000 notional on a $1,700 margin requirement).
        mul = self.multiplier(contract)

        assert mul > 0

        # total spend amount divided by price of thing to buy == how many things to buy
        # (rounding to fix IBKR error for fractional qty: "TotalQuantity field cannot contain more than 8 decimals")
        qty = float(amount) / (float(limitPrice) * float(mul))
        if qty <= 0:
            logger.error(
                "Sorry, your calculated quantity is {:,.2f} so we can't order anything!",
                qty,
            )
            return 0

        if isinstance(contract, Crypto):
            # only crypto orders support fractional quantities over the API.
            qty = round(qty, 8)
        else:
            # TODO: if IBKR ever enables fractional shares over the API,
            #       we can make the above Crypto check for (Crypto, Stock).
            qty = math.floor(qty)

        return qty

    def decimals(self, contract: Contract) -> int:
        """How many decimal places should a contract use?"""
        # if no decimal specification found, default to 2.
        if (digits := self.idb.decimals(contract)) is None:
            return 2

        # always use 2 digits for display even if only 1 digit of precision is required
        if digits == 1:
            return 2

        return digits
