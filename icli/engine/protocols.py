"""Narrow protocols for cross-module method calls.

These protocols define the minimal interfaces that engine modules need
from their peers, avoiding circular imports and direct coupling to
IBKRCmdlineApp.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ib_async import Bag, Contract, Order


@runtime_checkable
class ContractQualification(Protocol):
    """Contract qualification and resolution."""

    async def qualify(self, *contracts, overwrite: bool = False) -> list[Contract]: ...
    async def contractForOrderRequest(self, oreq) -> Contract | None: ...
    async def bagForSpread(self, oreq) -> Contract | Bag | None: ...
    async def contractForOrderSide(self, order, contract: Contract) -> Contract: ...
    async def addNonGuaranteeTagsIfRequired(
        self, contract: Contract, *reqorders: Order
    ) -> None: ...


@runtime_checkable
class PortfolioAccess(Protocol):
    """Portfolio position queries."""

    def decimals(self, contract: Contract) -> int: ...
    def multiplier(self, contract: Contract) -> float: ...
    def quantityForContract(self, contract: Contract) -> float: ...
    def quantityForAmount(
        self, contract: Contract, amount: float, limitPrice: float
    ) -> float: ...


@runtime_checkable
class QuoteAccess(Protocol):
    """Quote subscription and lookup."""

    def currentQuote(
        self, sym, show: bool = True
    ) -> tuple[float | None, float | None]: ...
    def addQuoteFromContract(self, contract: Contract) -> None: ...
