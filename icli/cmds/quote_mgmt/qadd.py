"""Command: qadd

Category: Quote Management
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command

if TYPE_CHECKING:
    pass


@command(names=["qadd"])
@dataclass
class IOpQuoteAppend(IOp):
    """Add symbols to a named quote group and populate for live tracking."""

    group: str = field(init=False)
    symbols: list[str] = field(init=False)

    def argmap(self):
        return [DArg("group"), DArg("*symbols")]

    async def run(self):
        cacheKey = ("quotes", self.group)
        symbols = self.cache.get(cacheKey)  # type: ignore
        if not symbols:
            logger.error(
                "[{}] No quote group found. Creating new quote group!", self.group
            )
            symbols = set()

        self.cache.set(cacheKey, symbols | set(self.symbols))  # type: ignore
        repopulate = [f'"{x}"' for x in self.symbols]
        await self.runoplive(
            "add",
            " ".join(repopulate),
        )
