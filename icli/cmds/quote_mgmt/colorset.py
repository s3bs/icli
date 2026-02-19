"""Command: colorset

Category: Quote Management
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mutil.dispatch import DArg

from icli.cmds.base import IOp, command

if TYPE_CHECKING:
    pass


@command(names=["colorset"])
@dataclass
class IOpColors(IOp):
    """Set toolbar color scheme by theme name or direct style string.

    Available themes:
      default  — terminal default colors (reversed white bg)
      solar1   — Solarized dark (fg:#002B36 bg:#839496)

    Custom style example:
      colorset fg:#708C4C bg:#33363D

    Each theme includes a matching alternating row color for quote rows.
    Override manually with: set altrow_color #hexcolor
    Disable alternating rows: set altrow_color off"""

    style: str = field(init=False)

    def argmap(self):
        return [DArg("style")]

    async def run(self):
        cacheKey = ("colors", f"client-{self.state.clientId}")
        cacheVal = dict(toolbar=self.style, altrow_color=self.state.altrowColor)

        self.state.updateToolbarStyle(self.style)

        # save after updateToolbarStyle so altrowColor reflects the theme
        cacheVal["altrow_color"] = self.state.altrowColor
        self.cache.set(cacheKey, {"colors": cacheVal})  # type: ignore
