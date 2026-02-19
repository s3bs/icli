"""Command: tasklist

Category: Task Management
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from icli.cmds.base import IOp, command

if TYPE_CHECKING:
    pass


@command(names=["tasklist"])
@dataclass
class IOpTaskList(IOp):
    """Display all current and running background tasks."""

    def argmap(self):
        return []

    async def run(self):
        self.state.task_report()
