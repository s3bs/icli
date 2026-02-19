"""Command: taskcancel

Category: Task Management
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command

if TYPE_CHECKING:
    pass


@command(names=["taskcancel"])
@dataclass
class IOpTaskCancel(IOp):
    """Cancel one or more running background tasks by ID."""

    ids: list[int] = field(init=False)

    def argmap(self):
        return [DArg("*ids", convert=lambda xs: list(map(int, xs)))]

    async def run(self):
        for taskid in self.ids:
            logger.info("[{}] Stopping task...", taskid)
            self.state.task_stop_id(taskid)
