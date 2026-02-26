"""Coordinated application clock for consistent time across engine modules."""
from __future__ import annotations

import dataclasses
import datetime

import whenever


@dataclasses.dataclass
class AppClock:
    """Shared time source updated once per toolbar render cycle.

    Instead of each module calling datetime.now() independently,
    the toolbar calls tick() at the top of each render, and all
    modules read the same snapshot.
    """

    now: whenever.ZonedDateTime = dataclasses.field(
        default_factory=lambda: whenever.ZonedDateTime.now("US/Eastern")
    )
    nowpy: datetime.datetime = dataclasses.field(
        default_factory=lambda: datetime.datetime.now()
    )

    def tick(self) -> None:
        """Advance to current wall-clock time."""
        self.now = whenever.ZonedDateTime.now("US/Eastern")
        self.nowpy = datetime.datetime.now()
