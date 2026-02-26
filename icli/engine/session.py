"""Write-once session configuration for engine modules."""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class SessionConfig:
    """Scalar session state shared across engine modules.

    These values are set once (or rarely) during connection setup
    and read frequently by engine modules. Mutable fields use
    simple attribute access rather than property indirection.
    """

    accountId: str = ""
    clientId: int = 0
    connected: bool = False
    isSandbox: bool | None = None
