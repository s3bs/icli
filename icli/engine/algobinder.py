"""WebSocket client for external algo data feed ingestion."""

from __future__ import annotations

import asyncio
import json
import platform
import types
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import websockets
from loguru import logger

# Use orjson for CPython (faster), stdlib json for PyPy
ourjson: types.ModuleType
if platform.python_implementation() == "CPython":
    try:
        import orjson
        ourjson = orjson
    except ImportError:
        ourjson = json
else:
    ourjson = json


@dataclass(slots=True)
class AlgoBinder:
    """Consume an external data feed and save results to a dot-addressable dict hierarchy."""

    url: str | None = "ws://127.0.0.1:4442/bars"

    # where we save our results for reading
    data: Any = field(default_factory=lambda: defaultdict(dict))

    # active websocket connection (if any)
    activeWS: Any | None = None

    def read(self, depth: str) -> Any | None:
        """Read something from the saved data in dotted string format.

        e.g. Requesting read("AAPL.30.lb.vwap.5.sma") would return the value
             for: {"AAPL": {"30": {"lb": {"vwap": {"5": {"sma": VALUE}}}}}}

        Then for detecting 5/10 crossovers you could do math on:
             read("AAPL.30.lb.vwap.5.sma") > read("AAPL.30.lb.vwap.10.sma")

        Due to how we receive JSON dicts, we expect ALL key types to be strings,
        so just splitting the input string on dots should work (if all your keys exist).

        Of course, your field names must not include dots in keys anywhere or the entire lookup will break.

        If you provide an invalid or non-existing path, the result is None because the depth will fail.
        """

        val = self.data
        for level in depth.split("."):
            if (val := val.get(level)) is None:
                break

        return val

    async def datafeed(self) -> None:
        """Generator for returning collecting external API results locally.

        We assume the external API is returning an N-level nested dict with layout like:
            {
                symbol1: {
                            duration1: {field1: ...},
                            duration2: {field1: ...},
                            ...
                         },
                symbol2: {
                            duration1: {field1: ...},
                            duration2: {field1: ...},
                            ...
                         },
                ...
            }
        The inner (second-level) 'duration' dicts don't need to be populated on
        every new data update because we just replace each 'duration' dict under
        each symbol on every update (i.e. we don't replace each _full symbol_ on
        each update, but rather we just replace _individual_ 2nd level "duration"
        keys inside each symbol dict on each update).
        """

        assert self.url
        logger.info("[Algo Binder] Connecting to: {}", self.url)

        try:
            # this async context manager automatically handles reconnecting when required
            async for ws in websockets.connect(
                self.url,
                ping_interval=10,
                ping_timeout=30,
                open_timeout=2,
                close_timeout=1,
                max_queue=2**32,
                # big read limit... some of our inbound data is big.
                read_limit=1024 * 2**20,
                # Set max size to "unlimited" (via None) because our initial data size can be 5+ MB for a full symbol+duration+algo+lookback state.
                max_size=None,
                compression=None,
                user_agent_header=None,
            ):
                self.activeWS = ws
                logger.info("[Algo Binder :: {}] Connected!", self.url)

                try:
                    # logger.info("Waiting for WS message...")
                    async for msg in ws:
                        # logger.info("Got msg: {:,} bytes", len(msg))
                        for symbol, durations in ourjson.loads(msg).items():
                            # Currently we expect each symbol to have about 8 first-level
                            # durations representing bar sizes in seconds with something like:
                            # 15, 35, 55, 90, 180, 300, 900, 1800
                            # Also note: we do not REPLACE all symbol durations each update, because
                            #            durations are not updated all at once. One update packet
                            #            may have durations 15, 35 while another may have 90 and 180,
                            #            so we must MERGE new durations into the symbol for each update.
                            # logger.info("Got data: {} :: {}", symbol, durations)
                            self.data[symbol] |= durations
                except websockets.ConnectionClosed:
                    # this reconnects the client with an exponential backoff delay
                    # (because websockets library v10 added async reconnect on continue)
                    logger.error(
                        "[Algo Binder :: {}] Connection dropped, reconnecting...",
                        self.url,
                    )
                    continue
                finally:
                    self.activeWS = None

                logger.error("How did we exit the forever loop?")
        except (KeyboardInterrupt, asyncio.CancelledError, SystemExit):
            # be quiet when manually exiting
            logger.error("[{}] Exiting!", self.url)
            return
        except:
            # on any error, reconnect
            logger.exception("[{}] Can't?", self.url)
            await asyncio.sleep(1)
        finally:
            self.activeWS = None
