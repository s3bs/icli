"""
Futures contract specification resolver with dynamic fetching via IBKR TWS API.

This module provides:
1. Dynamic contract details fetching via ib_async (TWS API)
2. Static fallback data for offline/fast startup
3. Validation between static and live data
4. Caching to minimize API calls

Usage:
    from icli.engine.resolver import FuturesResolver

    # Simple on-demand usage
    resolver = FuturesResolver()
    await resolver.connect()

    # Get contract details (fetches from TWS, falls back to static)
    details = await resolver.get_contract_details("ES")
    print(f"ES minTick: {details.minTick}")  # Decimal('0.25')

    # Validate static data against live
    mismatches = await resolver.validate_static_data()

    # Refresh and export updated static data
    updated = await resolver.refresh_all_tick_details()
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from loguru import logger

# ib_async is the successor to ib_insync
import ib_async as ibapi
from ib_async import IB, Future, ContractDetails

# Import static data from icli.engine.exchanges
from icli.engine.exchanges import FUTS_EXCHANGE, FUTS_TICK_DETAIL, FutureSymbol, FutureDetail


# =============================================================================
# Dynamic Tick Detail (replaces broken buildTickDetail)
# =============================================================================

@dataclass
class DynamicFutureDetail:
    """
    Futures contract details fetched dynamically from IBKR TWS API.

    This replaces the broken buildTickDetail() function that scraped
    futuresonline.com (now offline).
    """
    symbol: str
    exchange: str
    tradingClass: str
    currency: str

    # Tick and size info
    minTick: Decimal
    sizeIncrement: Decimal
    multiplier: str

    # Contract specs
    contractMonth: str  # Available expiration months
    timeZoneId: str
    tradingHours: str
    liquidHours: str

    # Metadata
    longName: Optional[str] = None
    marketName: Optional[str] = None
    evRule: Optional[str] = None
    evMultiplier: Optional[int] = None

    # Computed fields
    @property
    def decimals(self) -> int:
        """Number of decimal places for this contract."""
        if self.minTick == 0:
            return 0
        return max(0, -self.minTick.as_tuple().exponent)

    @property
    def tickValue(self) -> str:
        """
        Notional value per tick.

        Note: For CBOT contracts, the multiplier is often in cents,
        so tick value = minTick * multiplier * 100.
        For most others, tick value = minTick * multiplier.
        """
        try:
            mult = Decimal(self.multiplier)
            value = self.minTick * mult

            # CBOT special handling (most treasury and ag contracts)
            if self.exchange in ("CBOT", "XCBT"):
                value = value * Decimal("100")

            return f"${value:.2f}"
        except:
            return f"${self.minTick} x {self.multiplier}"

    def to_legacy_format(self, name: str = "", size: str = "", months: str = "") -> FutureDetail:
        """Convert to legacy FutureDetail format for compatibility."""
        return FutureDetail(
            symbol=self.tradingClass,
            exchange=self.exchange,
            name=name or self.longName or self.symbol,
            size=size or f"{self.multiplier} {self.currency}",
            months=months or self.contractMonth,
            tick=self.minTick,
            decimals=self.decimals,
            valuePerTick=self.tickValue
        )


class FuturesResolver:
    """
    Resolves futures contract specifications dynamically via IBKR TWS API.

    Features:
    - Dynamic fetching via ib_async (reqContractDetails)
    - Static fallback when TWS is unavailable
    - Caching to minimize API calls
    - Validation of static vs live data
    - Export functionality for updating static data

    Example:
        resolver = FuturesResolver()
        await resolver.connect()

        # Get details for ES futures
        details = await resolver.get_contract_details("ES")

        # Validate static data
        issues = await resolver.validate_static_data()

        resolver.disconnect()
    """

    def __init__(
        self,
        static_exchange: dict[str, FutureSymbol] | None = None,
        static_tick_detail: dict[str, FutureDetail] | None = None,
        tws_host: str = '127.0.0.1',
        tws_port: int = 7497,
        client_id: int = 1,
        cache_file: str | Path | None = None
    ):
        """
        Initialize the resolver.

        Args:
            static_exchange: Static FUTS_EXCHANGE data (symbol -> exchange info)
            static_tick_detail: Static FUTS_TICK_DETAIL data (symbol -> tick info)
            tws_host: TWS or Gateway host
            tws_port: TWS (7497) or Gateway (4001/4002) port
            client_id: Unique client ID for this connection
            cache_file: Optional file path for caching fetched details
        """
        self._static_exchange = static_exchange or dict(FUTS_EXCHANGE)
        self._static_tick_detail = static_tick_detail or dict(FUTS_TICK_DETAIL)
        self._tws_host = tws_host
        self._tws_port = tws_port
        self._client_id = client_id
        self._cache_file = Path(cache_file) if cache_file else None

        self._ib: IB | None = None
        self._cache: dict[str, DynamicFutureDetail] = {}
        self._connected = False

        # Load cache from file if available
        if self._cache_file and self._cache_file.exists():
            self._load_cache()

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to TWS/Gateway."""
        if self._connected:
            return

        self._ib = IB()
        try:
            await self._ib.connectAsync(
                self._tws_host,
                self._tws_port,
                clientId=self._client_id
            )
            self._connected = True
            logger.info(f"Connected to TWS at {self._tws_host}:{self._tws_port}")
        except Exception as e:
            logger.warning(f"Failed to connect to TWS: {e}")
            self._ib = None
            raise

    def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._ib:
            self._ib.disconnect()
            self._ib = None
            self._connected = False
            logger.info("Disconnected from TWS")

    @property
    def is_connected(self) -> bool:
        """Check if connected to TWS."""
        return self._connected and self._ib is not None

    # -------------------------------------------------------------------------
    # Core Functionality: Get Contract Details
    # -------------------------------------------------------------------------

    async def get_contract_details(
        self,
        symbol: str,
        exchange: str | None = None,
        currency: str = "USD",
        use_static_fallback: bool = True,
        use_cache: bool = True
    ) -> DynamicFutureDetail:
        """
        Get contract details for a futures symbol.

        Resolution order:
        1. Check cache (if use_cache)
        2. Query TWS (if connected)
        3. Fall back to static data (if use_static_fallback)

        Args:
            symbol: Futures symbol (e.g., "ES", "CL", "ZN")
            exchange: Exchange (e.g., "CME", "NYMEX"). If None, looks up in static data.
            currency: Currency code (default: "USD")
            use_static_fallback: Whether to fall back to static data if TWS fails
            use_cache: Whether to use cached results

        Returns:
            DynamicFutureDetail with contract specifications

        Raises:
            KeyError: If symbol not found and no fallback
            ConnectionError: If TWS not connected and no fallback
        """
        # Check cache
        cache_key = f"{symbol}.{exchange or 'AUTO'}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Determine exchange from static data if not provided
        if exchange is None:
            if symbol in self._static_exchange:
                exchange = self._static_exchange[symbol].exchange
                currency = self._static_exchange[symbol].currency
            else:
                raise KeyError(f"Unknown symbol '{symbol}' - no exchange mapping in static data")

        # Try TWS if connected
        if self.is_connected:
            try:
                details = await self._fetch_from_tws(symbol, exchange, currency)
                self._cache[cache_key] = details
                self._save_cache()
                return details
            except Exception as e:
                logger.warning(f"TWS query failed for {symbol}: {e}")
                if not use_static_fallback:
                    raise

        # Fall back to static data
        if use_static_fallback:
            return self._get_from_static(symbol, exchange)

        raise ConnectionError(f"Not connected to TWS and static fallback disabled for {symbol}")

    async def _fetch_from_tws(
        self,
        symbol: str,
        exchange: str,
        currency: str
    ) -> DynamicFutureDetail:
        """Fetch contract details from TWS via ib_async."""
        if not self._ib:
            raise ConnectionError("Not connected to TWS")

        # Create futures contract
        # Note: We don't specify lastTradeDateOrContractMonth to get generic contract info
        contract = Future(
            symbol=symbol,
            exchange=exchange,
            currency=currency
        )

        # Request contract details
        details_list = await self._ib.reqContractDetailsAsync(contract)

        if not details_list:
            # Try with tradingClass if symbol differs
            if symbol in self._static_exchange:
                tc = self._static_exchange[symbol].tradingClass
                if tc != symbol:
                    contract = Future(
                        symbol=symbol,
                        exchange=exchange,
                        currency=currency,
                        tradingClass=tc
                    )
                    details_list = await self._ib.reqContractDetailsAsync(contract)

        if not details_list:
            raise ValueError(f"No contract details found for {symbol} on {exchange}")

        # Use first result (there may be multiple for different expirations)
        cd = details_list[0]

        return self._contract_details_to_dynamic(cd)

    def _contract_details_to_dynamic(self, cd: ContractDetails) -> DynamicFutureDetail:
        """Convert IBKR ContractDetails to our DynamicFutureDetail."""
        return DynamicFutureDetail(
            symbol=cd.contract.symbol,
            exchange=cd.contract.exchange,
            tradingClass=cd.contract.tradingClass,
            currency=cd.contract.currency,
            minTick=Decimal(str(cd.minTick)),
            sizeIncrement=Decimal(str(cd.sizeIncrement)) if cd.sizeIncrement else Decimal("1"),
            multiplier=cd.contract.multiplier or "1",
            contractMonth=cd.contractMonth or "",
            timeZoneId=cd.timeZoneId or "",
            tradingHours=cd.tradingHours or "",
            liquidHours=cd.liquidHours or "",
            longName=cd.longName,
            marketName=cd.marketName,
            evRule=cd.evRule,
            evMultiplier=cd.evMultiplier
        )

    def _get_from_static(self, symbol: str, exchange: str) -> DynamicFutureDetail:
        """Get contract details from static fallback data."""
        # Check if we have tick detail data
        if symbol in self._static_tick_detail:
            td = self._static_tick_detail[symbol]
            return DynamicFutureDetail(
                symbol=td.symbol,
                exchange=td.exchange,
                tradingClass=td.symbol,
                currency="USD",  # Assumed from static data
                minTick=td.tick,
                sizeIncrement=Decimal("1"),
                multiplier=td.size.split()[0] if td.size else "1",
                contractMonth=td.months,
                timeZoneId="",
                tradingHours="",
                liquidHours="",
                longName=td.name
            )

        # Check if we have exchange mapping
        if symbol in self._static_exchange:
            fs = self._static_exchange[symbol]
            raise KeyError(
                f"Symbol '{symbol}' has exchange mapping ({fs.exchange}) but no tick detail. "
                f"Connect to TWS to fetch full details."
            )

        raise KeyError(f"Unknown symbol: {symbol}")

    # -------------------------------------------------------------------------
    # Batch Operations
    # -------------------------------------------------------------------------

    async def get_multiple_details(
        self,
        symbols: list[str],
        fail_fast: bool = False
    ) -> dict[str, DynamicFutureDetail]:
        """
        Get contract details for multiple symbols.

        Args:
            symbols: List of futures symbols
            fail_fast: If True, raise on first error. If False, continue and log errors.

        Returns:
            Dict mapping symbol -> DynamicFutureDetail
        """
        results = {}

        for symbol in symbols:
            try:
                details = await self.get_contract_details(symbol)
                results[symbol] = details
                logger.debug(f"Fetched details for {symbol}")
            except Exception as e:
                if fail_fast:
                    raise
                logger.warning(f"Failed to fetch {symbol}: {e}")

        return results

    async def refresh_all_tick_details(
        self,
        symbols: list[str] | None = None,
        batch_delay: float = 0.1
    ) -> dict[str, DynamicFutureDetail]:
        """
        Refresh tick details for all or specified symbols.

        This replaces the broken buildTickDetail() function.

        Args:
            symbols: List of symbols to refresh. If None, refreshes all in FUTS_EXCHANGE.
            batch_delay: Delay between requests to avoid rate limiting

        Returns:
            Dict mapping symbol -> DynamicFutureDetail
        """
        if symbols is None:
            symbols = list(self._static_exchange.keys())

        if not self.is_connected:
            await self.connect()

        results = {}
        total = len(symbols)

        for i, symbol in enumerate(symbols, 1):
            try:
                details = await self.get_contract_details(symbol, use_static_fallback=False)
                results[symbol] = details
                logger.info(f"[{i}/{total}] Refreshed: {symbol}")
            except Exception as e:
                logger.warning(f"[{i}/{total}] Failed {symbol}: {e}")

            # Small delay to avoid overwhelming TWS
            if batch_delay > 0 and i < total:
                await asyncio.sleep(batch_delay)

        self._save_cache()
        return results

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    async def validate_static_data(
        self,
        symbols: list[str] | None = None,
        tolerance: Decimal = Decimal("0.0001")
    ) -> dict[str, dict[str, Any]]:
        """
        Compare static tick data against live TWS data.

        Args:
            symbols: Symbols to validate. If None, validates symbols in FUTS_TICK_DETAIL.
            tolerance: Tolerance for tick size comparison

        Returns:
            Dict mapping symbol -> validation results
            {
                "ES": {
                    "status": "match" | "mismatch" | "error",
                    "static_tick": Decimal("0.25"),
                    "live_tick": Decimal("0.25"),
                    "diff": Decimal("0"),
                    "error": None
                },
                ...
            }
        """
        if symbols is None:
            symbols = list(self._static_tick_detail.keys())

        if not self.is_connected:
            await self.connect()

        results = {}

        for symbol in symbols:
            static = self._static_tick_detail.get(symbol)
            if not static:
                results[symbol] = {"status": "no_static", "error": "Not in static data"}
                continue

            try:
                live = await self.get_contract_details(
                    symbol,
                    exchange=static.exchange,
                    use_static_fallback=False
                )

                diff = abs(live.minTick - static.tick)

                if diff <= tolerance:
                    status = "match"
                else:
                    status = "mismatch"

                results[symbol] = {
                    "status": status,
                    "static_tick": static.tick,
                    "live_tick": live.minTick,
                    "static_decimals": static.decimals,
                    "live_decimals": live.decimals,
                    "diff": diff,
                    "exchange": static.exchange,
                    "error": None
                }

            except Exception as e:
                results[symbol] = {
                    "status": "error",
                    "static_tick": static.tick if static else None,
                    "live_tick": None,
                    "diff": None,
                    "error": str(e)
                }

        return results

    # -------------------------------------------------------------------------
    # Export / Code Generation
    # -------------------------------------------------------------------------

    def generate_tick_detail_code(
        self,
        details: dict[str, DynamicFutureDetail],
        output_file: str | Path | None = None
    ) -> str:
        """
        Generate Python code for FUTS_TICK_DETAIL dictionary.

        This can be used to update the static data in exchanges.py.

        Args:
            details: Dict of symbol -> DynamicFutureDetail
            output_file: Optional file to write the code to

        Returns:
            Generated Python code as string
        """
        lines = [
            "# Auto-generated futures tick details",
            f"# Generated: {datetime.now().isoformat()}",
            "from decimal import Decimal",
            "",
            "FUTS_TICK_DETAIL = {"
        ]

        for symbol, d in sorted(details.items()):
            lines.append(f'    "{d.tradingClass}": FutureDetail(')
            lines.append(f'        symbol="{d.tradingClass}",')
            lines.append(f'        exchange="{d.exchange}",')
            lines.append(f'        name="{d.longName or d.symbol}",')
            lines.append(f'        size="{d.multiplier} {d.currency}",')
            lines.append(f'        months="{d.contractMonth}",')
            lines.append(f'        tick=Decimal("{d.minTick}"),')
            lines.append(f'        decimals={d.decimals},')
            lines.append(f'        valuePerTick="{d.tickValue}",')
            lines.append("    ),")

        lines.append("}")
        code = "\n".join(lines)

        if output_file:
            Path(output_file).write_text(code)
            logger.info(f"Wrote tick details to {output_file}")

        return code

    # -------------------------------------------------------------------------
    # Caching
    # -------------------------------------------------------------------------

    def _load_cache(self) -> None:
        """Load cached details from file."""
        if not self._cache_file:
            return

        try:
            with open(self._cache_file, 'r') as f:
                data = json.load(f)

            for key, item in data.items():
                self._cache[key] = DynamicFutureDetail(
                    symbol=item['symbol'],
                    exchange=item['exchange'],
                    tradingClass=item['tradingClass'],
                    currency=item['currency'],
                    minTick=Decimal(item['minTick']),
                    sizeIncrement=Decimal(item['sizeIncrement']),
                    multiplier=item['multiplier'],
                    contractMonth=item['contractMonth'],
                    timeZoneId=item['timeZoneId'],
                    tradingHours=item['tradingHours'],
                    liquidHours=item['liquidHours'],
                    longName=item.get('longName'),
                    marketName=item.get('marketName'),
                    evRule=item.get('evRule'),
                    evMultiplier=item.get('evMultiplier')
                )

            logger.info(f"Loaded {len(self._cache)} cached entries from {self._cache_file}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self) -> None:
        """Save cached details to file."""
        if not self._cache_file:
            return

        try:
            data = {}
            for key, d in self._cache.items():
                data[key] = {
                    'symbol': d.symbol,
                    'exchange': d.exchange,
                    'tradingClass': d.tradingClass,
                    'currency': d.currency,
                    'minTick': str(d.minTick),
                    'sizeIncrement': str(d.sizeIncrement),
                    'multiplier': d.multiplier,
                    'contractMonth': d.contractMonth,
                    'timeZoneId': d.timeZoneId,
                    'tradingHours': d.tradingHours,
                    'liquidHours': d.liquidHours,
                    'longName': d.longName,
                    'marketName': d.marketName,
                    'evRule': d.evRule,
                    'evMultiplier': d.evMultiplier
                }

            with open(self._cache_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self._cache)} entries to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def clear_cache(self) -> None:
        """Clear the in-memory and file cache."""
        self._cache.clear()
        if self._cache_file and self._cache_file.exists():
            self._cache_file.unlink()
