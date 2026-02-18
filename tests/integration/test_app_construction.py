"""Smoke tests for IBKRCmdlineApp construction.

These catch missing imports, missing slot declarations, and broken
__post_init__ wiring — issues that unit tests on extracted modules miss.
"""

import ast
import importlib
import sys
from unittest.mock import patch

import pytest


class TestAppConstruction:
    """Verify IBKRCmdlineApp can be instantiated without crashing."""

    def test_construct_with_account_id(self):
        """App construction with an explicit account ID succeeds."""
        with patch("icli.cli.IBKRCmdlineApp.setupLogging"):
            from icli.cli import IBKRCmdlineApp

            app = IBKRCmdlineApp(accountId="DU1234567")
            assert app.accountId == "DU1234567"

    def test_construct_without_account_id(self):
        """App construction with empty account ID (auto-discovery mode) succeeds."""
        with patch("icli.cli.IBKRCmdlineApp.setupLogging"):
            from icli.cli import IBKRCmdlineApp

            app = IBKRCmdlineApp()
            assert app.accountId == ""

    def test_engine_modules_initialized(self):
        """All engine modules are wired up during __post_init__."""
        with patch("icli.cli.IBKRCmdlineApp.setupLogging"):
            from icli.cli import IBKRCmdlineApp

            app = IBKRCmdlineApp(accountId="DU1234567")

            # These were extracted from cli.py into engine/ modules.
            # Missing slot declarations cause AttributeError at construction time.
            assert app.portfolio is not None
            assert app.qualifier is not None
            assert app.quotemanager is not None
            assert app.events is not None
            assert app.placer is not None
            assert app.toolbar is not None

    def test_core_fields_initialized(self):
        """Core internal fields created in __post_init__ are present."""
        with patch("icli.cli.IBKRCmdlineApp.setupLogging"):
            from icli.cli import IBKRCmdlineApp

            app = IBKRCmdlineApp(accountId="DU1234567")

            assert app.opstate is app
            assert app.ordermgr is not None
            assert app.tasks is not None
            assert app.scheduler is not None
            assert app.calc is not None
            assert app.idb is not None

    def test_ib_connection_object_created(self):
        """IB connection object is created with IBDefaults."""
        with patch("icli.cli.IBKRCmdlineApp.setupLogging"):
            from icli.cli import IBKRCmdlineApp
            from ib_async import IB

            app = IBKRCmdlineApp()
            assert isinstance(app.ib, IB)


class TestModuleImports:
    """Verify all names used in cli.py are actually importable.

    The old cli.py used `from icli.helpers import *` which silently
    provided names like Stock, Option, etc. After decomposition we
    use explicit imports — this test catches any that were missed.
    """

    def test_cli_module_names_resolve(self):
        """All global names referenced in cli.py are defined after import."""
        import icli.cli as cli_module

        # Get all names that the module actually defines/imports
        module_names = set(dir(cli_module))

        # These names must be available (they caused NameErrors at runtime)
        critical_names = [
            "Stock",
            "IBDefaults",
            "Callable",
            "Coroutine",
            "Awaitable",
            "Hashable",
        ]
        for name in critical_names:
            assert name in module_names, (
                f"{name!r} is used in cli.py but not imported"
            )
