"""Integration tests for the debug command.

These tests verify that debug.py works with a real IBKRCmdlineApp instance,
catching mock-reality mismatches that unit tests with FakeState miss.

Critical catches:
- Attributes that exist on FakeState but not IBKRCmdlineApp
- Wrong attribute paths (e.g., state.qualify vs state.qualifier.qualify)
- Missing side effects from __post_init__
"""

from unittest.mock import patch
import pytest

from icli.cmds.utilities.debug import build_context, exec_debug
from icli.cmds.base import IOp


class FakeOp:
    """Minimal IOp wrapper for testing."""

    def __init__(self, state):
        self.state = state
        self.ib = state.ib
        self.cache = state.cache


class TestDebugWithRealApp:
    """Test debug command with real IBKRCmdlineApp state."""

    @pytest.fixture
    def app_state(self):
        """Real IBKRCmdlineApp with patched external dependencies."""
        with patch("icli.cli.IBKRCmdlineApp.setupLogging"):
            from icli.cli import IBKRCmdlineApp

            return IBKRCmdlineApp(accountId="DU1234567")

    @pytest.fixture
    def op(self, app_state):
        """IOp wrapper around real app state."""
        return FakeOp(app_state)

    @pytest.fixture
    def context(self, op):
        """Context built from real app state."""
        return build_context(op)

    def test_context_builds_without_attribute_error(self, context):
        """build_context() succeeds with real IBKRCmdlineApp.

        This would fail if context references non-existent attributes.
        """
        expected_keys = {
            "__builtins__",
            "state",
            "ib",
            "cache",
            "quotes",
            "calc",
            "idb",
            "cfn",
            "qualify",
            "conIds",
            "quotesPositional",
            "account",
            "pnl",
            "positions",
            "logger",
        }
        assert set(context.keys()) == expected_keys

    def test_cfn_is_real_function(self, context):
        """cfn shortcut points to real contractForName function.

        Would fail if context used state.contractForName (doesn't exist).
        """
        from icli.engine.contracts import contractForName

        assert context["cfn"] is contractForName
        assert callable(context["cfn"])

    def test_qualify_is_real_method(self, context, app_state):
        """qualify shortcut points to real app.qualifier.qualify.

        Would fail if context used state.qualify (doesn't exist).
        """
        assert hasattr(app_state, "qualifier")
        assert hasattr(app_state.qualifier, "qualify")
        # Both are bound methods from the same qualifier instance
        assert callable(context["qualify"])
        assert context["qualify"].__self__ is app_state.qualifier
        assert context["qualify"].__name__ == "qualify"

    def test_all_context_values_are_accessible(self, context, app_state):
        """All values in context are valid and accessible."""
        # These should not raise AttributeError
        assert context["state"] is app_state
        assert context["ib"] is app_state.ib
        assert context["cache"] is app_state.cache
        assert context["quotes"] is app_state.quoteState
        assert context["calc"] is app_state.calc
        assert context["idb"] is app_state.idb
        assert context["conIds"] is app_state.conIdCache
        assert context["quotesPositional"] is app_state.quotesPositional
        assert context["account"] is app_state.accountStatus
        assert context["pnl"] is app_state.pnlSingle
        assert context["positions"] is app_state.iposition

    @pytest.mark.asyncio
    async def test_exec_debug_with_real_context(self, context):
        """exec_debug() runs successfully with real context."""
        # Simple expression
        await exec_debug("2 + 2", context)

        # Context access
        await exec_debug("len(account)", context)

        # cfn shortcut
        await exec_debug("cfn('AAPL')", context)

        # Builtin function
        await exec_debug("type(ib)", context)

    def test_required_app_attributes_exist(self, app_state):
        """All attributes used by build_context exist on real app.

        This is a safety check - if any of these are missing,
        build_context will crash at runtime.
        """
        required_attrs = [
            "ib",
            "cache",
            "quoteState",
            "calc",
            "idb",
            "qualifier",
            "conIdCache",
            "quotesPositional",
            "accountStatus",
            "pnlSingle",
            "iposition",
        ]
        for attr in required_attrs:
            assert hasattr(app_state, attr), f"Missing required attribute: {attr}"

        # Nested qualifier.qualify
        assert hasattr(app_state.qualifier, "qualify"), "Missing qualifier.qualify method"
