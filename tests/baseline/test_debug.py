"""Tests for the debug command (icli/cmds/utilities/debug.py).

Tests the standalone functions (build_context, exec_debug) directly.
Does not require a live IB connection or full app construction.
"""

import types
from io import StringIO
from unittest.mock import MagicMock

import pytest
from loguru import logger

from icli.cmds.utilities.debug import build_context, exec_debug


# ── Fixtures ────────────────────────────────────────────────────────────────


class FakeState:
    """Minimal stand-in for IBKRCmdlineApp with the attributes build_context reads."""

    def __init__(self):
        self.ib = MagicMock(name="ib")
        self.cache = {}
        self.quoteState = {"AAPL": "ticker_aapl"}
        self.calc = MagicMock(name="calc")
        self.idb = MagicMock(name="idb")
        self.qualifier = MagicMock()
        self.qualifier.qualify = MagicMock(name="qualify")
        self.conIdCache = {123: "contract_123"}
        self.quotesPositional = [("AAPL", "ticker_aapl")]
        self.accountStatus = {"BuyingPower4": 125_000.0}
        self.pnlSingle = {}
        self.iposition = {}
        self.localvars = {}


class FakeOp:
    """Minimal stand-in for IOp with state shortcuts."""

    def __init__(self, state=None):
        self.state = state or FakeState()
        self.ib = self.state.ib
        self.cache = self.state.cache


@pytest.fixture
def fake_op():
    return FakeOp()


@pytest.fixture
def context(fake_op):
    return build_context(fake_op)


@pytest.fixture
def log_capture():
    """Capture loguru output for assertion. Yields a StringIO buffer."""
    buf = StringIO()
    handler_id = logger.add(buf, format="{message}", level="INFO")
    yield buf
    logger.remove(handler_id)


# ── build_context ───────────────────────────────────────────────────────────


class TestBuildContext:
    """Verify that build_context exposes the right keys and values."""

    def test_has_builtins(self, context):
        assert "__builtins__" in context

    def test_state_is_app(self, fake_op, context):
        assert context["state"] is fake_op.state

    def test_ib_shortcut(self, fake_op, context):
        assert context["ib"] is fake_op.ib

    def test_cache_shortcut(self, fake_op, context):
        assert context["cache"] is fake_op.cache

    def test_quotes_shortcut(self, fake_op, context):
        assert context["quotes"] is fake_op.state.quoteState

    def test_cfn_shortcut(self, context):
        from icli.engine.contracts import contractForName

        assert context["cfn"] is contractForName

    def test_qualify_shortcut(self, fake_op, context):
        assert context["qualify"] is fake_op.state.qualifier.qualify

    def test_account_shortcut(self, fake_op, context):
        assert context["account"] is fake_op.state.accountStatus

    def test_positions_shortcut(self, fake_op, context):
        assert context["positions"] is fake_op.state.iposition

    def test_all_expected_keys_present(self, context):
        expected = {
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
        assert expected == set(context.keys())


# ── exec_debug: eval path (expressions) ────────────────────────────────────


class TestExecDebugEval:
    """Test the eval path -- expressions that return a value."""

    @pytest.mark.asyncio
    async def test_simple_arithmetic(self, context):
        """Numeric expression evaluates and returns without error."""
        # exec_debug logs the result; we just verify no exception
        await exec_debug("2 + 3", context)

    @pytest.mark.asyncio
    async def test_string_expression(self, context):
        await exec_debug("'hello'.upper()", context)

    @pytest.mark.asyncio
    async def test_none_result_no_output(self, context):
        """Expression returning None should not raise."""
        await exec_debug("None", context)

    @pytest.mark.asyncio
    async def test_context_variable_access(self, context):
        """Can access pre-populated context variables."""
        await exec_debug("account.get('BuyingPower4')", context)

    @pytest.mark.asyncio
    async def test_list_comprehension(self, context):
        await exec_debug("[k for k in quotes.keys()]", context)

    @pytest.mark.asyncio
    async def test_builtin_functions_available(self, context):
        """len(), type(), etc. work because full builtins are provided."""
        await exec_debug("len([1,2,3])", context)

    @pytest.mark.asyncio
    async def test_cfn_shortcut_callable(self, context):
        """cfn() shortcut is available and callable."""
        await exec_debug('cfn("AAPL")', context)


# ── exec_debug: exec path (statements) ─────────────────────────────────────


class TestExecDebugExec:
    """Test the exec fallback -- statements that don't return a value."""

    @pytest.mark.asyncio
    async def test_assignment(self, context):
        """Assignment is a statement, handled by exec path."""
        await exec_debug("x = 42", context)
        assert context["x"] == 42

    @pytest.mark.asyncio
    async def test_import_statement(self, context):
        await exec_debug("import math", context)
        assert "math" in context

    @pytest.mark.asyncio
    async def test_print_statement(self, context, capsys):
        await exec_debug("print('hello debug')", context)
        assert "hello debug" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_for_loop(self, context):
        await exec_debug("total = sum(range(5))", context)
        assert context["total"] == 10

    @pytest.mark.asyncio
    async def test_multistatement_semicolons(self, context):
        """Multiple statements separated by semicolons (single line)."""
        await exec_debug("a = 1; b = 2; c = a + b", context)
        assert context["c"] == 3


# ── exec_debug: async/await path ───────────────────────────────────────────


class TestExecDebugAsync:
    """Test auto-awaiting of coroutines."""

    @pytest.mark.asyncio
    async def test_awaitable_result_auto_awaited(self, context):
        """If eval returns a coroutine, exec_debug awaits it."""

        async def async_add(a, b):
            return a + b

        context["async_add"] = async_add
        # This returns a coroutine from eval(); exec_debug should await it
        await exec_debug("async_add(3, 4)", context)

    @pytest.mark.asyncio
    async def test_qualify_mock_returns_coroutine(self, context):
        """Simulates the real use case: qualify() returns a coroutine."""

        async def fake_qualify(*contracts):
            return contracts

        context["qualify"] = fake_qualify
        # eval("qualify(...)") returns coroutine -> auto-awaited
        await exec_debug('qualify("ES")', context)

    @pytest.mark.asyncio
    async def test_await_keyword_wrapped_in_async(self, context):
        """Explicit 'await' keyword is wrapped in async function and returns output."""

        async def fake_qualify(*contracts):
            return contracts

        context["qualify"] = fake_qualify
        # User types 'await qualify(...)' - should work without SyntaxError
        # and should print the result
        await exec_debug('await qualify("ES")', context)

    @pytest.mark.asyncio
    async def test_await_statement_executed(self, context):
        """Statement with await executes and wrapper completes without error."""

        results = []

        async def fake_qualify(*contracts):
            results.append(contracts)
            return contracts

        context["qualify"] = fake_qualify
        context["results"] = results
        # The await wrapper runs in its own local scope, so we verify
        # execution via a side effect (appending to a shared list).
        await exec_debug('await qualify("ES")', context)
        assert len(results) == 1
        assert results[0] == ("ES",)

    @pytest.mark.asyncio
    async def test_await_false_positive_uses_normal_path(self, context):
        """Code containing 'await' as substring (not keyword) uses normal exec path.

        'awaitable' contains 'await' as a substring but is not the await keyword.
        With word-boundary matching, this should use the normal exec path and
        set the variable in the context dict.
        """
        await exec_debug("awaitable = True", context)
        assert context["awaitable"] is True

    @pytest.mark.asyncio
    async def test_await_in_string_literal_still_works(self, context):
        """Code with 'await' inside a string literal still executes correctly.

        The word-boundary regex can't distinguish keyword vs string content
        without a full tokenizer, so this takes the async wrapper path.
        The wrapper still executes the code correctly -- just via a
        different internal path.
        """
        await exec_debug("msg = 'please await further instructions'", context)
        # Variable is set in wrapper's local scope, not outer context,
        # but the code runs without error.  Verify via side effect instead.
        results = []
        context["results"] = results
        await exec_debug("results.append('please await further instructions')", context)
        assert results == ["please await further instructions"]


# ── exec_debug: output verification ─────────────────────────────────────────


class TestExecDebugOutput:
    """Verify that exec_debug logs results via logger.info."""

    @pytest.mark.asyncio
    async def test_eval_result_logged(self, context, log_capture):
        """Non-None eval result is logged via logger.info('-> {}', result)."""
        await exec_debug("2 + 3", context)
        assert "-> 5" in log_capture.getvalue()

    @pytest.mark.asyncio
    async def test_none_result_not_logged(self, context, log_capture):
        """None result produces no '-> ' output."""
        await exec_debug("None", context)
        assert "-> " not in log_capture.getvalue()

    @pytest.mark.asyncio
    async def test_await_expression_result_logged(self, context, log_capture):
        """Awaited expression result is logged."""

        async def async_add(a, b):
            return a + b

        context["async_add"] = async_add
        await exec_debug("async_add(3, 4)", context)
        assert "-> 7" in log_capture.getvalue()


# ── exec_debug: error handling ──────────────────────────────────────────────


class TestExecDebugErrors:
    """Errors propagate as exceptions (the command's run() catches them)."""

    @pytest.mark.asyncio
    async def test_name_error_propagates(self, context):
        with pytest.raises(NameError):
            await exec_debug("undefined_variable_xyz", context)

    @pytest.mark.asyncio
    async def test_type_error_propagates(self, context):
        with pytest.raises(TypeError):
            await exec_debug("len(42)", context)

    @pytest.mark.asyncio
    async def test_syntax_error_in_exec_propagates(self, context):
        """Completely broken syntax fails in both eval and exec."""
        with pytest.raises(SyntaxError):
            await exec_debug("def", context)

    @pytest.mark.asyncio
    async def test_zero_division_propagates(self, context):
        with pytest.raises(ZeroDivisionError):
            await exec_debug("1 / 0", context)


# ── cli.py prefix routing (characterization) ───────────────────────────────


class TestBangPrefixRewriting:
    """Test the `!` -> `debug` rewriting logic from cli.py.

    These test the string transformation in isolation, not the full
    dispatch pipeline. The logic under test is:

        if ccmd[0] == "!":
            ccmd = f"debug {ccmd[1:].lstrip()}"
    """

    @staticmethod
    def rewrite(ccmd: str) -> str:
        """Replicate the prefix rewriting logic from cli.py."""
        if ccmd[0] == "!":
            ccmd = f"debug {ccmd[1:].lstrip()}"
        return ccmd

    def test_bang_space_expr(self):
        assert self.rewrite("! ib.isConnected()") == "debug ib.isConnected()"

    def test_bang_no_space(self):
        assert self.rewrite("!ib.isConnected()") == "debug ib.isConnected()"

    def test_bang_with_extra_spaces(self):
        assert self.rewrite("!   2 + 3") == "debug 2 + 3"

    def test_bang_arithmetic(self):
        assert self.rewrite("! 6891.50 * 5") == "debug 6891.50 * 5"

    def test_bang_with_quotes(self):
        assert self.rewrite('! cfn("ES").secType') == 'debug cfn("ES").secType'

    def test_bare_bang(self):
        """Bare `!` with nothing after it becomes `debug ` (empty code)."""
        assert self.rewrite("!") == "debug "

    def test_non_bang_unchanged(self):
        """Non-! input is not rewritten."""
        assert self.rewrite("positions") == "positions"
