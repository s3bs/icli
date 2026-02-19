"""Command: debug (!)

Category: Utilities
"""

import inspect
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger
from mutil.dispatch import DArg

from icli.cmds.base import IOp, command
from icli.engine.contracts import contractForName

if TYPE_CHECKING:
    pass


def build_context(op: IOp) -> dict:
    """Build the execution namespace for debug code.

    Returns a dict suitable as globals for eval()/exec().
    Full builtins are included -- this is a local debugging tool,
    not a sandbox.
    """
    state = op.state
    return {
        "__builtins__": __builtins__,
        # App state
        "state": state,
        "ib": op.ib,
        "cache": op.cache,
        "quotes": state.quoteState,
        "calc": state.calc,
        "idb": state.idb,
        # Helper functions
        "cfn": contractForName,
        "qualify": state.qualifier.qualify,
        # Collections
        "conIds": state.conIdCache,
        "quotesPositional": state.quotesPositional,
        # Account / positions
        "account": state.accountStatus,
        "pnl": state.pnlSingle,
        "positions": state.iposition,
        # Utilities
        "logger": logger,
    }


async def exec_debug(code: str, context: dict):
    """Execute user code: try eval first, fall back to exec.

    - If code is an expression, eval it and print the result.
    - If eval raises SyntaxError (code is a statement), exec it instead.
    - If the result of eval/exec is awaitable, await it automatically.
    - If code contains 'await' keyword, wrap in async function first.
    """
    # Handle 'await' keyword by wrapping in async function.
    # Use word boundary match to avoid false positives like "awaitable = True".
    if re.search(r"\bawait\b", code):
        # Wrap code in async function - try return for expressions, exec for statements
        try:
            # First try to evaluate as expression with await
            wrapper_code = f"async def __debug_async_wrapper__():\n    return {code}\n"
            compile(wrapper_code, "<string>", "exec")
            exec(wrapper_code, context)
            result = await context["__debug_async_wrapper__"]()
            if result is not None:
                logger.info("-> {}", result)
        except SyntaxError:
            # It's a statement, not an expression - exec it directly
            wrapper_code = f"async def __debug_async_wrapper__():\n    {code}\n"
            compile(wrapper_code, "<string>", "exec")
            exec(wrapper_code, context)
            await context["__debug_async_wrapper__"]()
        return

    try:
        result = eval(code, context)
    except SyntaxError:
        # Statement, not expression -- use exec
        exec(code, context)
        return

    # Auto-await coroutines / awaitables
    if inspect.isawaitable(result):
        result = await result

    if result is not None:
        logger.info("-> {}", result)


@command(names=["debug"])
@dataclass
class IOpDebug(IOp):
    """Execute Python code with shortcuts to live app objects.

    Examples:
        debug ib.isConnected()
        debug cfn("ES").secType
        debug await qualify(cfn("ES"))
        debug [k for k in quotes.keys()]
    """

    code: list[str] = field(init=False)

    def argmap(self) -> list[DArg]:
        return [DArg("*code", desc="Python expression or statement to execute")]

    async def run(self):
        code_str = (self.oargs__ or "").strip()
        if not code_str:
            logger.error("No code provided. Usage: debug <python expression>")
            return

        context = build_context(self)
        try:
            await exec_debug(code_str, context)
        except Exception as e:
            logger.error("Debug error: {}", e)
            if self.state.localvars.get("bigerror"):
                logger.exception("Full traceback:")
