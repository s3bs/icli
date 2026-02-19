"""Command autocompletion for the IBKR REPL.

Provides tab-cycling completion for command names (with docstring descriptions)
and context-aware argument completion (symbols, quote groups, order IDs, etc.).
"""

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document


class CommandCompleter(Completer):
    """Completer for IBKR CLI commands and their arguments."""

    # Fixed clear targets from icli/cmds/utilities/clear.py
    _CLEAR_TARGETS = [
        "highlow",
        "pnl",
        "noquote",
        "opt",
        "options",
        "exp",
        "expired",
        "unused",
    ]

    # Map resolved command names to argument completer method names
    _ARG_COMPLETERS = {
        # Symbols (most common)
        "positions": "_complete_symbols",
        "ls": "_complete_symbols",
        "add": "_complete_symbols",
        "remove": "_complete_symbols",
        "rm": "_complete_symbols",
        "qquote": "_complete_symbols",
        "orders": "_complete_symbols",
        "chains": "_complete_symbols",
        # Quote groups
        "qrestore": "_complete_quote_groups",
        "qremove": "_complete_quote_groups",
        "qadd": "_complete_quote_groups",
        "qlist": "_complete_quote_groups",
        # Order IDs
        "cancel": "_complete_order_ids",
        # Fixed enums
        "clear": "_complete_clear_targets",
        # Themes
        "colorset": "_complete_colorset_themes",
        # Config keys
        "set": "_complete_set_keys",
        "unset": "_complete_set_keys",
        # Balance fields
        "balance": "_complete_balance_fields",
    }

    def __init__(self, app):
        """app is the IBKRCmdlineApp instance."""
        self.app = app

        # Build full command name -> class mapping from opcodes (not totalOps,
        # which expands every unambiguous prefix like adv/advi/advic/advice).
        self._fullname_to_cls: dict[str, type] = {}
        for val in app.dispatch.d.opcodes.values():
            if isinstance(val, dict):
                self._fullname_to_cls.update(val)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Handle multi-command: find text after the last unquoted ";"
        last_semi = text.rfind(";")
        if last_semi >= 0:
            segment = text[last_semi + 1 :].lstrip()
        else:
            segment = text

        # Split into command and rest
        parts = segment.split(None, 1)  # split on first whitespace
        if len(parts) <= 1 and not segment.endswith(" "):
            # Still typing the command name
            prefix = parts[0] if parts else ""
            yield from self._complete_command_name(prefix)
        else:
            # Typing arguments
            cmd_name = parts[0]
            arg_text = parts[1] if len(parts) > 1 else ""
            yield from self._complete_arguments(cmd_name, arg_text)

    def _complete_command_name(self, prefix):
        prefix_lower = prefix.lower()
        for cmd_name in sorted(self._fullname_to_cls):
            if cmd_name.lower().startswith(prefix_lower):
                cls = self._fullname_to_cls[cmd_name]
                # Extract first line of docstring as description
                doc = ""
                if cls.__doc__:
                    first_line = cls.__doc__.strip().split("\n")[0]
                    if "state: Any = None" not in first_line:
                        doc = first_line
                yield Completion(
                    cmd_name,
                    start_position=-len(prefix),
                    display_meta=doc,
                )

    def _resolve_command(self, typed):
        """Resolve a possibly-abbreviated command to its full name."""
        totalOps = self.app.dispatch.d.totalOps
        typed_lower = typed.lower()
        if typed_lower in totalOps:
            return typed_lower
        # Try prefix match (same logic as mutil.dispatch)
        matches = [name for name in totalOps if name.startswith(typed_lower)]
        if len(matches) == 1:
            return matches[0]
        return typed_lower  # unresolved, return as-is

    def _complete_arguments(self, cmd_name, arg_text):
        resolved = self._resolve_command(cmd_name)
        method_name = self._ARG_COMPLETERS.get(resolved)
        if method_name:
            method = getattr(self, method_name)
            # Get the word being typed (last whitespace-separated token)
            words = arg_text.split()
            current_word = words[-1] if words and not arg_text.endswith(" ") else ""
            yield from method(current_word, arg_text=arg_text)

    def _complete_symbols(self, prefix, **kwargs):
        symbols = set()

        # Live quote symbols
        symbols.update(self.app.quoteState.keys())

        # Portfolio position symbols
        try:
            for pos in self.app.ib.positions():
                symbols.add(pos.contract.localSymbol or pos.contract.symbol)
        except Exception:
            pass

        # Positional references
        for i, (sym, _) in enumerate(self.app.quotesPositional):
            ref = f":{i}"
            if ref.startswith(prefix) or sym.lower().startswith(prefix.lower()):
                yield Completion(ref, start_position=-len(prefix), display_meta=sym)

        prefix_upper = prefix.upper()
        for sym in sorted(symbols):
            if sym.upper().startswith(prefix_upper):
                yield Completion(sym, start_position=-len(prefix))

    def _complete_quote_groups(self, prefix, **kwargs):
        try:
            for key in self.app.cache:
                if isinstance(key, tuple) and len(key) == 2 and key[0] == "quotes":
                    group = key[1]
                    if group.lower().startswith(prefix.lower()):
                        yield Completion(group, start_position=-len(prefix))
        except Exception:
            pass

    def _complete_order_ids(self, prefix, **kwargs):
        try:
            for trade in self.app.ib.openTrades():
                oid = str(trade.order.orderId)
                sym = trade.contract.localSymbol or trade.contract.symbol
                if oid.startswith(prefix):
                    yield Completion(
                        oid, start_position=-len(prefix), display_meta=sym
                    )
        except Exception:
            pass

    def _complete_clear_targets(self, prefix, **kwargs):
        for target in self._CLEAR_TARGETS:
            if target.startswith(prefix.lower()):
                yield Completion(target, start_position=-len(prefix))

    _COLORSET_THEMES = {
        "default": "terminal default colors (reversed white bg)",
        "solar1": "Solarized dark (fg:#002B36 bg:#839496)",
    }

    # Known value sets for specific localvar keys
    _SET_VALUE_COMPLETIONS = {
        "loglevel": ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR"],
        "altrow_color": ["off", "c0c0c0", "003845"],
    }

    def _complete_colorset_themes(self, prefix, **kwargs):
        for name, desc in self._COLORSET_THEMES.items():
            if name.startswith(prefix.lower()):
                yield Completion(name, start_position=-len(prefix), display_meta=desc)

    def _complete_set_keys(self, prefix, arg_text=""):
        words = arg_text.split()
        # Second argument position: complete known values for the key
        if len(words) >= 2 or (len(words) == 1 and arg_text.endswith(" ")):
            key = words[0] if words else ""
            values = self._SET_VALUE_COMPLETIONS.get(key.lower())
            if values:
                for val in values:
                    if val.lower().startswith(prefix.lower()):
                        yield Completion(val, start_position=-len(prefix))
            return

        # First argument position: complete key names + special subcommands
        if "info".startswith(prefix.lower()):
            yield Completion("info", start_position=-len(prefix), display_meta="Show ICLI_ environment variables")

        for key in sorted(self.app.localvars.keys()):
            if key.lower().startswith(prefix.lower()):
                val = self.app.localvars[key]
                yield Completion(
                    key, start_position=-len(prefix), display_meta=str(val)
                )

    def _complete_balance_fields(self, prefix, **kwargs):
        for key in sorted(self.app.summary.keys()):
            if key.lower().startswith(prefix.lower()):
                yield Completion(key, start_position=-len(prefix))
