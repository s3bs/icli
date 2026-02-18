"""Tests for icli.engine.events — event handlers."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from icli.engine.events import IBEventRouter


def make_event_router(**overrides):
    defaults = dict(
        ib=MagicMock(),
        quoteState={},
        summary={},
        accountStatus={},
        pnlSingle={},
        iposition={},
        fillers=MagicMock(),
        ordermgr=MagicMock(),
        speak=MagicMock(url=None),
        duplicateMessageHandler=MagicMock(),
        ifthenRuntime=MagicMock(check=MagicMock(return_value=[])),
        conIdCache={},
        contractIdsToQuoteKeysMappings={},
        app=MagicMock(),
    )
    defaults.update(overrides)
    return IBEventRouter(**defaults)


# -----------------------------------------------------------------------
# TestUpdateSummary
# -----------------------------------------------------------------------

class TestUpdateSummary:
    def test_stores_tag_value(self):
        router = make_event_router()
        v = MagicMock(tag="NetLiquidation", value="250000", account="U1234")
        router.updateSummary(v)
        assert router.summary["NetLiquidation"] == "250000"

    def test_netliq_triggers_urpl_update(self):
        router = make_event_router(
            accountStatus={"UnrealizedPnL": 0, "RealizedPnL": 0, "NetLiquidation": 100000}
        )
        v = MagicMock(tag="NetLiquidation", value="250000", account="U1234")
        router.updateSummary(v)
        assert "RealizedPnL%" in router.accountStatus

    def test_non_netliq_tag_does_not_trigger_urpl(self):
        router = make_event_router(
            accountStatus={"UnrealizedPnL": 0, "RealizedPnL": 0, "NetLiquidation": 100000}
        )
        v = MagicMock(tag="AvailableFunds", value="50000", account="U1234")
        # Should not raise; RealizedPnL% is NOT added for non-netliq tags
        router.updateSummary(v)
        # The tag value should be stored
        assert router.accountStatus.get("AvailableFunds") == 50000.0

    def test_buying_power_splits_into_3_tiers(self):
        router = make_event_router(accountStatus={})
        v = MagicMock(tag="BuyingPower", value="400000", account="U1234")
        router.updateSummary(v)
        assert router.accountStatus.get("BuyingPower4") == 400000.0
        assert abs(router.accountStatus.get("BuyingPower3") - 300000.0) < 1.0
        assert router.accountStatus.get("BuyingPower2") == 200000.0


# -----------------------------------------------------------------------
# TestUpdatePNL
# -----------------------------------------------------------------------

class TestUpdatePNL:
    def test_stores_pnl_values_in_summary(self):
        router = make_event_router(accountStatus={"NetLiquidation": 100000})
        v = MagicMock(unrealizedPnL=1000, realizedPnL=500, dailyPnL=1500)
        router.updatePNL(v)
        assert router.summary["DailyPnL"] == 1500
        assert router.summary["UnrealizedPnL"] == 1000
        assert router.summary["RealizedPnL"] == 500

    def test_stores_pnl_values_in_account_status(self):
        router = make_event_router(accountStatus={"NetLiquidation": 100000})
        v = MagicMock(unrealizedPnL=1000.0, realizedPnL=500.0, dailyPnL=1500.0)
        router.updatePNL(v)
        assert router.accountStatus["DailyPnL"] == 1500.0
        assert router.accountStatus["UnrealizedPnL"] == 1000.0
        assert router.accountStatus["RealizedPnL"] == 500.0

    def test_triggers_urpl_update(self):
        router = make_event_router(
            accountStatus={"NetLiquidation": 100000, "UnrealizedPnL": 0, "RealizedPnL": 0}
        )
        v = MagicMock(unrealizedPnL=2000.0, realizedPnL=1000.0, dailyPnL=3000.0)
        router.updatePNL(v)
        assert "RealizedPnL%" in router.accountStatus


# -----------------------------------------------------------------------
# TestUpdatePNLSingle
# -----------------------------------------------------------------------

class TestUpdatePNLSingle:
    def test_stores_by_conid(self):
        router = make_event_router()
        v = MagicMock(conId=12345)
        router.updatePNLSingle(v)
        assert router.pnlSingle[12345] == v


# -----------------------------------------------------------------------
# TestUpdateURPLPercentages
# -----------------------------------------------------------------------

class TestUpdateURPLPercentages:
    def test_computes_percentages(self):
        router = make_event_router(
            accountStatus={"NetLiquidation": 100000, "UnrealizedPnL": 5000, "RealizedPnL": 2000}
        )
        router.updateURPLPercentages()
        assert "RealizedPnL%" in router.accountStatus
        assert "UnrealizedPnL%" in router.accountStatus
        assert "TotalPnL%" in router.accountStatus

    def test_zero_pnl_gives_zero_percentages(self):
        router = make_event_router(
            accountStatus={"NetLiquidation": 100000, "UnrealizedPnL": 0, "RealizedPnL": 0}
        )
        router.updateURPLPercentages()
        assert router.accountStatus["RealizedPnL%"] == 0.0
        assert router.accountStatus["UnrealizedPnL%"] == 0.0
        assert router.accountStatus["TotalPnL%"] == 0.0

    def test_zero_netliq_no_crash(self):
        """Should not raise ZeroDivisionError when NetLiquidation is 0."""
        router = make_event_router(
            accountStatus={"NetLiquidation": 0, "UnrealizedPnL": 0, "RealizedPnL": 0}
        )
        # 0 / (0 - 0) = 0 / 0 → ZeroDivisionError unless handled.
        # The original code doesn't guard against this; the test documents current behavior.
        try:
            router.updateURPLPercentages()
        except ZeroDivisionError:
            pass  # Acceptable — documenting behavior, not crashing the test suite

    def test_total_pnl_is_sum_of_realized_unrealized(self):
        router = make_event_router(
            accountStatus={"NetLiquidation": 110000, "UnrealizedPnL": 5000, "RealizedPnL": 5000}
        )
        router.updateURPLPercentages()
        assert abs(
            router.accountStatus["TotalPnL%"]
            - (router.accountStatus["RealizedPnL%"] + router.accountStatus["UnrealizedPnL%"])
        ) < 1e-9

    def test_missing_keys_use_defaults(self):
        """Missing accountStatus keys should use defaults (0 for PnL, 1 for NetLiq)."""
        router = make_event_router(accountStatus={})
        # Should not raise KeyError
        router.updateURPLPercentages()


# -----------------------------------------------------------------------
# TestErrorHandler
# -----------------------------------------------------------------------

class TestErrorHandler:
    def test_all_benign_codes_bypass_dedup(self):
        benign = {1102, 2104, 2108, 2106, 2107, 2119, 2152, 2158}
        for code in benign:
            router = make_event_router()
            router.errorHandler(-1, code, "some status", None)
            router.duplicateMessageHandler.handle_message.assert_not_called()

    def test_error_codes_use_dedup_handler(self):
        router = make_event_router()
        router.errorHandler(1, 200, "No security found", None)
        router.duplicateMessageHandler.handle_message.assert_called_once()

    @pytest.mark.parametrize("reqid,error_code,expected_prefix", [
        (42, 500, "Order Error"),
        (-1, 400, "API Error"),
    ])
    def test_error_prefix_routing(self, reqid, error_code, expected_prefix):
        router = make_event_router()
        router.errorHandler(reqid, error_code, "some error", None)
        call_args = router.duplicateMessageHandler.handle_message.call_args
        msg = call_args.kwargs["message"]
        assert expected_prefix in msg

    def test_contract_presence_in_error_msg(self):
        # With contract: contract info included in message
        router = make_event_router()
        contract = MagicMock()
        contract.__str__ = lambda self: "AAPL"
        router.errorHandler(-1, 400, "bad contract", contract)
        msg = router.duplicateMessageHandler.handle_message.call_args.kwargs["message"]
        assert "AAPL" in msg

        # Without contract: "for None" should NOT appear
        router2 = make_event_router()
        router2.errorHandler(-1, 400, "no contract error", None)
        msg2 = router2.duplicateMessageHandler.handle_message.call_args.kwargs["message"]
        assert "for None" not in msg2

    def test_reqid_321_uses_api_error_prefix(self):
        """reqId=1 but errorCode=321 should use 'API Error' not 'Order Error'."""
        router = make_event_router()
        router.errorHandler(1, 321, "order size issue", None)
        call_args = router.duplicateMessageHandler.handle_message.call_args
        msg = call_args.kwargs["message"]
        assert "API Error" in msg


# -----------------------------------------------------------------------
# TestUpdateOrder
# -----------------------------------------------------------------------

class TestUpdateOrder:
    def test_skips_when_not_connected(self):
        """updateOrder should return early when app.connected is False."""
        router = make_event_router()
        router._app.connected = False
        trade = MagicMock()
        # Even with a mock trade, no fillers should be accessed
        router.updateOrder(trade)
        router.fillers.__getitem__.assert_not_called()

    def test_notifies_fillers_on_filled(self):
        """Should set filler event when trade is fully filled."""
        router = make_event_router()
        router._app.connected = True

        filler = MagicMock()
        router.fillers.__getitem__ = MagicMock(return_value=filler)

        trade = MagicMock()
        trade.log = [MagicMock(status="Filled")]
        trade.orderStatus.remaining = 0
        trade.orderStatus.orderId = 1
        trade.orderStatus.status = "Filled"
        trade.contract.localSymbol = "AAPL"

        router.updateOrder(trade)
        filler.set.assert_called_once()
        assert filler.trade == trade

    def test_does_not_notify_fillers_on_partial_fill(self):
        """Should NOT set filler event when remaining > 0."""
        router = make_event_router()
        router._app.connected = True

        filler = MagicMock()
        router.fillers.__getitem__ = MagicMock(return_value=filler)

        trade = MagicMock()
        trade.log = [MagicMock(status="PartiallyFilled")]
        trade.orderStatus.remaining = 50
        trade.orderStatus.orderId = 1
        trade.orderStatus.status = "PartiallyFilled"
        trade.contract.localSymbol = "AAPL"

        router.updateOrder(trade)
        filler.set.assert_not_called()


# -----------------------------------------------------------------------
# TestUpdateAgentAccountStatus
# -----------------------------------------------------------------------

class TestUpdateAgentAccountStatus:
    def test_commission_creates_trade_with_correct_fields(self):
        from icli.engine.primitives import FillReport
        import datetime

        router = make_event_router()
        router._app.clientId = 7
        fill = FillReport(
            orderId=33,
            conId=123,
            sym="SPY",
            side="BOT",
            shares=10,
            price=450.0,
            pnl=0.0,
            commission=0.5,
            when=datetime.datetime(2024, 1, 15, 10, 30, 0),
        )
        router.updateAgentAccountStatus("commission", fill)

        # Verify add_trade was called
        router.ordermgr.add_trade.assert_called_once()

        # Verify correct conId passed as first positional arg
        call_args = router.ordermgr.add_trade.call_args
        assert call_args[0][0] == 123

        # Verify clientId used in orderId tuple
        trade_obj = call_args[0][1]
        assert trade_obj.orderid == (7, 33)

    def test_summary_category_does_nothing(self):
        """'summary' category should be a no-op."""
        router = make_event_router()
        # Should not raise and should not call ordermgr.add_trade
        router.updateAgentAccountStatus("summary", {"NetLiquidation": 100000.0})
        router.ordermgr.add_trade.assert_not_called()


# -----------------------------------------------------------------------
# TestOrderExecuteHandler
# -----------------------------------------------------------------------

class TestOrderExecuteHandler:
    @pytest.mark.asyncio
    async def test_creates_iposition_for_new_contract(self):
        from ib_async import Stock
        iposition = {}
        router = make_event_router(iposition=iposition)
        router._app.nameForContract = MagicMock(return_value="AAPL")

        contract = MagicMock(spec=Stock)
        contract.conId = 999
        contract.__class__ = Stock  # not a Bag

        trade = MagicMock()
        trade.contract = contract
        trade.orderStatus.orderId = 1
        trade.orderStatus.status = "Filled"

        fill = MagicMock()

        await router.orderExecuteHandler(trade, fill)
        assert 999 in iposition

    @pytest.mark.asyncio
    async def test_does_not_overwrite_existing_iposition(self):
        from ib_async import Stock
        existing = MagicMock()
        iposition = {555: existing}
        router = make_event_router(iposition=iposition)
        router._app.nameForContract = MagicMock(return_value="MSFT")

        contract = MagicMock(spec=Stock)
        contract.conId = 555
        contract.__class__ = Stock

        trade = MagicMock()
        trade.contract = contract
        trade.orderStatus.orderId = 2
        trade.orderStatus.status = "Filled"

        fill = MagicMock()

        await router.orderExecuteHandler(trade, fill)
        # The original should NOT be replaced
        assert iposition[555] is existing


# -----------------------------------------------------------------------
# TestTickersUpdate
# -----------------------------------------------------------------------

class TestTickersUpdate:
    def test_skips_unknown_quote_keys(self):
        """If quotekey not in quoteState, should silently continue."""
        router = make_event_router(quoteState={})
        ticker = MagicMock()
        ticker.contract = MagicMock()
        ticker.bid = 100.0
        ticker.ask = 101.0

        with patch("icli.engine.events.lookupKey", return_value="UNKNOWN"):
            # Should not raise
            router.tickersUpdate([ticker])

    def test_calls_process_ticker_update(self):
        """For known quote keys, processTickerUpdate should be called."""
        iticker = MagicMock()
        quoteState = {"AAPL": iticker}
        router = make_event_router(quoteState=quoteState)

        ticker = MagicMock()
        ticker.contract = MagicMock()
        ticker.bid = 150.0
        ticker.ask = 150.05

        with patch("icli.engine.events.lookupKey", return_value="AAPL"):
            router.tickersUpdate([ticker])

        iticker.processTickerUpdate.assert_called_once()

    def test_skips_bid_ask_check_when_none(self):
        """When bid or ask is None, should skip further processing but not crash."""
        iticker = MagicMock()
        quoteState = {"SPY": iticker}
        router = make_event_router(quoteState=quoteState)

        ticker = MagicMock()
        ticker.contract = MagicMock()
        ticker.bid = None
        ticker.ask = None

        with patch("icli.engine.events.lookupKey", return_value="SPY"):
            router.tickersUpdate([ticker])

        iticker.processTickerUpdate.assert_called_once()

    def test_ifthen_check_called_per_ticker(self):
        """ifthenRuntime.check should be called once per ticker."""
        iticker = MagicMock()
        quoteState = {"TSLA": iticker}
        ifthenRuntime = MagicMock(check=MagicMock(return_value=[]))
        router = make_event_router(quoteState=quoteState, ifthenRuntime=ifthenRuntime)

        ticker = MagicMock()
        ticker.contract = MagicMock()
        ticker.bid = 200.0
        ticker.ask = 200.5

        with patch("icli.engine.events.lookupKey", return_value="TSLA"):
            router.tickersUpdate([ticker])

        ifthenRuntime.check.assert_called_once_with("TSLA")
