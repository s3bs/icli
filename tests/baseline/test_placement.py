"""Tests for icli.engine.placement — order placement & pricing.

Focus: business logic in OrderPlacer that can be tested without a live IBKR connection.
"""
import math
import pytest
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock, patch

ib_async = pytest.importorskip("ib_async")

from ib_async import Contract, Bag, Order, Stock, ComboLeg
from icli.engine.placement import OrderPlacer


def make_placer(**overrides):
    """Create an OrderPlacer with sensible mock defaults."""
    defaults = dict(
        ib=MagicMock(),
        conIdCache={},
        idb=MagicMock(),
        app=MagicMock(),
    )
    defaults.update(overrides)
    return OrderPlacer(
        ib=defaults["ib"],
        conIdCache=defaults["conIdCache"],
        idb=defaults["idb"],
        app=defaults["app"],
    )


# ---------------------------------------------------------------------------
# Tick compliance
# ---------------------------------------------------------------------------

class TestComply:
    async def test_delegates_to_idb_round(self):
        idb = MagicMock()
        idb.round = AsyncMock(return_value=Decimal("150.01"))
        placer = make_placer(idb=idb)
        from icli import instrumentdb
        result = await placer.comply(MagicMock(), Decimal("150.013"), instrumentdb.ROUND.NEAR)
        idb.round.assert_called_once()
        assert result == Decimal("150.01")

    @pytest.mark.parametrize("method_name,expected_direction", [
        ("complyNear", "NEAR"),
        ("complyUp", "UP"),
        ("complyDown", "DOWN"),
    ])
    async def test_comply_direction_routing(self, method_name, expected_direction):
        idb = MagicMock()
        idb.round = AsyncMock(return_value=Decimal("5.00"))
        placer = make_placer(idb=idb)
        method = getattr(placer, method_name)
        await method(MagicMock(), Decimal("4.999"))
        from icli import instrumentdb
        _, _, called_direction = idb.round.call_args[0]
        assert called_direction == getattr(instrumentdb.ROUND, expected_direction)

    async def test_tick_increment_delegates_to_comply_up_near_zero(self):
        """tickIncrement returns the minimum valid price above zero — i.e. complyUp(0.00001)."""
        idb = MagicMock()
        idb.round = AsyncMock(return_value=Decimal("0.01"))
        placer = make_placer(idb=idb)
        result = await placer.tickIncrement(MagicMock())
        # complyUp is called with near-zero seed
        called_price = idb.round.call_args[0][1]
        assert called_price == Decimal("0.00001")
        assert result == Decimal("0.01")


# ---------------------------------------------------------------------------
# Safe order modification
# ---------------------------------------------------------------------------

class TestSafeModify:
    @pytest.mark.parametrize("input_type,expected_type", [
        ("IBALGO", "LMT"),
        ("LMT", "LMT"),
    ])
    async def test_order_type_normalization(self, input_type, expected_type):
        idb = MagicMock()
        idb.round = AsyncMock(side_effect=lambda c, p, d: p)
        placer = make_placer(idb=idb)
        order = Order(orderType=input_type, action="BUY", lmtPrice=None, auxPrice=None,
                      parentId=0, transmit=True, volatility=None)
        result = await placer.safeModify(MagicMock(), order)
        assert result.orderType == expected_type

    async def test_clears_parent_id(self):
        idb = MagicMock()
        idb.round = AsyncMock(side_effect=lambda c, p, d: p)
        placer = make_placer(idb=idb)
        order = Order(orderType="LMT", action="BUY", lmtPrice=None, auxPrice=None,
                      parentId=999, transmit=False, volatility=None)
        result = await placer.safeModify(MagicMock(), order)
        assert result.parentId == 0

    async def test_always_sets_transmit_true(self):
        idb = MagicMock()
        idb.round = AsyncMock(side_effect=lambda c, p, d: p)
        placer = make_placer(idb=idb)
        order = Order(orderType="LMT", action="BUY", lmtPrice=None, auxPrice=None,
                      parentId=0, transmit=False, volatility=None)
        result = await placer.safeModify(MagicMock(), order)
        assert result.transmit is True

    @pytest.mark.parametrize("order_type,input_vol,expected_vol", [
        ("LMT", 0.25, None),
        ("VOL", 0.25, 0.25),
    ])
    async def test_volatility_handling(self, order_type, input_vol, expected_vol):
        idb = MagicMock()
        idb.round = AsyncMock(side_effect=lambda c, p, d: p)
        placer = make_placer(idb=idb)
        order = Order(orderType=order_type, action="BUY", lmtPrice=None, auxPrice=None,
                      parentId=0, transmit=True, volatility=input_vol)
        result = await placer.safeModify(MagicMock(), order)
        assert result.volatility == expected_vol

    async def test_applies_kwargs_overrides(self):
        idb = MagicMock()
        idb.round = AsyncMock(side_effect=lambda c, p, d: p)
        placer = make_placer(idb=idb)
        order = Order(orderType="LMT", action="BUY", lmtPrice=None, auxPrice=None,
                      parentId=0, transmit=True, totalQuantity=10.0, volatility=None)
        result = await placer.safeModify(MagicMock(), order, totalQuantity=20.0)
        assert result.totalQuantity == 20.0


# ---------------------------------------------------------------------------
# Bracket order creation
# ---------------------------------------------------------------------------

class TestCreateBracketAttachParent:
    def test_assigns_order_id_if_missing(self):
        ib = MagicMock()
        ib.client.getReqId = MagicMock(side_effect=[100, 101])
        placer = make_placer(ib=ib)

        order = Order(orderId=0, transmit=True)
        profitOrder, lossOrder = placer.createBracketAttachParent(
            order,
            sideClose="SELL",
            qty=10.0,
            profitLimit=Decimal("155.00"),
            lossLimit=None,
            lossStopPrice=None,
            outsideRth=True,
            tif="GTC",
            orderTypeProfit="LMT",
            orderTypeLoss="STP",
        )
        assert order.orderId == 100
        assert profitOrder is not None
        assert lossOrder is None

    def test_sets_parent_id_on_profit_order(self):
        ib = MagicMock()
        ib.client.getReqId = MagicMock(side_effect=[100, 101])
        placer = make_placer(ib=ib)

        order = Order(orderId=0, transmit=True)
        profitOrder, _ = placer.createBracketAttachParent(
            order, "SELL", 10.0, Decimal("155.00"), None, None,
            True, "GTC", "LMT", "STP",
        )
        assert profitOrder.parentId == order.orderId

    def test_profit_order_transmits_when_no_loss_order(self):
        ib = MagicMock()
        ib.client.getReqId = MagicMock(side_effect=[100, 101])
        placer = make_placer(ib=ib)

        order = Order(orderId=0, transmit=True)
        profitOrder, lossOrder = placer.createBracketAttachParent(
            order, "SELL", 10.0, Decimal("155.00"), None, None,
            True, "GTC", "LMT", "STP",
        )
        assert lossOrder is None
        assert profitOrder.transmit is True

    def test_loss_order_transmits_last_when_both_exist(self):
        ib = MagicMock()
        ib.client.getReqId = MagicMock(side_effect=[100, 101, 102])
        placer = make_placer(ib=ib)

        order = Order(orderId=0, transmit=True)
        profitOrder, lossOrder = placer.createBracketAttachParent(
            order, "SELL", 10.0,
            profitLimit=Decimal("155.00"),
            lossLimit=Decimal("140.00"),
            lossStopPrice=Decimal("139.50"),
            outsideRth=True,
            tif="GTC",
            orderTypeProfit="LMT",
            orderTypeLoss="STP LMT",
        )
        assert profitOrder is not None
        assert lossOrder is not None
        # loss order always transmits last
        assert lossOrder.transmit is True
        # profit order should NOT transmit when loss order also exists
        assert profitOrder.transmit is False

    def test_both_none_when_no_profit_or_loss(self):
        ib = MagicMock()
        ib.client.getReqId = MagicMock(side_effect=[100])
        placer = make_placer(ib=ib)

        order = Order(orderId=0, transmit=True)
        profitOrder, lossOrder = placer.createBracketAttachParent(
            order, "SELL", 10.0, None, None, None,
            True, "GTC", "LMT", "STP",
        )
        assert profitOrder is None
        assert lossOrder is None


# ---------------------------------------------------------------------------
# Exit price discovery
# ---------------------------------------------------------------------------

class TestOrderPriceForContract:
    def test_long_position_finds_sell_order(self):
        """A long position (positionSize > 0) is closed by a SELL order."""
        ib = MagicMock()
        # Create a fake open trade
        fake_trade = MagicMock()
        fake_trade.order.action = "SELL"
        fake_trade.contract.localSymbol = "AAPL"
        fake_trade.order.lmtPrice = 150.00
        fake_trade.orderStatus.remaining = 10

        ib.openTrades.return_value = [fake_trade]
        placer = make_placer(ib=ib)

        contract = MagicMock()
        contract.localSymbol = "AAPL"

        result = placer.orderPriceForContract(contract, positionSize=10)
        # For LONG position (size > 0), closing credit is negative (credit received)
        # math.copysign(1, positionSize=10) * -1 * lmtPrice = 1 * -1 * 150 = -150
        assert result == -150.0

    def test_short_position_finds_buy_order(self):
        """A short position (positionSize < 0) is closed by a BUY order."""
        ib = MagicMock()
        fake_trade = MagicMock()
        fake_trade.order.action = "BUY"
        fake_trade.contract.localSymbol = "AAPL"
        fake_trade.order.lmtPrice = 148.00
        fake_trade.orderStatus.remaining = 5

        ib.openTrades.return_value = [fake_trade]
        placer = make_placer(ib=ib)

        contract = MagicMock()
        contract.localSymbol = "AAPL"

        result = placer.orderPriceForContract(contract, positionSize=-5)
        # For SHORT position (size < 0), closing debit is positive (debit paid)
        # math.copysign(1, -5) * -1 * 148 = -1 * -1 * 148 = 148
        assert result == 148.0

    def test_empty_list_when_no_matching_orders(self):
        ib = MagicMock()
        ib.openTrades.return_value = []
        placer = make_placer(ib=ib)

        contract = MagicMock()
        contract.localSymbol = "AAPL"

        result = placer.orderPriceForContract(contract, positionSize=10)
        assert result == []

    def test_multiple_partial_orders_sorted_by_price(self):
        """Multiple partial exit orders should be sorted by absolute price."""
        ib = MagicMock()

        def make_trade(price, remaining):
            t = MagicMock()
            t.order.action = "SELL"
            t.contract.localSymbol = "AAPL"
            t.order.lmtPrice = price
            t.orderStatus.remaining = remaining
            return t

        ib.openTrades.return_value = [
            make_trade(155.0, 3),
            make_trade(160.0, 7),
        ]
        placer = make_placer(ib=ib)

        contract = MagicMock()
        contract.localSymbol = "AAPL"

        result = placer.orderPriceForContract(contract, positionSize=10)
        # Should return a sorted list, not a single price (because len(ts) != 1 matching full position)
        assert isinstance(result, list)
        assert len(result) == 2


class TestOrderPriceForSpread:
    def test_empty_when_no_bag_trades(self):
        ib = MagicMock()
        ib.openTrades.return_value = []
        placer = make_placer(ib=ib)
        result = placer.orderPriceForSpread([MagicMock(conId=1)], positionSize=1)
        assert result == []

    def test_returns_price_when_single_matching_full_position(self):
        ib = MagicMock()

        fake_trade = MagicMock()
        fake_trade.contract = MagicMock(spec=Bag)
        # Ensure isinstance(t.contract, Bag) returns True
        fake_trade.contract.__class__ = Bag
        leg1 = MagicMock()
        leg1.conId = 1001
        leg2 = MagicMock()
        leg2.conId = 1002
        fake_trade.contract.comboLegs = [leg1, leg2]
        fake_trade.orderStatus.remaining = 5
        fake_trade.order.lmtPrice = 3.50

        ib.openTrades.return_value = [fake_trade]
        placer = make_placer(ib=ib)

        c1 = MagicMock()
        c1.conId = 1001
        c2 = MagicMock()
        c2.conId = 1002

        result = placer.orderPriceForSpread([c1, c2], positionSize=5)
        assert result == 3.50


